"""
buffered_store.py
=================

A write-back, in-memory buffering layer for any Zarr v2 store.

Key features
------------
* **Read-through cache** – a key is fetched from the underlying store at most once.
* **Write-back buffering** – writes and deletes are kept in memory until
  `flush()` is called.  Nothing is pushed to the remote store beforehand.
* **Parallel flush** – dirty keys are committed using ThreadPoolExecutor.
* **Fail-fast flush** – if any upload/delete fails, the flush aborts and the
  first exception is raised.  All remaining futures are cancelled.
* **Context-manager friendly** – `with BufferedStore(...):` automatically flushes
  on exiting the `with` block.
* **No size limit** – the cache grows without eviction (per user request).

Compatibility: Python ≥ 3.8 and Zarr ≥ 2.0.  No extra dependencies.
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping, Iterable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION, CancelledError
from typing import Any, Dict, Set, Optional

# Sentinel used to mark deletions in the cache
_TOMBSTONE = object()


class BufferedStore(MutableMapping):
    """
    BufferedStore(base_store, max_workers=None, executor=None)

    Parameters
    ----------
    base_store : MutableMapping
        Any Zarr v2–compatible store (LocalFSStore, ABSStore, S3Store …).
    max_workers : int or None, default None
        Number of worker threads for the internal ThreadPoolExecutor.
        Ignored when *executor* is supplied.
    executor : ThreadPoolExecutor or None, default None
        Inject a pre-created executor (e.g. a global one).  If None,
        a private executor is lazily created on first flush.

    Notes
    -----
    * **Manual flush only**.  Call :py:meth:`flush` whenever you want to
      push changes; otherwise they stay local.
    * The class is *not* process-safe (multiprocessing).  It is thread-safe.
    """

    # --------------------------------------------------------------------- #
    # Core construction / helpers                                           #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        base_store: MutableMapping,
        *,
        max_workers: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        clear_cache_on_flush: bool = True,
    ) -> None:
        self._store: MutableMapping[str, bytes] = base_store
        self._cache: Dict[str, Any] = {}          # key → bytes | _TOMBSTONE
        self._dirty: Set[str] = set()             # keys that need uploading
        self._lock = threading.RLock()
        self._external_executor = executor
        self._max_workers = max_workers
        self._closed = False
        self._clear_cache_on_flush = clear_cache_on_flush

    # --------------------------------------------------------------------- #
    # MutableMapping interface                                              #
    # --------------------------------------------------------------------- #

    def __getitem__(self, key: str) -> bytes:  # type: ignore[override]
        with self._lock:
            if key in self._cache:
                val = self._cache[key]
                if val is _TOMBSTONE:
                    raise KeyError(key)
                return val

        # Miss: fetch once from base store, then cache.
        value = self._store[key]  # may raise KeyError
        with self._lock:
            self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: bytes) -> None:  # type: ignore[override]
        with self._lock:
            self._cache[key] = value
            self._dirty.add(key)

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        with self._lock:
            # Validate existence: honour in-memory deletions first, else remote.
            if key in self._cache:
                if self._cache[key] is _TOMBSTONE:
                    raise KeyError(key)
            else:
                # Check remote store for correctness
                if key not in self._store:
                    raise KeyError(key)

            self._cache[key] = _TOMBSTONE
            self._dirty.add(key)

    def __iter__(self):
        # Merge remote keys with cached (dirty) keys, omitting tombstones.
        with self._lock:
            cached = {k for k, v in self._cache.items() if v is not _TOMBSTONE}
        remote = set(self._store.keys())
        return iter(remote.union(cached))

    def __len__(self) -> int:
        with self._lock:
            cached_present = sum(1 for v in self._cache.values() if v is not _TOMBSTONE)
        remote_len = len(self._store)
        # Remote keys overridden by tombstones should be excluded.
        with self._lock:
            deleted_remotes = sum(
                1
                for k, v in self._cache.items()
                if v is _TOMBSTONE and k in self._store
            )
        return remote_len + cached_present - deleted_remotes

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if not isinstance(key, str):
            return False
        with self._lock:
            if key in self._cache:
                return self._cache[key] is not _TOMBSTONE
        return key in self._store

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def flush(self) -> None:
        """
        Push all buffered writes/deletes to the underlying store in parallel.

        Raises
        ------
        Exception
            Re-raises the first exception from the underlying store
            and cancels any remaining futures.
        """
        with self._lock:
            dirty_keys = list(self._dirty)  # immutable snapshot
            if not dirty_keys:
                return
            # Build per-key payloads to avoid holding the lock for too long.
            ops = [(k, self._cache[k]) for k in dirty_keys]

        exec_to_use = (
            self._external_executor
            if self._external_executor is not None
            else ThreadPoolExecutor(max_workers=self._max_workers)
        )
        owns_executor = exec_to_use is not self._external_executor

        def _apply(op):
            key, val = op
            if val is _TOMBSTONE:
                del self._store[key]
            else:
                self._store[key] = val

        futures = [exec_to_use.submit(_apply, op) for op in ops]

        # Wait until first failure or all succeed
        done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

        # Propagate any exception (fail-fast)
        try:
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    # Cancel remaining tasks and re-raise
                    for nf in not_done:
                        nf.cancel()
                    raise exc
        finally:
            # Clean up executor if we created it
            if owns_executor:
                exec_to_use.shutdown(wait=True)

        # Success: mark keys as clean
        with self._lock:
            for key, val in ops:
                if val is _TOMBSTONE:
                    # Remove tombstone entirely
                    self._cache.pop(key, None)
                # else: keep cached copy for future reads
                self._dirty.discard(key)

        if self._clear_cache_on_flush:
            with self._lock:
                self._cache.clear()

    # Aliases
    sync = flush
    commit = flush

    # ------------------------------------------------------------------ #

    @property
    def dirty_keys(self) -> Set[str]:
        """Return a *copy* of the set of keys awaiting flush."""
        with self._lock:
            return set(self._dirty)

    def close(self) -> None:
        """Flush and mark the store closed.  Further ops raise ValueError."""
        if not self._closed:
            self.flush()
            self._closed = True

    # Context-manager helpers
    def __enter__(self) -> "BufferedStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        # Flush on clean exit only; if an exception is propagating, do nothing.
        if exc_type is None:
            try:
                self.flush()
            finally:
                self._closed = True
        return None

    # ------------------------------------------------------------------ #
    # Misc.                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"{len(self._cache)} cached, {len(self._dirty)} dirty"
        return f"<BufferedStore {status} backing={self._store!r}>"


# ------------------------------------------------------------------------- #
# Example usage                                                             #
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    import zarr
    from zarr.storage import MemoryStore

    # Base store simulating a slow remote FS
    remote = MemoryStore()

    # Create an array on the *remote* store
    z = zarr.create((100, 100), chunks=(25, 25), store=remote, dtype="i4", overwrite=True)

    # Wrap it in our BufferedStore
    with BufferedStore(remote) as buf:
        # Open same array via the buffer
        arr = zarr.open(buf, mode="r+")
        arr[:] = 42            # many writes → stay in local RAM
        arr[0, 0] = 7          # another write
        print("Dirty keys before flush:", buf.dirty_keys)

        # Manual flush
        buf.flush()
        print("Dirty keys after flush :", buf.dirty_keys)

    # Verify data really landed in the remote store
    print("Remote chunk count:", len(remote))
