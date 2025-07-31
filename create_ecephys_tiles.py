import os
import time
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import zarr
from zarr.convenience import consolidate_metadata

import s3fs
import zarr.storage

from nwbextractors import NwbRecordingExtractor
from spikeinterface.preprocessing import (
    BandpassFilterRecording,
    CommonReferenceRecording,
)

import lindi

import fsspec

compression = "gzip"
compression_opts = 3


def get_status_file_path(store):
    """Get the path for the status JSON file based on the store type."""
    if isinstance(store, zarr.DirectoryStore):
        # For local directory store, add .status.json to the directory path
        return store.path + ".status.json"
    elif isinstance(store, fsspec.mapping.FSMap):
        # For S3 store, add .status.json to the S3 path
        return store.root + ".status.json"
    else:
        raise ValueError(f"Unsupported store type: {type(store)}")


def read_status_file(store):
    """Read the status JSON file and return the status data."""
    status_path = get_status_file_path(store)

    if isinstance(store, zarr.DirectoryStore):
        # Local file system
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    elif isinstance(store, fsspec.mapping.FSMap):
        # S3 file system
        fs = store.fs
        if fs.exists(status_path):
            with fs.open(status_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    else:
        raise ValueError(f"Unsupported store type: {type(store)}")


def write_status_file(store, status_data):
    """Write the status data to the status JSON file."""
    status_path = get_status_file_path(store)

    if isinstance(store, zarr.DirectoryStore):
        # Local file system
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
    elif isinstance(store, fsspec.mapping.FSMap):
        # S3 file system
        fs = store.fs
        with fs.open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported store type: {type(store)}")


def create_ecephys_tiles(
    *,
    input_url: str,
    electrical_series_path: str,
    store,
    chunk_size: tuple[int, int],
    downsampling_base: int,
    read_status_func,
    write_status_func,
):
    freq_min = 300
    freq_max = 6000

    local_cache = lindi.LocalCache()
    h5f = lindi.LindiH5pyFile.from_hdf5_file(input_url, local_cache=local_cache)

    R = NwbRecordingExtractor(file=h5f, electrical_series_path=electrical_series_path)

    R = BandpassFilterRecording(R, freq_min=freq_min, freq_max=freq_max)
    R = CommonReferenceRecording(R, reference="global", operator="median")

    # Ensure the root store exists
    root = zarr.group(store=store)

    obj_grp = root.require_group(electrical_series_path)

    tiles_attrs = {
        "downsampling_base": downsampling_base,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "common_reference": "global",
        "common_reference_operator": "median",
        "sampling_frequency": R.get_sampling_frequency(),
        "num_samples": R.get_num_samples(),
        "num_channels": R.get_num_channels(),
    }

    if "ecephys_tiles" in obj_grp:
        # check that the attributes match
        tiles_grp = obj_grp["ecephys_tiles"]
        assert isinstance(tiles_grp, zarr.Group)
        for key, value in tiles_attrs.items():
            if key not in tiles_grp.attrs or not values_match(
                tiles_grp.attrs[key], value
            ):
                raise ValueError(
                    f"Attribute mismatch: {key}/{tiles_grp.attrs[key]} does not match expected value: {value}. "
                )
    else:
        tiles_grp = obj_grp.create_group("ecephys_tiles")
        for key, value in tiles_attrs.items():
            tiles_grp.attrs[key] = value

    num_levels = 0
    num_samples_in_level = R.get_num_samples()
    while num_samples_in_level > chunk_size[0]:
        if num_levels == 0:
            shape0 = (num_samples_in_level, R.get_num_channels())
            chunks0 = (chunk_size[0], chunk_size[1])
        else:
            shape0 = (num_samples_in_level, 2, R.get_num_channels())
            chunks0 = (chunk_size[0], 2, chunk_size[1])
        print(
            f"Creating level {num_levels} with shape {shape0} and chunk size {chunk_size}"
        )
        ds = tiles_grp.require_dataset(
            f"level_{num_levels}/data",
            shape=shape0,
            chunks=chunks0,
            dtype="int16",
            compression=compression,
            compression_opts=compression_opts,
            fill_value=0,
            order="C",
        )
        ds.attrs["downsampling_factor"] = downsampling_base**num_levels
        ds_spike_counts = tiles_grp.require_dataset(
            f"level_{num_levels}/spike_counts",
            shape=[num_samples_in_level, R.get_num_channels()],
            chunks=[chunk_size[0], chunk_size[1]],
            dtype="uint16",
            compression=compression,
            compression_opts=compression_opts,
            fill_value=0,
            order="C",
        )
        ds_spike_counts.attrs["detect_threshold"] = 5
        ds_spike_counts.attrs["exclude_sweep_ms"] = 1
        ds_spike_counts.attrs["downsampling_factor"] = downsampling_base**num_levels
        num_levels += 1
        num_samples_in_level //= downsampling_base

    print("Consolidating metadata")
    consolidate_metadata(store)

    segment_size = chunk_size[0]
    while segment_size < downsampling_base**num_levels:
        segment_size *= downsampling_base
    # Let's do at least 10 seconds at a time
    while segment_size < R.get_sampling_frequency() * 10:
        segment_size *= 2

    # Read status from JSON file, with fallback to zarr attributes for backward compatibility
    status_data = read_status_func()
    last_start_sample = status_data.get("last_start_sample", None)

    if last_start_sample is None:
        start_sample = 0
    else:
        start_sample = last_start_sample + segment_size
    while start_sample + segment_size <= R.get_num_samples():
        traces = R.get_traces(
            start_frame=start_sample,
            end_frame=start_sample + segment_size,
        )
        prev_traces_min = None
        prev_traces_max = None
        prev_spike_counts = None
        for level in range(num_levels):
            print(
                f"Processing level {level}, Start time {start_sample / R.get_sampling_frequency():.3f} sec"
            )
            i1 = start_sample // (downsampling_base**level)
            i2 = (start_sample + segment_size) // (downsampling_base**level)
            ds = tiles_grp[f"level_{level}/data"]
            ds_spike_counts = tiles_grp[f"level_{level}/spike_counts"]
            if level == 0:
                ds[i1:i2, :] = traces.astype(np.int16)
                prev_traces_min = traces
                prev_traces_max = traces
                print(
                    f"Detecting peaks for level {level}, Start time {start_sample / R.get_sampling_frequency():.3f} sec"
                )
                peaks = detect_peaks(
                    traces=traces,
                    detect_threshold=5,
                    exclude_sweep_ms=1,
                    sampling_frequency=R.get_sampling_frequency(),
                )
                print(f"Found {len(peaks)} peaks")
                spike_counts = np.zeros(
                    (segment_size, R.get_num_channels()), dtype=np.int32
                )
                for peak in peaks:
                    sample_index = peak[0] + start_sample
                    channel_index = peak[1]
                    if start_sample <= sample_index < start_sample + segment_size:
                        spike_counts[sample_index - start_sample, channel_index] += 1
                ds_spike_counts[i1:i2, :] = spike_counts.astype(np.uint16)
                prev_spike_counts = spike_counts
            else:
                assert prev_traces_min is not None and prev_traces_max is not None
                assert prev_spike_counts is not None
                traces_min_for_downsampling = prev_traces_min.reshape(
                    (
                        prev_traces_min.shape[0] // downsampling_base,
                        downsampling_base,
                        R.get_num_channels(),
                    )
                )
                traces_max_for_downsampling = prev_traces_max.reshape(
                    (
                        prev_traces_max.shape[0] // downsampling_base,
                        downsampling_base,
                        R.get_num_channels(),
                    )
                )
                prev_traces_min = np.min(traces_min_for_downsampling, axis=1)
                prev_traces_max = np.max(traces_max_for_downsampling, axis=1)
                ds[i1:i2, :] = np.stack(
                    (
                        prev_traces_min.astype(np.int16),
                        prev_traces_max.astype(np.int16),
                    ),
                    axis=1,
                ).astype(np.int16)
                spike_counts_for_downsampling = prev_spike_counts.reshape(
                    (
                        prev_spike_counts.shape[0] // downsampling_base,
                        downsampling_base,
                        R.get_num_channels(),
                    )
                )
                prev_spike_counts = np.sum(spike_counts_for_downsampling, axis=1)
                ds_spike_counts[i1:i2, :] = prev_spike_counts.astype(np.uint16)

        # Write status to JSON file so we can pick off where we left off
        status_data = {"last_start_sample": start_sample}
        write_status_func(status_data)
        start_sample += segment_size


def values_match(value1, value2):
    if isinstance(value1, np.ndarray):
        value1 = value1.tolist()
    if isinstance(value2, np.ndarray):
        value2 = value2.tolist()
    if isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        return len(value1) == len(value2) and all(
            values_match(v1, v2) for v1, v2 in zip(value1, value2)
        )
    else:
        return value1 == value2


def determine_channel_groups(
    num_channels: int, channel_group_size: int
) -> list[list[int]]:
    channel_groups = []
    ii = 0
    while True:
        start_channel = ii * channel_group_size
        if num_channels - start_channel < channel_group_size * 1.5:
            channel_groups.append([start_channel, num_channels])
            break
        else:
            if start_channel + channel_group_size > num_channels:
                channel_groups.append([start_channel, num_channels])
            else:
                channel_groups.append(
                    [start_channel, start_channel + channel_group_size]
                )
        ii += 1
    return channel_groups


def detect_peaks(
    *,
    traces: np.ndarray,  # shape (num_samples, num_channels)
    detect_threshold: float,
    exclude_sweep_ms: float,
    sampling_frequency: float,
):
    # num_samples, num_channels = traces.shape
    exclude_sweep_samples = int(round(exclude_sweep_ms * sampling_frequency / 1000.0))

    # 1. Channel-wise noise estimate (MAD → σ̂)
    noise_level = 1.4826 * np.median(
        np.abs(traces - np.median(traces, axis=0)), axis=0
    )  # shape (C,)

    threshold = detect_threshold * noise_level  # shape (C,)

    # 2. All samples that cross the (negative) threshold
    candidate_mask = traces < -threshold  # shape (N, C) boolean

    # 3. Non-maximum suppression in a ±exclude_sweep_samples window
    if exclude_sweep_samples == 0:
        peak_mask = candidate_mask
    else:
        w = 2 * exclude_sweep_samples + 1  # full window length

        # Pad with +∞ so padded values can never be minima
        pad_val = 99
        padded = np.pad(
            traces,
            ((exclude_sweep_samples, exclude_sweep_samples), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )

        # windows.shape -> (num_samples, w, num_channels)
        windows = sliding_window_view(padded, window_shape=w, axis=0)

        # Local minimum in each window
        local_min = windows.min(axis=-1)  # shape (N, C)

        # A true peak: most-negative in its window *and* crosses threshold
        peak_mask = candidate_mask & (traces == local_min)

    # 4. Return as list[(sample_index, channel_index)]
    return list(map(tuple, np.argwhere(peak_mask)))


if __name__ == "__main__":
    # 000409
    # nwb_url = "https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/"
    # electrical_series_path = "/acquisition/ElectricalSeriesAp"

    # https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/d4bd92fc-4119-4393-b807-f007a86778a1/download/&dandisetId=000957&dandisetVersion=draft
    dandiset_id = "000957"
    nwb_url = "https://api.dandiarchive.org/api/assets/d4bd92fc-4119-4393-b807-f007a86778a1/download/"
    electrical_series_path = "/acquisition/ElectricalSeriesAP"

    assert nwb_url.split("/")[-2] == "download", "URL must end with 'download/'"
    asset_id = nwb_url.split("/")[-3]

    local = False
    if local:
        output_path = "000957_example.ns.zarr"
        store = zarr.DirectoryStore(output_path)
        status_file_path = output_path + ".status.json"
        # Create status file functions using local file system
        def read_status_func():
            """Read status file from local file system."""
            if os.path.exists(status_file_path):
                with open(status_file_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        def write_status_func(status_data):
            """Write status file to local file system."""
            with open(status_file_path, 'w') as f:
                json.dump(status_data, f, indent=2)
    else:
        fs = s3fs.S3FileSystem(
            anon=False,
            key=os.environ['NEUROSIFT_TILES_AWS_ACCESS_KEY_ID'],
            secret=os.environ['NEUROSIFT_TILES_AWS_SECRET_ACCESS_KEY'],
            client_kwargs={
                "endpoint_url": os.environ['NEUROSIFT_TILES_S3_ENDPOINT'],
                "region_name": "auto"
            }
        )
        s3_path = f'neurosift-tiles/dandisets/{dandiset_id}/{asset_id}/tiles.zarr'
        store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
        store = zarr.storage.LRUStoreCache(
            store=store,
            max_size=50 * 2**20,  # 50 MB
        )

        # Create status file functions using S3 filesystem directly
        status_file_path = s3_path + '.status.json'

        def read_status_func():
            """Read status file using S3 filesystem directly."""
            if fs.exists(status_file_path):
                with fs.open(status_file_path, 'r') as f:
                    return json.load(f)
            else:
                return {}

        def write_status_func(status_data):
            """Write status file using S3 filesystem directly."""
            with fs.open(status_file_path, 'w') as f:
                json.dump(status_data, f, indent=2)

    try:
        create_ecephys_tiles(
            input_url=nwb_url,
            electrical_series_path=electrical_series_path,
            store=store,
            chunk_size=(
                2**12,
                128,
            ),  # 4096 samples and 128 channels per chunk (should be on the order of what fits on the screen)
            downsampling_base=4,
            read_status_func=read_status_func,
            write_status_func=write_status_func,
        )
    # catch keyboard interrupt to allow graceful exit
    except KeyboardInterrupt:
        print("Interrupted by user, consolidating metadata and exiting...")
        consolidate_metadata(store)  # type: ignore
        print("Metadata consolidated, exiting.")
        raise
