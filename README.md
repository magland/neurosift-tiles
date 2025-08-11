# Neurosift Tiles - Ecephys Processing

This repository contains tools for processing electrophysiology (ecephys) datasets and creating tiles for visualization in Neurosift.

## Files

- `ecephys_catalog.json` - Catalog of ecephys datasets to be processed
- `create_ecephys_tiles.py` - Main processing script
- `nwbextractors.py` - NWB file reading utilities
- `BufferedStore.py` - Buffered storage layer for efficient S3 operations

## Usage

### Process all datasets in the catalog

```bash
python create_ecephys_tiles.py
```

### Process a specific dataset

```bash
python create_ecephys_tiles.py --dataset 000409
```

### Use local storage instead of S3

```bash
python create_ecephys_tiles.py --local
```

### Use a custom catalog file

```bash
python create_ecephys_tiles.py --catalog my_catalog.json
```

## Catalog Format

The `ecephys_catalog.json` file contains a list of datasets to process:

```json
{
  "datasets": [
    {
      "dandiset_id": "000409",
      "asset_id": "c04f6b30-82bf-40e1-9210-34f0bcd8be24",
      "nwb_url": "https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/",
      "electrical_series_path": "/acquisition/ElectricalSeriesAp"
    }
  ],
  "metadata": {
    "version": "1.0"
  }
}
```

## Environment Variables (for S3 storage)

When not using `--local`, the following environment variables must be set:

- `NEUROSIFT_TILES_AWS_ACCESS_KEY_ID`
- `NEUROSIFT_TILES_AWS_SECRET_ACCESS_KEY`
- `NEUROSIFT_TILES_S3_ENDPOINT`

## Processing Details

The script processes each dataset by:

1. Loading the NWB file from the specified URL
2. Applying bandpass filtering (300-6000 Hz)
3. Applying common reference (global median)
4. Creating multi-level tiles with downsampling
5. Detecting spikes and creating spike count tiles
6. Storing results in Zarr format

Processing can be resumed if interrupted, as progress is tracked in status files.
