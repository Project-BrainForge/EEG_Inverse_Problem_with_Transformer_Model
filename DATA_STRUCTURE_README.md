# EEG Inverse Problem Data Structure

## Overview

This project now uses the `nmm_spikes` data structure for loading EEG source localization data. The new loader dynamically generates samples by loading neural mass model (NMM) data from region-specific folders.

## Directory Structure

```
project_root/
├── source/
│   └── nmm_spikes/
│       ├── a0/
│       │   ├── nmm_1.mat
│       │   ├── nmm_2.mat
│       │   └── ... (variable number of files)
│       ├── a1/
│       │   ├── nmm_1.mat
│       │   └── ... (variable number of files)
│       ├── a2/
│       │   └── ...
│       └── ... (up to a993)
├── forward_matrix.mat  (75 × 994 forward/leadfield matrix)
├── train.py
├── test_loader.py
└── configs/
    └── config.py
```

### Data Structure Details

- **Region Folders**: `a0` to `a993` (994 total regions)
- **MAT Files**: Each folder contains variable number of `.mat` files
- **MAT File Format**: Each `.mat` file should contain:
  - `data`: numpy array of shape `(time_points, 994)` - neural activity across all regions
  - Time points can be variable (will be resampled to 500)
  - Regions dimension should be 994 (or will be padded/truncated)

## New Data Loader

### Key Features

1. **Dynamic Sample Generation**: Instead of loading pre-generated samples, the loader creates samples on-the-fly by:
   - Randomly selecting source regions (default: 2 sources per sample)
   - Creating source patches around selected regions (default: 20 regions per patch)
   - Loading NMM data from the region folders
   - Projecting to sensor space using forward matrix
   - Adding Gaussian noise with variable SNR

2. **Flexible File Structure**: Each region folder can have a different number of MAT files

3. **Automatic Resampling**: Handles different time lengths and automatically resamples to 500 time points

### Configuration Parameters

Edit `configs/config.py` to customize:

```python
# Data parameters
DATA_DIR = "."  # Root directory containing 'source/nmm_spikes'
FWD_MATRIX_PATH = "forward_matrix.mat"  # Path to forward matrix file
EEG_CHANNELS = 75
SOURCE_REGIONS = 994
SEQ_LEN = 500

# Dataset generation parameters
DATASET_LEN = 1000  # Number of samples to generate per epoch
NUM_SOURCES = 2  # Number of active sources per sample
PATCH_SIZE = 20  # Size of each source patch
SNR_RANGE = (0, 30)  # SNR range in dB for noise addition
```

## Usage

### 1. Prepare Your Data

Organize your NMM data files into the required structure:

```bash
source/nmm_spikes/
├── a0/
│   ├── nmm_1.mat
│   ├── nmm_2.mat
│   └── ...
├── a1/
│   └── ...
...
```

### 2. Prepare Forward Matrix

Create or place your forward matrix file:

```python
# Example: Create a forward matrix file
import numpy as np
from scipy.io import savemat

fwd = np.random.randn(75, 994)  # Replace with your actual forward matrix
savemat('forward_matrix.mat', {'fwd': fwd})
```

### 3. Test the Loader

Run the test script to verify your data structure:

```bash
python test_loader.py
```

This will:
- Check if the directory structure is correct
- Load a few samples
- Verify data shapes and ranges
- Test the dataloader with batching

### 4. Train the Model

Run training with the new loader:

```bash
python train.py
```

Or with custom parameters:

```bash
python train.py --data_dir . --batch_size 16 --epochs 50
```

## API Reference

### SpikeEEGDataset

The main dataset class for loading spike data.

```python
from utils.loader import SpikeEEGDataset

dataset = SpikeEEGDataset(
    data_root=".",              # Root directory
    fwd=forward_matrix,         # Forward matrix (75, 994)
    num_sources=2,              # Sources per sample
    patch_size=20,              # Regions per patch
    dataset_len=1000,           # Total samples
    snr_range=(0, 30),         # SNR range in dB
    normalize=True              # Normalize output
)

# Get a sample
eeg_data, source_data = dataset[0]
# eeg_data: (500, 75) - EEG sensor data
# source_data: (500, 994) - Source space activity
```

### create_dataloaders_from_spikes

Creates train/val/test dataloaders.

```python
from utils.loader import create_dataloaders_from_spikes

train_loader, val_loader, test_loader, _ = create_dataloaders_from_spikes(
    data_root=".",
    fwd_matrix_path="forward_matrix.mat",
    batch_size=8,
    train_split=0.8,
    val_split=0.1,
    num_workers=4,
    dataset_len=1000,
    num_sources=2,
    patch_size=20,
    snr_range=(0, 30)
)
```

## MAT File Format Requirements

Each `.mat` file in the region folders should contain:

```python
{
    'data': numpy.ndarray  # Shape: (time_points, 994)
                          # time_points: flexible (will be resampled to 500)
                          # 994: number of brain regions
}
```

Alternative field names also supported: `nmm_data`, `source_data`, `signals`

## Supported MAT File Formats

The loader supports multiple MAT file formats:
- MATLAB v7 and earlier (via scipy.io.loadmat)
- MATLAB v7.3 / HDF5 format (requires h5py)
- Octave text format

## Troubleshooting

### Error: "No mat files found in source/nmm_spikes"

**Solution**: Ensure your directory structure is correct and contains `.mat` files:
```bash
ls source/nmm_spikes/a0/  # Should show .mat files
```

### Error: "Could not find forward matrix in file"

**Solution**: Ensure your forward matrix file contains a field named `fwd`, `forward`, or `leadfield` with shape (75, 994)

### Warning: "Could not load NMM data for index X"

**Solution**: Check that the region folder exists and contains valid `.mat` files. The loader will return zeros for missing data.

### Data shape mismatch

**Solution**: The loader automatically handles:
- Time dimension: Resamples to 500 time points
- Region dimension: Truncates to 994 or pads with zeros

## Migration from Old Dataset

If you're migrating from the old `dataset_with_label` structure:

1. **Old structure**: Pre-generated `.mat` files with `eeg_data` and `source_data`
2. **New structure**: Raw NMM data in region folders, samples generated on-the-fly

Benefits of new structure:
- More flexible data augmentation
- Variable SNR levels during training
- Less storage space (no need to pre-generate all samples)
- Easy to add new NMM data files

## Performance Tips

1. **Increase dataset_len**: Generate more samples per epoch for better convergence
2. **Adjust num_workers**: Use 4-8 workers for faster data loading (but 0 for debugging)
3. **Tune SNR_RANGE**: Adjust noise levels based on your data quality
4. **Modify PATCH_SIZE**: Larger patches = more diffuse sources, smaller = more focal

## Example Workflow

```python
# 1. Load the data
from utils.loader import create_dataloaders_from_spikes

train_loader, val_loader, test_loader, _ = create_dataloaders_from_spikes(
    data_root=".",
    fwd_matrix_path="forward_matrix.mat",
    batch_size=8,
    dataset_len=1000
)

# 2. Iterate through batches
for eeg_batch, source_batch in train_loader:
    # eeg_batch: (batch_size, 500, 75)
    # source_batch: (batch_size, 500, 994)
    
    # Your training code here
    pass
```

## Contact & Support

For issues or questions about the data loader, please check:
1. The test script output: `python test_loader.py`
2. Configuration settings in `configs/config.py`
3. Data directory structure

