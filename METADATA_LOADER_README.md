# Metadata-Based Data Loader

## Overview

This document describes how to use the **metadata-based loader** to train your EEG source localization model **directly from metadata files** (`train_sample_source1.mat`, `test_sample_source1.mat`) **without pre-extracting samples**.

The loader reads sample configurations from metadata files and generates training samples on-the-fly by loading the appropriate NMM spike data from the `source/nmm_spikes` directory.

## Key Benefits

✅ **No Pre-extraction Required**: Train directly from metadata files  
✅ **Memory Efficient**: Loads only needed data during training  
✅ **Fast Setup**: No need to run `extract_labeled_data.py` first  
✅ **Flexible**: Uses your existing metadata structure  
✅ **Production Ready**: Same quality as pre-extracted data  

---

## Quick Start

### 1. Verify Your Data Structure

```
project_root/
├── source/
│   ├── train_sample_source1.mat      # Training metadata
│   ├── test_sample_source1.mat       # Test metadata
│   └── nmm_spikes/                   # NMM spike data
│       ├── a0/
│       │   ├── nmm_1.mat
│       │   ├── nmm_2.mat
│       │   └── nmm_3.mat
│       ├── a1/
│       │   ├── nmm_1.mat
│       │   ├── nmm_2.mat
│       │   └── ... (up to nmm_24.mat or more)
│       └── ... (other region folders)
├── anatomy/
│   └── leadfield_75_20k.mat          # Forward/leadfield matrix
├── configs/
│   └── config.py
└── train.py
```

### 2. Configure `configs/config.py`

```python
class Config:
    # Set to True to use metadata-based loader
    USE_METADATA_LOADER = True
    
    # Metadata file paths
    TRAIN_METADATA_PATH = "source/train_sample_source1.mat"
    TEST_METADATA_PATH = "source/test_sample_source1.mat"
    NMM_SPIKES_DIR = "source/nmm_spikes"
    FWD_MATRIX_PATH = "anatomy/leadfield_75_20k.mat"
    
    # Optional: limit dataset size (None = use all samples)
    TRAIN_DATASET_LEN = None  # Or set to e.g., 10000 for subset
    TEST_DATASET_LEN = None
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    # ... other parameters ...
```

### 3. Test the Loader

```bash
python test_metadata_loader.py
```

This will:
- ✓ Check if all required files exist
- ✓ Load forward matrix
- ✓ Create dataset from metadata
- ✓ Test sample loading
- ✓ Test batch generation

### 4. Start Training

```bash
python train.py
```

That's it! The model will train directly from your metadata files.

---

## How It Works

### Metadata File Structure

Your `train_sample_source1.mat` contains:

```matlab
selected_region: [num_samples × num_sources × max_patch_size]
    - Defines which brain regions are active for each sample
    - Value 999 indicates padding

nmm_idx: [num_samples × num_sources] or [num_samples]
    - Index to select which NMM spike file to use

scale_ratio: [num_samples × num_sources × num_scale_options]
    - Amplitude scaling factors for source activity

mag_change: [num_samples × num_sources × max_patch_size]
    - Spatial decay weights within source patches

current_snr: [num_samples]
    - SNR value for noise injection (in dB)
```

### Sample Generation Process

For each sample during training:

1. **Load Metadata**: Read source configuration from metadata file
2. **Select Regions**: Get active brain regions from `selected_region`
3. **Load NMM Data**: Load spike data from `nmm_spikes/aX/nmm_Y.mat` based on `nmm_idx`
4. **Apply Scaling**: Scale activity using `scale_ratio`
5. **Apply Spatial Pattern**: Apply `mag_change` weights to create realistic source patches
6. **Project to Sensors**: Multiply by forward matrix to get EEG sensor data
7. **Add Noise**: Add Gaussian noise based on `current_snr`
8. **Normalize**: Normalize EEG and source data

---

## Configuration Options

### Basic Configuration

```python
# configs/config.py

# Enable metadata-based loader
USE_METADATA_LOADER = True

# Required paths
TRAIN_METADATA_PATH = "source/train_sample_source1.mat"
TEST_METADATA_PATH = "source/test_sample_source1.mat"
NMM_SPIKES_DIR = "source/nmm_spikes"
FWD_MATRIX_PATH = "anatomy/leadfield_75_20k.mat"
```

### Dataset Size Control

```python
# Use all samples from metadata (default)
TRAIN_DATASET_LEN = None
TEST_DATASET_LEN = None

# Or limit to specific number
TRAIN_DATASET_LEN = 50000  # Use first 50k training samples
TEST_DATASET_LEN = 5000     # Use first 5k test samples
```

### Validation Split

```python
VAL_SPLIT = 0.1  # Use 10% of training data for validation
```

### Training Parameters

```python
BATCH_SIZE = 8          # Batch size
NUM_EPOCHS = 100        # Training epochs
LEARNING_RATE = 1e-4    # Learning rate
NUM_WORKERS = 4         # Data loading workers (0 for debugging)
```

---

## Usage Examples

### Example 1: Full Training Dataset

```python
# configs/config.py
USE_METADATA_LOADER = True
TRAIN_METADATA_PATH = "source/train_sample_source1.mat"
TEST_METADATA_PATH = "source/test_sample_source1.mat"
TRAIN_DATASET_LEN = None  # Use all 190,656 samples
TEST_DATASET_LEN = None
```

```bash
python train.py
```

### Example 2: Quick Testing with Subset

```python
# configs/config.py
USE_METADATA_LOADER = True
TRAIN_DATASET_LEN = 1000  # Just 1000 samples for quick test
TEST_DATASET_LEN = 100
BATCH_SIZE = 4
NUM_EPOCHS = 5
```

```bash
python train.py
```

### Example 3: Custom Paths

```python
# configs/config.py
USE_METADATA_LOADER = True
TRAIN_METADATA_PATH = "custom/path/train_data.mat"
TEST_METADATA_PATH = "custom/path/test_data.mat"
NMM_SPIKES_DIR = "custom/path/nmm_spikes"
FWD_MATRIX_PATH = "custom/path/leadfield.mat"
```

---

## API Reference

### SpikeEEGMetadataDataset

Main dataset class for metadata-based loading.

```python
from utils.loader import SpikeEEGMetadataDataset

dataset = SpikeEEGMetadataDataset(
    metadata_path="source/train_sample_source1.mat",
    fwd=forward_matrix,           # (75, 994) numpy array
    nmm_spikes_dir="source/nmm_spikes",
    dataset_len=None,             # None = use all samples
    normalize=True
)

# Get a sample
eeg_data, source_data = dataset[0]
# eeg_data: (500, 75) - EEG sensor data
# source_data: (500, 994) - Source space activity
```

### create_dataloaders_from_metadata

Creates train/val/test dataloaders from metadata files.

```python
from utils.loader import create_dataloaders_from_metadata

train_loader, val_loader, test_loader, _ = create_dataloaders_from_metadata(
    train_metadata_path="source/train_sample_source1.mat",
    test_metadata_path="source/test_sample_source1.mat",
    fwd_matrix_path="anatomy/leadfield_75_20k.mat",
    batch_size=8,
    val_split=0.1,
    num_workers=4,
    nmm_spikes_dir="source/nmm_spikes",
    train_dataset_len=None,
    test_dataset_len=None
)
```

---

## Comparison: Metadata vs Dynamic Generation

| Feature | Metadata-Based | Dynamic Generation |
|---------|----------------|-------------------|
| **Input** | Metadata files | Region folders only |
| **Sample Config** | Pre-defined in metadata | Randomly generated |
| **Reproducibility** | ✓ Exact same samples | ✗ Random each time |
| **Flexibility** | ✓ Full control via metadata | Limited |
| **Setup** | Requires metadata files | No metadata needed |
| **Use Case** | Production training | Quick experiments |

**Recommendation**: Use **metadata-based** for actual training with your curated dataset.

---

## Switching Between Loaders

You can easily switch between metadata-based and dynamic generation:

```python
# configs/config.py

# For metadata-based (recommended for production)
USE_METADATA_LOADER = True

# For dynamic generation (quick experiments)
USE_METADATA_LOADER = False
```

No other code changes needed!

---

## Troubleshooting

### Error: "No mat files found in source/nmm_spikes"

**Problem**: NMM spike data directory not found or empty

**Solution**: 
1. Check that `source/nmm_spikes/` exists
2. Verify that subdirectories like `a0/`, `a1/` contain `.mat` files
3. Ensure paths in config are correct

```bash
ls source/nmm_spikes/a0/  # Should show nmm_*.mat files
ls source/nmm_spikes/a1/  # Should show nmm_*.mat files
```

### Error: "Could not find forward matrix in file"

**Problem**: Forward matrix file doesn't have expected field name

**Solution**:
1. Check that file exists: `anatomy/leadfield_75_20k.mat`
2. The file should contain one of: `fwd`, `forward`, `leadfield`, or `L`
3. Matrix should have shape `(75, 994)` or `(994, 75)`

```python
# Check forward matrix
from scipy.io import loadmat
data = loadmat('anatomy/leadfield_75_20k.mat')
print(data.keys())  # Should show available fields
```

### Error: "Could not load metadata file"

**Problem**: Metadata file format issue

**Solution**:
The loader supports multiple formats:
- MATLAB v7 (scipy)
- MATLAB v7.3 / HDF5 (h5py)  
- Octave text format

Ensure your file is in one of these formats.

### Warning: "Could not load NMM data for index X"

**Problem**: Specific NMM spike file missing or corrupted

**Solution**:
- The loader will use zeros for missing files (non-fatal)
- Check if the referenced file exists
- Verify file is valid `.mat` format
- If many files are missing, you may need to regenerate spike data

### Slow Data Loading

**Problem**: Training is slow due to data loading

**Solutions**:
1. Increase `NUM_WORKERS` in config (try 4-8)
2. Reduce `BATCH_SIZE` if memory constrained
3. Use subset for initial testing (`TRAIN_DATASET_LEN = 10000`)
4. Check disk I/O speed

```python
# For faster loading
NUM_WORKERS = 8  # More parallel workers
PIN_MEMORY = True  # If using GPU
```

### Out of Memory

**Problem**: Training crashes with OOM error

**Solutions**:
1. Reduce `BATCH_SIZE` (try 4 or 2)
2. Reduce number of workers
3. Use gradient accumulation
4. Enable mixed precision training (`USE_AMP = True`)

---

## Performance Tips

### Optimal Settings for Training

```python
# For fast training on GPU
BATCH_SIZE = 16         # Larger batches if GPU memory allows
NUM_WORKERS = 8         # 2x number of CPU cores
PIN_MEMORY = True       # Faster GPU transfer
USE_AMP = True          # Mixed precision training
```

### For Debugging

```python
# For debugging / testing
TRAIN_DATASET_LEN = 100  # Small dataset
BATCH_SIZE = 4
NUM_WORKERS = 0          # Single process for easier debugging
NUM_EPOCHS = 2
```

### For Maximum Accuracy

```python
# Use full dataset
TRAIN_DATASET_LEN = None  # All 190k samples
TEST_DATASET_LEN = None
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 8           # Stable training
```

---

## Data Statistics

Based on your metadata files:

- **Training samples**: ~190,656 (from train_sample_source1.mat)
- **Test samples**: Check test_sample_source1.mat
- **Sources per sample**: 1-2 (configurable in metadata)
- **Brain regions**: 994 total
- **EEG channels**: 75
- **Time points**: 500

---

## Next Steps

1. **Test the loader**:
   ```bash
   python test_metadata_loader.py
   ```

2. **Configure** training parameters in `configs/config.py`

3. **Start training**:
   ```bash
   python train.py
   ```

4. **Monitor** progress:
   - Check console output for loss metrics
   - View TensorBoard: `tensorboard --logdir logs/`
   - Check checkpoints in `checkpoints/` directory

5. **Evaluate** on test set automatically at end of training

---

## Advanced Usage

### Custom Metadata Format

If your metadata has a different structure, modify `SpikeEEGMetadataDataset.__getitem__()` in `utils/loader.py`:

```python
def __getitem__(self, index):
    # Customize how you read metadata fields
    raw_lb = self.dataset_meta["your_field_name"][index]
    # ... rest of the method
```

### Custom NMM File Mapping

To change how `nmm_idx` maps to files, modify `_load_nmm_data()`:

```python
def _load_nmm_data(self, nmm_idx):
    # Your custom file selection logic
    a_dir, file_num = your_mapping_function(nmm_idx)
    file_path = f".../{a_dir}/nmm_{file_num}.mat"
    # ...
```

---

## Support

For issues:
1. Run `python test_metadata_loader.py` to diagnose
2. Check console error messages
3. Verify file paths in `configs/config.py`
4. Ensure data structure matches requirements

---

## Summary

The metadata-based loader provides a seamless way to train your model directly from your curated dataset metadata files, without the need for pre-extraction. It's:

- ✅ **Fast to set up** - no pre-extraction needed
- ✅ **Memory efficient** - loads data on-the-fly  
- ✅ **Flexible** - fully configurable via metadata
- ✅ **Production ready** - same quality as pre-extracted data

Just configure paths in `config.py` and run `python train.py`!

