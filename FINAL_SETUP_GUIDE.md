# 🚀 Final Setup Guide - Direct Training from Metadata

## What Changed?

You can now **train your model directly from metadata files** (`train_sample_source1.mat`, `test_sample_source1.mat`) without needing to pre-extract samples first!

### Before (Old Workflow):
```
metadata files → extract_labeled_data.py → sample_*.mat files → train.py
```

### Now (New Workflow):
```
metadata files + nmm_spikes/ → train.py ✨
```

---

## 🎯 Quick Start (3 Steps)

### Step 1: Verify Your Files

Make sure you have:
- ✅ `source/train_sample_source1.mat`
- ✅ `source/test_sample_source1.mat`
- ✅ `source/nmm_spikes/a0/`, `a1/`, etc. with `.mat` files
- ✅ `anatomy/leadfield_75_20k.mat`

### Step 2: Configure

Open `configs/config.py` and ensure:

```python
# Enable metadata-based loader (already set!)
USE_METADATA_LOADER = True

# Paths (verify these match your structure)
TRAIN_METADATA_PATH = "source/train_sample_source1.mat"
TEST_METADATA_PATH = "source/test_sample_source1.mat"
NMM_SPIKES_DIR = "source/nmm_spikes"
FWD_MATRIX_PATH = "anatomy/leadfield_75_20k.mat"

# Optional: limit samples for testing
TRAIN_DATASET_LEN = None  # None = use all (~190k samples)
TEST_DATASET_LEN = None
```

### Step 3: Test & Train

```bash
# Test the loader (recommended first)
python test_metadata_loader.py

# If tests pass, start training!
python train.py
```

That's it! 🎉

---

## 📊 What the New Loader Does

For each training sample:

1. Reads source configuration from `train_sample_source1.mat`
2. Loads NMM spike data from `source/nmm_spikes/`
3. Applies spatial patterns and scaling from metadata
4. Projects to EEG sensor space using forward matrix
5. Adds noise based on SNR from metadata
6. Returns normalized EEG and source data

All happens **on-the-fly during training** - no pre-extraction needed!

---

## 🔧 Configuration Options

### Use All Samples (Default)

```python
TRAIN_DATASET_LEN = None  # ~190k samples
TEST_DATASET_LEN = None
BATCH_SIZE = 8
NUM_EPOCHS = 100
```

### Quick Test Run

```python
TRAIN_DATASET_LEN = 1000   # Just 1k samples
TEST_DATASET_LEN = 100
BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_WORKERS = 0  # Single process for debugging
```

### High-Performance Training

```python
TRAIN_DATASET_LEN = None
BATCH_SIZE = 16      # If GPU memory allows
NUM_WORKERS = 8      # Parallel data loading
USE_AMP = True       # Mixed precision
LEARNING_RATE = 1e-4
```

---

## 📁 Updated Files

### New Files:
1. **`utils/loader.py`** - Enhanced with metadata-based loading
   - Added `SpikeEEGMetadataDataset` class
   - Added `create_dataloaders_from_metadata()` function
   - Kept `SpikeEEGDataset` for dynamic generation mode

2. **`test_metadata_loader.py`** - Test suite for metadata loader
   - Checks data structure
   - Tests dataset creation
   - Tests dataloader batching

3. **`METADATA_LOADER_README.md`** - Comprehensive documentation
   - Usage guide
   - API reference
   - Troubleshooting

4. **`FINAL_SETUP_GUIDE.md`** - This file!

### Modified Files:
1. **`configs/config.py`** - Added metadata loader settings
2. **`train.py`** - Supports both loader modes (metadata vs dynamic)
3. **`utils/data_utils.py`** - Helper functions

---

## 🎮 Two Loader Modes Available

### Mode 1: Metadata-Based (Recommended for Production)

```python
# configs/config.py
USE_METADATA_LOADER = True
```

**Advantages:**
- ✅ Uses your curated dataset metadata
- ✅ Reproducible (same samples every time)
- ✅ Full control over source configurations
- ✅ Direct training from metadata files

**Use when:** Training with your ~190k curated samples

### Mode 2: Dynamic Generation

```python
# configs/config.py
USE_METADATA_LOADER = False
```

**Advantages:**
- ✅ No metadata files needed
- ✅ Infinite variations
- ✅ Quick experimentation

**Use when:** Quick tests or when metadata not available

---

## 🧪 Testing

### Test the Metadata Loader

```bash
python test_metadata_loader.py
```

**Expected output:**
```
======================================================================
METADATA-BASED LOADER TEST SUITE
======================================================================

Checking Data Structure
----------------------------------------------------------------------
✓ source/train_sample_source1.mat
✓ source/test_sample_source1.mat
✓ anatomy/leadfield_75_20k.mat
✓ source/nmm_spikes
✓ source/nmm_spikes/a0
✓ source/nmm_spikes/a1

Testing SpikeEEGMetadataDataset
----------------------------------------------------------------------
Loading metadata from source/train_sample_source1.mat...
Dataset length: 10
Found 2 regions with data
✓ Metadata dataset test passed!

Testing create_dataloaders_from_metadata  
----------------------------------------------------------------------
Train batches: 2
Val batches: 1
Test batches: 3
✓ Dataloader test passed!

TEST SUMMARY
----------------------------------------------------------------------
✓ Data structure check: PASSED
✓ Dataset test: PASSED
✓ Dataloader test: PASSED

🎉 All tests passed!
```

---

## 🚂 Training

### Start Training

```bash
python train.py
```

### Monitor Progress

```bash
# View TensorBoard
tensorboard --logdir logs/

# Check checkpoints
ls checkpoints/
```

### Command Line Options

```bash
# Custom batch size
python train.py --batch_size 16

# Custom epochs
python train.py --epochs 50

# Custom learning rate
python train.py --lr 5e-5

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pt
```

---

## 📈 Expected Performance

### Training Dataset

- **Total samples**: ~190,656 (from metadata)
- **Validation split**: 10% (configurable)
- **Test samples**: From test_sample_source1.mat
- **Batch size**: 8 (default)
- **Epochs**: 100 (default)

### Memory Usage

- **Per sample**: ~4 MB
- **Per batch (8)**: ~32 MB
- **Model**: ~50 MB
- **Total**: ~500 MB - 1 GB (comfortable for most GPUs)

### Training Speed

- **CPU**: ~10-20 samples/sec
- **GPU**: ~100-500 samples/sec (depends on GPU)
- **Full epoch**: ~10-60 minutes (depends on hardware)

---

## ⚠️ Common Issues & Solutions

### Issue 1: "No mat files found"

**Cause**: `source/nmm_spikes/` directory missing or empty

**Solution**:
```bash
ls source/nmm_spikes/a0/  # Should show nmm_*.mat files
ls source/nmm_spikes/a1/  # Should show nmm_*.mat files
```

### Issue 2: "Could not find forward matrix"

**Cause**: Wrong field name in leadfield file

**Solution**:
```python
from scipy.io import loadmat
data = loadmat('anatomy/leadfield_75_20k.mat')
print(data.keys())  # Check available fields
```
File should contain: `fwd`, `forward`, `leadfield`, or `L`

### Issue 3: Slow loading

**Cause**: Too few workers

**Solution**:
```python
# configs/config.py
NUM_WORKERS = 8  # Increase for faster loading
```

### Issue 4: Out of memory

**Cause**: Batch size too large

**Solution**:
```python
# configs/config.py
BATCH_SIZE = 4  # Reduce batch size
```

---

## 📚 Documentation

- **Quick start**: This file
- **Detailed guide**: `METADATA_LOADER_README.md`
- **Data structure**: `DATA_STRUCTURE_README.md`
- **All changes**: `CHANGES_SUMMARY.md`

---

## ✅ Verification Checklist

Before training, verify:

- [ ] Metadata files exist and are readable
- [ ] NMM spike files exist in `source/nmm_spikes/`
- [ ] Forward matrix exists and has correct shape
- [ ] `configs/config.py` paths are correct
- [ ] `USE_METADATA_LOADER = True` in config
- [ ] Test script passes: `python test_metadata_loader.py`

---

## 🎯 Recommended Workflow

### 1. First Time Setup

```bash
# Test the loader
python test_metadata_loader.py

# Quick training test (small dataset)
# Edit config: TRAIN_DATASET_LEN = 1000
python train.py

# Check results
tensorboard --logdir logs/
```

### 2. Full Training

```bash
# Edit config: TRAIN_DATASET_LEN = None (use all samples)
python train.py

# Monitor in real-time
tensorboard --logdir logs/
```

### 3. Resume Training

```bash
# If training interrupted
python train.py --resume checkpoints/checkpoint_epoch_X.pt
```

---

## 🔍 Understanding Your Data

### Metadata Structure

Your `train_sample_source1.mat` contains:

```
selected_region: [190656 × num_sources × 70]
  - Which regions are active per sample
  
nmm_idx: [190656 × num_sources]
  - Which NMM file to load
  
scale_ratio: [190656 × num_sources × num_snr_levels]
  - Amplitude scaling factors
  
mag_change: [190656 × num_sources × 70]
  - Spatial decay patterns
  
current_snr: [190656]
  - Noise levels (dB)
```

### NMM Spike Files

Located in `source/nmm_spikes/`:
- `a0/nmm_1.mat`, `nmm_2.mat`, `nmm_3.mat`
- `a1/nmm_1.mat` to `nmm_24.mat`
- Each contains `data` array: `(time_points, 994)`

---

## 🎓 Next Steps

1. **Test**: Run `python test_metadata_loader.py`
2. **Configure**: Edit `configs/config.py` if needed
3. **Train**: Run `python train.py`
4. **Evaluate**: Check performance on test set
5. **Iterate**: Adjust hyperparameters and retrain

---

## 💡 Tips

- Start with a small `TRAIN_DATASET_LEN` (1000) to verify everything works
- Use `NUM_WORKERS = 0` for debugging
- Enable `USE_AMP = True` for faster training on GPU
- Monitor TensorBoard for loss curves
- Save best checkpoints automatically

---

## 🎉 Summary

You now have a complete system that:

✅ Loads data directly from metadata files  
✅ No pre-extraction needed  
✅ Memory efficient  
✅ Fast and flexible  
✅ Production ready  

Just run `python train.py` and you're good to go!

---

## 📧 Need Help?

1. Check `METADATA_LOADER_README.md` for detailed docs
2. Run `python test_metadata_loader.py` for diagnostics
3. Review error messages in console
4. Verify paths in `configs/config.py`

Happy training! 🚀

