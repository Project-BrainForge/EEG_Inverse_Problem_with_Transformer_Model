# Checkpoint NaN Issue - Fix Summary

## Problem Identified

After running `train.py`, checkpoints were being saved with **NaN (Not a Number)** values in model weights and losses, while `best_model.pt` occasionally had valid values but with extremely large losses (in the trillions).

### Root Causes

1. **Gradient Explosion**: Training was experiencing numerical instability leading to exploding gradients
2. **No NaN Detection**: The training loop had no checks to detect or prevent NaN/Inf values
3. **Unsafe Checkpoint Saving**: Checkpoints were saved even when containing NaN/Inf values
4. **Suboptimal Hyperparameters**: Learning rate and initialization were too aggressive

## Fixes Implemented

### 1. NaN/Inf Detection in Training Loop (`train.py`)

Added comprehensive checks at every stage:
- **Input validation**: Check EEG and source data before forward pass
- **Prediction validation**: Check model outputs for NaN/Inf
- **Loss validation**: Check loss values before backward pass
- **Gradient validation**: Check all gradients before optimizer step
- **Gradient norm monitoring**: Warn when gradients are extremely large

```python
def check_for_nan_inf(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        return f"{name} contains NaN"
    if torch.isinf(tensor).any():
        return f"{name} contains Inf"
    return None
```

### 2. Safe Checkpoint Saving (`train.py`)

Modified `save_checkpoint()` to validate before saving:
- Check if train_loss or val_loss is NaN/Inf
- Check all model parameters for NaN/Inf
- Return `False` if validation fails (checkpoint not saved)
- Only save valid checkpoints

### 3. Training Loop Protection (`train.py`)

Added early stopping when NaN/Inf detected:
- Stop training immediately if NaN/Inf occurs
- Print detailed error message
- Prevent further corruption of checkpoints

### 4. Model Numerical Stability (`models/transformer_model.py`)

**Improved Weight Initialization:**
- Changed from standard Xavier to Xavier with `gain=0.5` (more conservative)
- Initialize all biases to zero for stability

**Added Input Clamping:**
```python
# Clamp input to prevent extreme values
eeg_data = torch.clamp(eeg_data, min=-10, max=10)
```

**Added Layer Normalization:**
- Added LayerNorm in output projection for better gradient flow

### 5. Reduced Learning Rate (`configs/config.py`)

Changed learning rate from `1e-4` to `5e-5` for more stable training.

## How to Use

### Before Training

1. **Backup existing good checkpoints** (if any):
   ```bash
   cp checkpoints/best_model.pt checkpoints/best_model_backup.pt
   ```

2. **Remove corrupted checkpoints**:
   ```bash
   rm checkpoints/checkpoint_epoch_*.pt
   ```

### During Training

The training script will now:
- Automatically detect NaN/Inf and stop training
- Print detailed error messages showing where the issue occurred
- Only save valid checkpoints
- Show warnings for very large gradients

### Monitoring for Issues

Watch for these messages during training:
- `ERROR: [tensor] contains NaN/Inf` - Indicates where NaN first appeared
- `WARNING: Very large gradient norm` - Early warning of potential instability
- `WARNING: Not saving checkpoint` - Checkpoint validation failed

### If NaN Still Occurs

If you still encounter NaN values, try:

1. **Further reduce learning rate**:
   ```python
   LEARNING_RATE = 1e-5  # Even more conservative
   ```

2. **Increase gradient clipping**:
   ```python
   CLIP_GRAD_NORM = 0.5  # More aggressive clipping
   ```

3. **Reduce model size**:
   ```python
   D_MODEL = 128  # Smaller model
   NUM_LAYERS = 4  # Fewer layers
   ```

4. **Check data normalization**:
   - Ensure input data is properly normalized
   - Check for outliers in the dataset

## Verification

To verify a checkpoint is valid:

```python
python inspect_checkpoint.py
```

This will show:
- Epoch number
- Train and validation losses
- Whether model weights contain NaN/Inf
- Sample weight statistics

## Files Modified

1. `train.py` - Added NaN detection and safe checkpoint saving
2. `models/transformer_model.py` - Improved numerical stability
3. `configs/config.py` - Reduced learning rate
4. `inspect_checkpoint.py` - Created diagnostic tool

## Expected Behavior After Fix

- Training will stop immediately if NaN/Inf is detected
- Only valid checkpoints will be saved
- Clear error messages will indicate where issues occur
- `best_model.pt` will always contain valid weights
- Periodic checkpoints will only be saved if valid

## Testing

Run a short training session to verify:

```bash
python train.py --data_dir ./dataset_with_label --epochs 10
```

Monitor the output for:
- No NaN/Inf errors
- Stable loss values (not increasing exponentially)
- Successful checkpoint saves
- Reasonable gradient norms (< 10)

