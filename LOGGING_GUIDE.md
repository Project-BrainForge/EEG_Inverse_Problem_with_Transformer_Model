# 📝 Training Logging Guide

## Overview

The training system now includes comprehensive logging to track errors, warnings, and training progress. All logs are automatically saved to files for later analysis.

---

## 🎯 Log Files

### Location

All logs are saved in the `logs/` directory with timestamps:

```
logs/
├── training_20241225_143052_all.log      # All logs (DEBUG level and above)
├── training_20241225_143052_errors.log   # Errors only (ERROR level)
└── ...
```

### File Types

1. **`training_YYYYMMDD_HHMMSS_all.log`**
   - Contains ALL log messages (DEBUG, INFO, WARNING, ERROR)
   - Detailed format with function names and line numbers
   - Use for detailed debugging

2. **`training_YYYYMMDD_HHMMSS_errors.log`**
   - Contains ONLY errors (ERROR level)
   - Quick reference for what went wrong
   - Use for error analysis

---

## 📊 What Gets Logged

### Configuration

```
2024-12-25 14:30:52 - INFO - Training Configuration
2024-12-25 14:30:52 - INFO - EEG_CHANNELS: 75
2024-12-25 14:30:52 - INFO - SOURCE_REGIONS: 994
2024-12-25 14:30:52 - INFO - BATCH_SIZE: 8
...
```

### Dataset Loading

```
2024-12-25 14:30:53 - INFO - Using metadata-based loader
2024-12-25 14:30:55 - INFO - Datasets loaded: train=2000 batches, val=250 batches, test=625 batches
```

### Model Initialization

```
2024-12-25 14:30:56 - INFO - Initializing model...
2024-12-25 14:30:57 - INFO - Total parameters: 6,040,034
2024-12-25 14:30:57 - INFO - Trainable parameters: 6,040,034
2024-12-25 14:30:57 - INFO - Device: cuda:0
```

### Training Progress

```
2024-12-25 14:31:00 - INFO - Starting epoch 1/100
2024-12-25 14:35:42 - INFO - Epoch 1/100 - Train Loss: 0.031246, Val Loss: 0.008482, Val MAE: 0.002134, LR: 0.000100, Time: 282.15s
2024-12-25 14:35:42 - INFO - New best model saved with val_loss: 0.008482
```

### Batch-Level Details (Every 100 batches)

```
2024-12-25 14:32:15 - DEBUG - train_epoch:95 - Epoch 1, Batch 100/2000, Loss: 0.034521
2024-12-25 14:33:30 - DEBUG - train_epoch:95 - Epoch 1, Batch 200/2000, Loss: 0.029876
```

### Errors

```
2024-12-25 14:35:00 - ERROR - train_epoch:75 - NaN loss detected at epoch 1, batch 523
2024-12-25 14:35:00 - ERROR - train_epoch:76 - EEG data stats: min=-1.234, max=5.678, mean=0.123
2024-12-25 14:35:00 - ERROR - train_epoch:77 - Source data stats: min=0.0, max=0.999, mean=0.045
```

### Warnings

```
2024-12-25 14:40:25 - WARNING - train_epoch:105 - Epoch 1: 5 batches failed out of 2000
```

### Checkpoints

```
2024-12-25 14:40:30 - INFO - Periodic checkpoint saved at epoch 5
2024-12-25 15:20:15 - INFO - New best model saved with val_loss: 0.005123
```

### Early Stopping

```
2024-12-25 16:45:30 - INFO - Early stopping triggered after 45 epochs
```

### Test Evaluation

```
2024-12-25 16:45:35 - INFO - Evaluating on test set...
2024-12-25 16:48:20 - INFO - Test Loss: 0.005234, Test MAE: 0.001987
```

### Completion

```
2024-12-25 16:48:22 - INFO - Final model saved
2024-12-25 16:48:22 - INFO - Training completed successfully!
```

---

## 🔍 Log Levels

### DEBUG
- Batch-level details (every 100 batches)
- Detailed function execution
- Only saved to `*_all.log`

### INFO
- Normal training progress
- Epoch summaries
- Model saves
- Configuration
- Saved to both files

### WARNING
- Failed batches (but training continues)
- Non-fatal issues
- Saved to both files

### ERROR
- Errors during training
- NaN losses
- Failed operations
- Saved to both files + console

---

## 📖 Reading Logs

### View All Logs in Real-Time

```bash
# Follow all logs
tail -f logs/training_20241225_143052_all.log

# Follow errors only
tail -f logs/training_20241225_143052_errors.log
```

### Search for Specific Issues

```bash
# Find all errors
grep "ERROR" logs/training_20241225_143052_all.log

# Find NaN losses
grep "NaN" logs/training_20241225_143052_all.log

# Find warnings
grep "WARNING" logs/training_20241225_143052_all.log

# Find specific epoch
grep "Epoch 10/" logs/training_20241225_143052_all.log
```

### Count Errors

```bash
# Count total errors
grep -c "ERROR" logs/training_20241225_143052_errors.log

# Count NaN losses
grep -c "NaN loss" logs/training_20241225_143052_errors.log
```

### View Last N Lines

```bash
# Last 50 lines of all logs
tail -n 50 logs/training_20241225_143052_all.log

# Last 20 errors
tail -n 20 logs/training_20241225_143052_errors.log
```

---

## 🛠️ Debugging with Logs

### Problem: Training Crashed

**Step 1**: Check error log
```bash
cat logs/training_YYYYMMDD_HHMMSS_errors.log
```

**Step 2**: Find last successful epoch
```bash
grep "Epoch.*completed" logs/training_YYYYMMDD_HHMMSS_all.log | tail -n 1
```

**Step 3**: Check what happened next
```bash
# Get context around the last epoch
grep -A 20 "Epoch 10/100" logs/training_YYYYMMDD_HHMMSS_all.log
```

### Problem: NaN Losses

**Find when it started**:
```bash
grep "NaN loss" logs/training_YYYYMMDD_HHMMSS_errors.log | head -n 1
```

**Get data statistics**:
```bash
grep -A 2 "NaN loss detected" logs/training_YYYYMMDD_HHMMSS_errors.log
```

### Problem: Slow Training

**Check batch times**:
```bash
# Get timing for each epoch
grep "Time:" logs/training_YYYYMMDD_HHMMSS_all.log
```

**Check for warnings**:
```bash
grep "WARNING" logs/training_YYYYMMDD_HHMMSS_all.log
```

### Problem: Model Not Improving

**Track validation loss**:
```bash
grep "Val Loss:" logs/training_YYYYMMDD_HHMMSS_all.log
```

**Check learning rate changes**:
```bash
grep "LR:" logs/training_YYYYMMDD_HHMMSS_all.log
```

---

## 📊 Analyzing Training

### Extract Epoch Summary

```bash
# Get all epoch summaries
grep "Epoch [0-9]*/[0-9]* - Train Loss" logs/training_YYYYMMDD_HHMMSS_all.log > epoch_summary.txt
```

### Create Loss Plot Data

```bash
# Extract training losses
grep "Train Loss:" logs/training_YYYYMMDD_HHMMSS_all.log | \
  awk '{print $7}' | sed 's/,$//' > train_losses.txt

# Extract validation losses
grep "Val Loss:" logs/training_YYYYMMDD_HHMMSS_all.log | \
  awk '{print $10}' | sed 's/,$//' > val_losses.txt
```

### Count Failed Batches Per Epoch

```bash
grep "batches failed" logs/training_YYYYMMDD_HHMMSS_all.log
```

---

## 🚨 Error Types and Solutions

### NaN Loss Detected

**Log Entry**:
```
ERROR - NaN loss detected at epoch 5, batch 234
ERROR - EEG data stats: min=-inf, max=inf, mean=nan
```

**Possible Causes**:
- Learning rate too high
- Data contains NaN or Inf values
- Gradient explosion

**Solutions**:
1. Reduce learning rate
2. Check data preprocessing
3. Enable gradient clipping
4. Check forward matrix for invalid values

### Data Loading Error

**Log Entry**:
```
ERROR - Error in batch 123 at epoch 2: Could not load NMM data
```

**Possible Causes**:
- Missing data files
- Corrupted .mat files
- Incorrect file paths

**Solutions**:
1. Verify all data files exist
2. Check file permissions
3. Validate .mat file format

### Out of Memory

**Log Entry**:
```
ERROR - Error in epoch 1: CUDA out of memory
```

**Solutions**:
1. Reduce batch size in `config.py`
2. Reduce model size (D_MODEL, NUM_LAYERS)
3. Enable gradient checkpointing
4. Use CPU if GPU memory insufficient

### Validation Error

**Log Entry**:
```
ERROR - Error in validation batch 45 at epoch 3
WARNING - Validation epoch 3: 12 batches failed out of 250
```

**Solutions**:
1. Check validation data integrity
2. Review error details in errors log
3. May indicate overfitting if only val fails

---

## 🎯 Best Practices

### 1. Monitor Logs During Training

```bash
# In one terminal: run training
python train.py

# In another terminal: follow logs
tail -f logs/training_*_all.log
```

### 2. Check for Errors Immediately

```bash
# After training, check error log
cat logs/training_*_errors.log
```

### 3. Archive Important Runs

```bash
# Save logs for important runs
mkdir -p archived_logs/run_20241225
cp logs/training_20241225_* archived_logs/run_20241225/
```

### 4. Compare Runs

```bash
# Extract key metrics from multiple runs
for log in logs/training_*_all.log; do
    echo "=== $log ==="
    grep "Best val_loss" $log
done
```

### 5. Clean Old Logs

```bash
# Remove logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete
```

---

## 📈 Log Format

### Detailed Format (*_all.log)

```
YYYY-MM-DD HH:MM:SS - LEVEL - function_name:line_number - message
```

Example:
```
2024-12-25 14:30:52 - ERROR - train_epoch:75 - NaN loss detected at epoch 1, batch 523
```

### Simple Format (Console)

```
YYYY-MM-DD HH:MM:SS - LEVEL - message
```

Example:
```
2024-12-25 14:30:52 - INFO - Training completed successfully!
```

---

## 🔧 Customizing Logging

### Change Log Level

Edit `train.py`, `setup_logger()` function:

```python
# More verbose (show DEBUG messages on console)
console_handler.setLevel(logging.DEBUG)

# Less verbose (only show warnings and errors)
console_handler.setLevel(logging.WARNING)
```

### Change Log Location

Edit `configs/config.py`:

```python
LOG_DIR = "my_custom_logs"
```

### Add Custom Logging

In your code:

```python
from logging import getLogger
logger = getLogger('EEG_Training')

# In your function
logger.info("Custom message")
logger.warning("Something unusual happened")
logger.error("An error occurred")
logger.debug("Detailed debug info")
```

---

## 📚 Quick Reference

| Task | Command |
|------|---------|
| View all logs | `tail -f logs/training_*_all.log` |
| View errors only | `tail -f logs/training_*_errors.log` |
| Count errors | `grep -c ERROR logs/training_*_errors.log` |
| Find NaN losses | `grep "NaN" logs/training_*_all.log` |
| Extract epoch summaries | `grep "Epoch [0-9]*/[0-9]*" logs/training_*_all.log` |
| Last 50 lines | `tail -n 50 logs/training_*_all.log` |
| Search term | `grep "search_term" logs/training_*_all.log` |

---

## 🎉 Summary

The logging system provides:

✅ **Automatic error tracking** - All errors saved to dedicated file  
✅ **Detailed debugging** - Function names and line numbers  
✅ **Progress monitoring** - Real-time training updates  
✅ **Issue diagnosis** - NaN detection, failed batches, warnings  
✅ **Historical record** - Timestamped files for each run  
✅ **Easy analysis** - Structured format for grep/awk processing  

Check your logs regularly to catch issues early!

