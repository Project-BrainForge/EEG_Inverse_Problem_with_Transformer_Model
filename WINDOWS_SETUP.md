# 🪟 Windows Setup Guide

## Common Windows-Specific Issues

### Issue 1: Multiprocessing Error (OSError: [Errno 22] Invalid argument)

**Error Message:**
```
OSError: [Errno 22] Invalid argument
_pickle.UnpicklingError: pickle data was truncated
```

**Cause:**
Windows uses a different multiprocessing method (`spawn`) than Unix systems (`fork`), which requires pickling all dataset objects. PyTorch DataLoader with `num_workers > 0` can fail on Windows.

**Solution:**
Set `NUM_WORKERS = 0` in `configs/config.py`:

```python
# Training settings
NUM_WORKERS = 0  # Must be 0 on Windows
```

The training script now **automatically detects Windows** and sets `NUM_WORKERS=0`, but you should update your config to avoid the warning.

**Performance Impact:**
- Single-threaded data loading (slower)
- For Windows, this is the only stable option
- Still works fine for training, just slightly slower between batches

---

### Issue 2: CUDA/GPU on Windows

**If you have an NVIDIA GPU:**

1. **Check CUDA availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

2. **Install correct PyTorch version:**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio
```

3. **Update config:**
```python
# configs/config.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**If training on CPU:**
- Set `USE_AMP = False` in config (mixed precision only works with GPU)
- Expect slower training (10-50x slower than GPU)

---

### Issue 3: File Path Issues

**Problem:** Paths with spaces or special characters

**Solution:** Use raw strings or forward slashes:

```python
# Good
DATA_DIR = r"D:\fyp\EEG_Inverse_Problem_with_Transformer_Model"
# or
DATA_DIR = "D:/fyp/EEG_Inverse_Problem_with_Transformer_Model"

# Bad (backslash issues)
DATA_DIR = "D:\fyp\..."  # May cause escape sequence issues
```

---

### Issue 4: Long Path Names

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Cause:** Windows has a 260 character path limit by default

**Solution 1:** Enable long paths in Windows:
1. Open Registry Editor (regedit)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart computer

**Solution 2:** Use shorter paths:
```python
# Instead of
DATA_DIR = "D:/very/long/path/with/many/nested/folders/..."

# Use
DATA_DIR = "D:/data"
```

---

### Issue 5: Memory Issues

**Symptoms:**
- Training crashes without clear error
- "Out of memory" errors
- Computer becomes unresponsive

**Solutions:**

1. **Reduce batch size:**
```python
BATCH_SIZE = 4  # or even 2
```

2. **Close other applications**

3. **Monitor memory usage:**
```python
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```

4. **Use gradient accumulation** (in `train.py`):
```python
# Simulate larger batch size
ACCUMULATION_STEPS = 4
if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

### Issue 6: TensorBoard on Windows

**Starting TensorBoard:**
```bash
# In Command Prompt or PowerShell
tensorboard --logdir=logs

# If port already in use
tensorboard --logdir=logs --port=6007
```

**Accessing TensorBoard:**
Open browser and go to: `http://localhost:6006`

---

## Recommended Windows Settings

### `configs/config.py` for Windows:

```python
import torch

class Config:
    # Data parameters
    USE_METADATA_LOADER = True
    TRAIN_METADATA_PATH = "source/train_sample_source1.mat"
    TEST_METADATA_PATH = "source/test_sample_source1.mat"
    NMM_SPIKES_DIR = "source/nmm_spikes"
    FWD_MATRIX_PATH = "anatomy/leadfield_75_20k.mat"
    
    # Model parameters
    EEG_CHANNELS = 75
    SOURCE_REGIONS = 994
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 4  # Smaller for Windows RAM constraints
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # Windows-specific settings
    NUM_WORKERS = 0  # MUST BE 0 ON WINDOWS
    PIN_MEMORY = False  # Set to False if not using GPU
    USE_AMP = False  # Set to False if using CPU
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Performance Tips for Windows

### 1. Use SSD for Data Storage
Store your data on an SSD rather than HDD for faster loading

### 2. Disable Windows Defender During Training
Temporarily disable real-time scanning:
- Windows Security → Virus & threat protection → Manage settings
- Turn off "Real-time protection" (remember to turn it back on!)

### 3. Set High Performance Power Plan
- Control Panel → Power Options → High Performance

### 4. Close Background Applications
- Chrome/Edge browsers (memory hogs)
- OneDrive syncing
- Windows Update
- Antivirus software

### 5. Use Task Manager to Monitor
- Press `Ctrl+Shift+Esc`
- Monitor CPU, RAM, GPU usage
- Close processes consuming too many resources

---

## Troubleshooting Commands

### Check Python Environment
```bash
python --version
pip list | findstr torch
pip list | findstr numpy
```

### Check CUDA
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check File Paths
```bash
dir source\train_sample_source1.mat
dir anatomy\leadfield_75_20k.mat
dir source\nmm_spikes\a0
```

### Clean Up Space
```bash
# Remove old checkpoints
del /Q checkpoints\checkpoint_epoch_*.pt

# Remove old logs (older than 7 days)
forfiles /P logs /S /D -7 /C "cmd /c del @path"
```

---

## Training Workflow on Windows

### 1. Setup (One Time)
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_metadata_loader.py
```

### 2. Configure
Edit `configs/config.py`:
- Set `NUM_WORKERS = 0`
- Set appropriate `BATCH_SIZE` (start with 4)
- Set `DEVICE` based on GPU availability
- Set `USE_AMP = False` if using CPU

### 3. Train
```bash
# Activate environment
venv\Scripts\activate

# Start training
python train.py

# In another terminal, monitor logs
powershell -Command "Get-Content -Wait logs\training_*_all.log"
```

### 4. Monitor
```bash
# Start TensorBoard
tensorboard --logdir=logs
```

Open: http://localhost:6006

### 5. Evaluate
```bash
python eval_sim.py --checkpoint checkpoints\best_model.pt
```

---

## Quick Fixes

### Training Hangs at Start
**Solution:** Check NUM_WORKERS
```python
NUM_WORKERS = 0  # in config.py
```

### Out of Memory
**Solution:** Reduce batch size
```python
BATCH_SIZE = 2  # in config.py
```

### NaN Loss
**Solution:** Reduce learning rate
```python
LEARNING_RATE = 1e-5  # in config.py
```

### Can't Find Files
**Solution:** Use absolute paths
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_METADATA_PATH = os.path.join(BASE_DIR, "source", "train_sample_source1.mat")
```

---

## Performance Comparison

| Setting | Windows (CPU) | Windows (GPU) | Linux (GPU) |
|---------|---------------|---------------|-------------|
| Batch Size | 2-4 | 8-16 | 16-32 |
| NUM_WORKERS | 0 | 0 | 4-8 |
| Speed | ~5 samples/s | ~50 samples/s | ~100 samples/s |
| RAM Usage | 2-4 GB | 4-8 GB | 4-8 GB |

---

## Getting Help

If you encounter other issues:

1. **Check error logs:**
   ```bash
   type logs\training_*_errors.log
   ```

2. **Check all logs:**
   ```bash
   type logs\training_*_all.log
   ```

3. **Test data loading:**
   ```bash
   python test_metadata_loader.py
   ```

4. **Verify config:**
   ```bash
   python -c "from configs.config import Config; Config.display()"
   ```

---

## Summary

✅ **Always set `NUM_WORKERS = 0` on Windows**  
✅ Use smaller batch sizes (2-4 instead of 8-16)  
✅ Monitor memory usage  
✅ Use SSD for data storage  
✅ Close background applications  
✅ Check logs for errors  
✅ Start with CPU, upgrade to GPU if available  

Training on Windows works perfectly with these settings! 🎉

