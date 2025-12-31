# Complete Guide: Improving Prediction Results & Reducing Loss

This guide provides comprehensive strategies to improve your model's performance using GPU acceleration.

## Table of Contents
1. [Quick Start (GPU Training)](#quick-start-gpu-training)
2. [Configuration Options](#configuration-options)
3. [Model Architecture Improvements](#model-architecture-improvements)
4. [Training Strategies](#training-strategies)
5. [Monitoring & Optimization](#monitoring--optimization)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start (GPU Training)

### Option 1: Optimized Configuration (Best Performance)

```bash
# Use GPU-optimized config with enhanced model
python train_optimized.py --config optimized --data_dir ./dataset_with_label

# This uses:
# - Larger model (D_MODEL=512, LAYERS=8)
# - Bigger batch size (16)
# - Cosine annealing learning rate
# - Model EMA for stable predictions
# - 200 epochs for better convergence
```

### Option 2: Balanced Configuration (Moderate GPU)

```bash
# Use balanced config for moderate GPU memory
python train_optimized.py --config balanced --data_dir ./dataset_with_label

# This uses:
# - Medium model (D_MODEL=384, LAYERS=6)
# - Moderate batch size (12)
# - Balanced between speed and performance
# - 150 epochs
```

### Option 3: Custom Training

```bash
# Override specific parameters
python train_optimized.py \
    --config optimized \
    --data_dir ./dataset_with_label \
    --batch_size 32 \
    --epochs 300 \
    --lr 1e-4
```

---

## Configuration Options

### Available Configurations

#### 1. **ConfigGPUOptimized** - Maximum Performance
Best for: Good GPU (8GB+ VRAM), want best results

```python
# Key parameters:
D_MODEL = 512           # Large model capacity
NUM_LAYERS = 8          # Deep network
BATCH_SIZE = 16         # Large batches for stable gradients
LEARNING_RATE = 1e-4    # Aggressive learning
NUM_EPOCHS = 200        # Long training for convergence
```

**Expected GPU Usage:** ~6-8 GB VRAM
**Training Time:** ~2-4 hours (depends on dataset size)
**Best For:** Final model training, maximum accuracy

#### 2. **ConfigBalanced** - Good Tradeoff
Best for: Moderate GPU (4-6GB VRAM), faster experimentation

```python
# Key parameters:
D_MODEL = 384           # Medium capacity
NUM_LAYERS = 6          # Moderate depth
BATCH_SIZE = 12         # Balanced batch size
LEARNING_RATE = 7e-5    # Conservative learning
NUM_EPOCHS = 150        # Moderate training time
```

**Expected GPU Usage:** ~4-5 GB VRAM
**Training Time:** ~1-2 hours
**Best For:** Experimentation, hyperparameter tuning

#### 3. **Original Config** - Stable Baseline
Best for: CPU or limited GPU, debugging

```python
# Use the fixed train.py
python train.py --data_dir ./dataset_with_label
```

**Expected GPU Usage:** ~2-3 GB VRAM
**Training Time:** ~30-60 minutes
**Best For:** Baseline, quick tests

---

## Model Architecture Improvements

### New Enhanced Model (V3)

The new `EEGSourceTransformerV3` includes several improvements:

#### Key Features:
1. **Pre-LayerNorm Architecture** - More stable training than post-LN
2. **GELU Activation** - Better than ReLU for transformers
3. **Skip Connections** - Direct path from input to output
4. **Enhanced Output Projection** - Deeper, more expressive
5. **Input Normalization** - Stable across different data distributions
6. **Conservative Initialization** - Prevents early divergence

#### Available Model Sizes:

**Small** - Fast experimentation (3.5M params)
```python
from models.transformer_model_v3 import EEGSourceTransformerV3Small
model = EEGSourceTransformerV3Small()
```

**Default** - Good balance (18M params)
```python
from models.transformer_model_v3 import EEGSourceTransformerV3
model = EEGSourceTransformerV3()
```

**Large** - Maximum performance (65M params)
```python
from models.transformer_model_v3 import EEGSourceTransformerV3Large
model = EEGSourceTransformerV3Large()
```

### To Use Enhanced Model:

Modify `train_optimized.py` line ~240:
```python
# Replace:
from models.transformer_model import EEGSourceTransformerV2

# With:
from models.transformer_model_v3 import EEGSourceTransformerV3

# And update model initialization:
model = EEGSourceTransformerV3(
    eeg_channels=config.EEG_CHANNELS,
    source_regions=config.SOURCE_REGIONS,
    d_model=config.D_MODEL,
    nhead=config.NHEAD,
    num_layers=config.NUM_LAYERS,
    dim_feedforward=config.DIM_FEEDFORWARD,
    dropout=config.DROPOUT
).to(config.DEVICE)
```

---

## Training Strategies

### 1. Progressive Training (Recommended)

Train in stages for best results:

**Stage 1: Quick Validation (1-2 hours)**
```bash
python train_optimized.py --config balanced --epochs 50
```
- Verify training is stable
- Check if loss is decreasing
- Identify any issues early

**Stage 2: Medium Training (3-4 hours)**
```bash
python train_optimized.py --config optimized --epochs 150
```
- Better model capacity
- More training time
- Good results

**Stage 3: Final Training (6-8 hours)**
```bash
python train_optimized.py --config optimized --epochs 300 --lr 8e-5
```
- Maximum epochs
- Fine-tuned learning rate
- Best possible results

### 2. Learning Rate Strategies

#### Cosine Annealing with Warm Restarts (Default)
```python
USE_COSINE_ANNEALING = True
T_0 = 20              # Restart every 20 epochs
T_MULT = 2            # Double period after each restart
ETA_MIN = 1e-6        # Minimum LR
```

**Benefits:**
- Escapes local minima with periodic restarts
- Smooth annealing for convergence
- Proven effective for transformers

#### Finding Optimal Learning Rate

Run learning rate finder:
```bash
# Test different learning rates
python train_optimized.py --epochs 10 --lr 1e-5
python train_optimized.py --epochs 10 --lr 5e-5
python train_optimized.py --epochs 10 --lr 1e-4
python train_optimized.py --epochs 10 --lr 5e-4
```

Look for:
- Loss decreases steadily: ✓ Good LR
- Loss plateaus: LR too small
- Loss explodes/NaN: LR too large

### 3. Model EMA (Exponential Moving Average)

Already enabled in `train_optimized.py`:
```python
USE_EMA = True
EMA_DECAY = 0.999
```

**Benefits:**
- More stable predictions
- Better generalization
- Smoother training curves

The EMA model is automatically used for validation and saved in checkpoints.

### 4. Gradient Accumulation (For limited GPU memory)

If you get out-of-memory errors:

```python
# In config:
BATCH_SIZE = 4          # Reduce batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Add this

# Effective batch size = 4 * 4 = 16
```

### 5. Mixed Precision Training

Already enabled automatically on GPU:
```python
USE_AMP = True  # Automatic Mixed Precision
```

**Benefits:**
- 2-3x faster training
- Reduces memory usage
- No accuracy loss (usually slight improvement)

---

## Monitoring & Optimization

### 1. Monitor Training with TensorBoard

```bash
# In a separate terminal:
tensorboard --logdir logs_gpu_optimized

# Open browser to: http://localhost:6006
```

**What to Monitor:**
- **Loss curves**: Should decrease steadily
- **Learning rate**: Should follow cosine schedule
- **Gradient norms**: Should stay < 10
- **MAE (Mean Absolute Error)**: Lower is better

### 2. Validate Checkpoints

```bash
# Check all checkpoints are valid
python -m utils.checkpoint_validator --dir checkpoints_gpu_optimized

# Inspect specific checkpoint
python -m utils.checkpoint_validator --checkpoint checkpoints_gpu_optimized/best_model.pt
```

### 3. GPU Utilization

Monitor GPU usage during training:
```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or use:
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

**Target GPU Utilization:** 80-95%
- Too low (<50%): Increase batch size or model size
- Too high (>98%): Risk of OOM, reduce batch size

### 4. Expected Loss Values

For EEG source localization (MSE loss):

| Epoch | Train Loss | Val Loss | Quality |
|-------|-----------|----------|---------|
| 10    | 0.01-0.1  | 0.01-0.1 | Initial |
| 50    | 0.001-0.01| 0.001-0.01| Good |
| 100   | 0.0001-0.001 | 0.0005-0.002 | Very Good |
| 200+  | <0.0001   | <0.001   | Excellent |

**Note:** Your values may differ based on data and normalization.

---

## Advanced Optimizations

### 1. Data Augmentation (If needed)

Add to `utils/dataset.py`:
```python
def augment_eeg(eeg_data):
    """Simple augmentation for EEG data"""
    # Add small Gaussian noise
    noise = torch.randn_like(eeg_data) * 0.01
    augmented = eeg_data + noise
    
    # Random scaling
    scale = 0.9 + torch.rand(1) * 0.2  # 0.9 to 1.1
    augmented = augmented * scale
    
    return augmented
```

### 2. Curriculum Learning

Train on easier examples first:
```python
# Sort dataset by complexity
# Start with shorter sequences or simpler patterns
# Gradually introduce harder examples
```

### 3. Multi-GPU Training

For multiple GPUs:
```bash
# Use DataParallel
python train_optimized.py --config optimized

# Or DistributedDataParallel (faster):
python -m torch.distributed.launch --nproc_per_node=2 train_optimized.py
```

### 4. Hyperparameter Tuning

Systematic approach:
```bash
# Test different configurations
for lr in 5e-5 8e-5 1e-4; do
    for bs in 12 16 24; do
        python train_optimized.py --lr $lr --batch_size $bs --epochs 50
    done
done
```

---

## Troubleshooting

### Issue 1: Training is Slow

**Solutions:**
1. Verify GPU is being used:
   ```python
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

2. Increase batch size if memory allows:
   ```bash
   python train_optimized.py --batch_size 24
   ```

3. Reduce num_workers if CPU bottleneck:
   ```python
   NUM_WORKERS = 4  # Try lower values
   ```

### Issue 2: Loss Not Decreasing

**Solutions:**
1. **Check data normalization:**
   ```python
   # Verify statistics are computed correctly
   # Check for outliers in data
   ```

2. **Increase learning rate:**
   ```bash
   python train_optimized.py --lr 2e-4
   ```

3. **Increase model capacity:**
   ```python
   # Use larger model
   D_MODEL = 768
   NUM_LAYERS = 12
   ```

4. **Train longer:**
   ```bash
   python train_optimized.py --epochs 300
   ```

### Issue 3: Out of Memory (OOM)

**Solutions:**
1. **Reduce batch size:**
   ```bash
   python train_optimized.py --batch_size 8
   ```

2. **Use balanced config:**
   ```bash
   python train_optimized.py --config balanced
   ```

3. **Enable gradient checkpointing:**
   ```python
   # In model: trade compute for memory
   torch.utils.checkpoint.checkpoint(layer, x)
   ```

4. **Reduce model size:**
   ```python
   D_MODEL = 256
   NUM_LAYERS = 4
   ```

### Issue 4: Overfitting (Val Loss >> Train Loss)

**Solutions:**
1. **Increase dropout:**
   ```python
   DROPOUT = 0.2  # From 0.15
   ```

2. **Increase weight decay:**
   ```python
   WEIGHT_DECAY = 5e-4  # From 1e-4
   ```

3. **Use more data augmentation**

4. **Reduce model capacity**

5. **Add early stopping:**
   ```python
   PATIENCE = 20  # Stop if no improvement
   ```

### Issue 5: NaN Loss (Still occurring)

**Solutions:**
1. **Reduce learning rate:**
   ```bash
   python train_optimized.py --lr 1e-5
   ```

2. **Increase gradient clipping:**
   ```python
   CLIP_GRAD_NORM = 0.5  # From 1.0
   ```

3. **Check data for anomalies:**
   ```bash
   python -c "import scipy.io as sio; import numpy as np; data = sio.loadmat('dataset_with_label/sample_00000.mat'); print('EEG range:', np.min(data['eeg_data']), np.max(data['eeg_data']))"
   ```

---

## Performance Comparison

Expected improvements with optimizations:

| Configuration | Model Size | Train Time | Expected Val Loss | Memory |
|--------------|------------|------------|-------------------|---------|
| Original (CPU) | 6M params | 2-3 hrs | 0.001-0.01 | 2 GB |
| Balanced (GPU) | 12M params | 1-2 hrs | 0.0005-0.002 | 4-5 GB |
| Optimized (GPU) | 18M params | 2-3 hrs | 0.0001-0.001 | 6-8 GB |
| V3 Large (GPU) | 65M params | 4-6 hrs | <0.0001 | 10-12 GB |

---

## Best Practices Summary

### For Best Results:

1. ✓ **Start with balanced config** to verify everything works
2. ✓ **Use optimized config** for final training  
3. ✓ **Train for 150-200 epochs** minimum
4. ✓ **Monitor with TensorBoard** to catch issues early
5. ✓ **Validate checkpoints** regularly
6. ✓ **Use Model EMA** (already enabled)
7. ✓ **Save best model** based on validation loss
8. ✓ **Keep multiple checkpoints** for comparison

### Recommended Workflow:

```bash
# Step 1: Clean old checkpoints
python -m utils.cleanup_checkpoints --no-dry-run

# Step 2: Verify GPU
python -c "from configs.config_gpu_optimized import ConfigGPUOptimized; ConfigGPUOptimized.verify_gpu()"

# Step 3: Quick test (10 epochs)
python train_optimized.py --config balanced --epochs 10

# Step 4: Medium training (if test looks good)
python train_optimized.py --config balanced --epochs 100

# Step 5: Full training (for best results)
python train_optimized.py --config optimized --epochs 200

# Step 6: Validate results
python -m utils.checkpoint_validator --dir checkpoints_gpu_optimized
```

---

## Next Steps

After training completes:

1. **Evaluate on test set:**
   ```bash
   python evaluate.py --checkpoint checkpoints_gpu_optimized/best_model.pt
   ```

2. **Test on real data:**
   ```bash
   python eval_real.py --checkpoint checkpoints_gpu_optimized/best_model.pt --data_dir source --subjects VEP
   ```

3. **Compare predictions:**
   - Check MAE values
   - Visualize predictions vs ground truth
   - Analyze error patterns

4. **If results not satisfactory:**
   - Try V3 Large model
   - Train for more epochs (300-500)
   - Adjust learning rate
   - Check data quality

---

## Support

If you encounter issues:

1. Check logs in `logs_gpu_optimized/`
2. Validate checkpoints with `checkpoint_validator`
3. Monitor GPU usage with `nvidia-smi`
4. Review TensorBoard curves
5. Try balanced config first

Remember: Deep learning is iterative. Start simple, validate, then scale up!


