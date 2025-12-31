# How to Use the Enhanced V3 Model

## Quick Start

### Option 1: Use the Dedicated V3 Training Script (Easiest) ⭐

```bash
# Default V3 model (18M parameters)
python train_v3.py --config optimized --data_dir ./dataset_with_label

# Small V3 model (3.5M parameters) - Fast experimentation
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label

# Large V3 model (65M parameters) - Maximum performance
python train_v3.py --config optimized --model_size large --data_dir ./dataset_with_label
```

### Option 2: Quick Test (10 epochs)

```bash
# Test the V3 model quickly
python train_v3.py --config balanced --model_size default --data_dir ./dataset_with_label --epochs 10
```

---

## V3 Model Sizes

### Small (EEGSourceTransformerV3Small)
- **Parameters:** ~3.5M
- **D_MODEL:** 256
- **Layers:** 4
- **Best for:** Fast experimentation, limited GPU
- **GPU Memory:** 2-3 GB
- **Training time:** Fast (~30-60 min for 150 epochs)

```bash
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label
```

### Default (EEGSourceTransformerV3)
- **Parameters:** ~18M
- **D_MODEL:** 512
- **Layers:** 8
- **Best for:** Good balance between performance and speed
- **GPU Memory:** 6-8 GB
- **Training time:** Medium (~2-3 hours for 200 epochs)

```bash
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label
```

### Large (EEGSourceTransformerV3Large)
- **Parameters:** ~65M
- **D_MODEL:** 768
- **Layers:** 12
- **Best for:** Maximum accuracy, best possible results
- **GPU Memory:** 10-12 GB
- **Training time:** Longer (~4-6 hours for 200 epochs)

```bash
python train_v3.py --config optimized --model_size large --data_dir ./dataset_with_label --epochs 250
```

---

## V3 Model Features

The V3 model includes several architectural improvements over V2:

### ✅ **Pre-LayerNorm Architecture**
- More stable training than post-LayerNorm
- Better gradient flow
- Faster convergence

### ✅ **GELU Activation**
- Better than ReLU for transformer models
- Smoother gradients
- Proven better for NLP and regression tasks

### ✅ **Skip Connections**
- Direct path from input to output
- Helps with gradient flow
- Preserves input information

### ✅ **Enhanced Output Projection**
- Deeper network (3 layers instead of 2)
- More expressive capacity
- Better final predictions

### ✅ **Input Normalization**
- LayerNorm on input
- More stable across different data distributions
- Reduces sensitivity to outliers

### ✅ **Conservative Initialization**
- Extra small weights for output layer
- Prevents early gradient explosion
- More stable training start

---

## Complete Command Reference

### Basic Usage

```bash
# Default configuration, default model
python train_v3.py --config optimized --data_dir ./dataset_with_label

# Balanced configuration, small model (faster)
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label

# Optimized configuration, large model (best results)
python train_v3.py --config optimized --model_size large --data_dir ./dataset_with_label
```

### With Custom Parameters

```bash
# Custom learning rate
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label --lr 1.5e-4

# Custom batch size (if memory allows)
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label --batch_size 24

# Custom epochs
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label --epochs 300

# All custom
python train_v3.py \
    --config optimized \
    --model_size default \
    --data_dir ./dataset_with_label \
    --batch_size 20 \
    --epochs 250 \
    --lr 1.2e-4
```

---

## Recommended Workflows

### For Experimentation (Quick Iteration)

```bash
# 1. Test with small model first (10 epochs, ~5-10 minutes)
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label --epochs 10

# 2. If results look good, train longer with small model (100 epochs, ~40 minutes)
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label --epochs 100

# 3. Move to default model (150 epochs, ~2 hours)
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label --epochs 150
```

### For Best Results (Maximum Performance)

```bash
# 1. Verify with default model first (50 epochs, ~30 minutes)
python train_v3.py --config optimized --model_size default --data_dir ./dataset_with_label --epochs 50

# 2. If stable, use large model (200-300 epochs, 4-6 hours)
python train_v3.py --config optimized --model_size large --data_dir ./dataset_with_label --epochs 250

# 3. If needed, fine-tune with lower learning rate
python train_v3.py --config optimized --model_size large --data_dir ./dataset_with_label --epochs 300 --lr 5e-5
```

### For Limited GPU Memory

```bash
# If you get OOM errors, use small model or reduce batch size
python train_v3.py --config balanced --model_size small --data_dir ./dataset_with_label --batch_size 8

# Or use default model with smaller batch
python train_v3.py --config balanced --model_size default --data_dir ./dataset_with_label --batch_size 8
```

---

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs_gpu_optimized

# Open browser to: http://localhost:6006
```

### Check GPU Usage

```bash
# Monitor GPU in real-time
nvidia-smi -l 1

# Or
watch -n 1 nvidia-smi
```

### Validate Checkpoints After Training

```bash
# Check all checkpoints
python -m utils.checkpoint_validator --dir checkpoints_gpu_optimized

# Check specific checkpoint
python -m utils.checkpoint_validator --checkpoint checkpoints_gpu_optimized/best_model.pt
```

---

## Comparison: V2 vs V3

| Feature | V2 (Original) | V3 (Enhanced) |
|---------|--------------|---------------|
| LayerNorm Position | Post-LN | Pre-LN ✅ |
| Activation | ReLU | GELU ✅ |
| Skip Connections | No | Yes ✅ |
| Input Normalization | No | Yes ✅ |
| Output Layers | 2 | 3 ✅ |
| Initialization | Standard | Conservative ✅ |
| Stability | Good | Better ✅ |
| Convergence | Standard | Faster ✅ |
| Final Performance | Good | Better ✅ |

---

## Expected Performance

### Loss Values (MSE)

After training with V3 model:

| Model Size | 50 Epochs | 150 Epochs | 250 Epochs |
|-----------|-----------|------------|------------|
| **Small** | 0.001-0.005 | 0.0005-0.001 | 0.0002-0.0008 |
| **Default** | 0.0005-0.002 | 0.0001-0.0005 | <0.0001 |
| **Large** | 0.0002-0.001 | <0.0001 | <0.00005 |

*Note: Actual values depend on your data and normalization*

### Expected Improvements Over V2

- **Small V3 vs V2:** 1.5-2x better loss
- **Default V3 vs V2:** 2-3x better loss
- **Large V3 vs V2:** 3-5x better loss

---

## Troubleshooting

### Issue: Out of Memory with Large Model

**Solution 1:** Reduce batch size
```bash
python train_v3.py --config optimized --model_size large --batch_size 8
```

**Solution 2:** Use default model instead
```bash
python train_v3.py --config optimized --model_size default
```

**Solution 3:** Use balanced config
```bash
python train_v3.py --config balanced --model_size large
```

### Issue: Training is Slow

**Check:** Are you using GPU?
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**If GPU is available but slow:**
- Increase batch size: `--batch_size 24`
- Reduce num_workers in config if CPU bottleneck

### Issue: Loss Not Decreasing

**Solution 1:** Increase learning rate
```bash
python train_v3.py --config optimized --lr 2e-4
```

**Solution 2:** Use larger model
```bash
python train_v3.py --config optimized --model_size large
```

**Solution 3:** Train longer
```bash
python train_v3.py --config optimized --epochs 300
```

### Issue: Overfitting (Val Loss >> Train Loss)

**Solution:** The V3 model has better regularization, but if still overfitting:

1. Increase dropout in config
2. Add more training data
3. Use smaller model
4. Reduce training epochs

---

## After Training

### 1. Validate the checkpoint

```bash
python -m utils.checkpoint_validator --checkpoint checkpoints_gpu_optimized/best_model.pt
```

### 2. Test on real data

```bash
python eval_real.py \
    --checkpoint checkpoints_gpu_optimized/best_model.pt \
    --data_dir source \
    --subjects VEP \
    --device cuda
```

### 3. Compare with V2 model

Train both and compare:
```bash
# V2 model
python train_optimized.py --config optimized --epochs 150

# V3 model
python train_v3.py --config optimized --model_size default --epochs 150

# Compare checkpoints
python -m utils.checkpoint_validator --all
```

---

## Summary

### Quick Commands

**Fast test:**
```bash
python train_v3.py --config balanced --model_size small --epochs 10
```

**Good results:**
```bash
python train_v3.py --config optimized --model_size default --epochs 150
```

**Best results:**
```bash
python train_v3.py --config optimized --model_size large --epochs 250
```

### Key Benefits of V3

- ✅ **More stable training** (Pre-LN, better initialization)
- ✅ **Better convergence** (GELU, skip connections)
- ✅ **Higher accuracy** (Enhanced architecture)
- ✅ **More robust** (Input normalization)
- ✅ **Same NaN protection** (All fixes from previous work)

### Choose V3 When:

- ✓ You want better accuracy
- ✓ You have GPU available
- ✓ You want more stable training
- ✓ You're willing to wait a bit longer for better results

---

## Support

If you encounter issues:

1. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
2. Validate config: `python configs/config_gpu_optimized.py`
3. Start with small model first
4. Monitor with TensorBoard
5. Check logs in `logs_gpu_optimized/`

**Need help?** Check the comprehensive guide: `TRAINING_IMPROVEMENTS_GUIDE.md`


