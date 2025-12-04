# Setup Guide - EEG Source Localization Transformer

This guide will help you set up and run the EEG Source Localization Transformer model.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Hardware**: 
  - Minimum: 8GB RAM, CPU
  - Recommended: 16GB RAM, NVIDIA GPU with 4GB+ VRAM

## 🚀 Installation Steps

### Step 1: Verify Python Installation

```bash
python3 --version
```

Expected output: Python 3.8.x or higher

### Step 2: Install PyTorch

**For CPU-only (macOS/Linux/Windows):**
```bash
pip install torch torchvision torchaudio
```

**For GPU with CUDA 11.8 (Linux/Windows):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU with CUDA 12.1 (Linux/Windows):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For macOS with Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision torchaudio
```
Note: PyTorch will automatically use Metal Performance Shaders (MPS) for acceleration.

### Step 3: Install Other Dependencies

```bash
cd /Users/pasindusankalpa/Documents/dataset_deepSIF/transformer_model
pip install -r requirements.txt
```

This will install:
- numpy
- scipy
- tensorboard
- tqdm
- matplotlib
- h5py

### Step 4: Verify Installation

Run the test script to verify everything is set up correctly:

```bash
python test_setup.py
```

Expected output: All tests should pass with ✓ marks.

## 📊 Dataset Verification

Your dataset is already in place:
- Location: `dataset_with_label/`
- Files: `sample_00000.mat` to `sample_00009.mat` (10 samples)
- Each file contains:
  - `eeg_data`: (500, 75) - EEG signals
  - `source_data`: (500, 994) - Brain source activity

## 🎯 Quick Start

### Option 1: Using the Quick Start Script

```bash
bash quick_start.sh
```

This will:
1. Check Python installation
2. Install dependencies
3. Create necessary directories
4. Verify dataset

### Option 2: Manual Setup

```bash
# Create directories
mkdir -p checkpoints logs data results

# Verify setup
python test_setup.py
```

## 🏃 Running the Model

### 1. Training

**Basic training (recommended for first run):**
```bash
python train.py
```

**Training with custom parameters:**
```bash
python train.py --batch_size 4 --epochs 50 --lr 0.0001
```

**For quick testing (2 epochs):**
```bash
python train.py --epochs 2
```

### 2. Monitoring Training

Open a new terminal and run:
```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser to see:
- Training/validation loss curves
- Learning rate schedule
- Real-time metrics

### 3. Evaluation

After training completes, evaluate the model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize --save_predictions
```

This will:
- Evaluate on test set
- Generate visualizations
- Save predictions to `results/`
- Display metrics (MSE, MAE, Correlation)

### 4. Single Sample Inference

To predict source data for a single EEG sample:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --input dataset_with_label/sample_00000.mat --output my_prediction.mat
```

## 🔧 Configuration

Edit `configs/config.py` to customize:

### Common Adjustments:

**For faster training (less accurate):**
```python
BATCH_SIZE = 4
D_MODEL = 128
NUM_LAYERS = 4
NUM_EPOCHS = 30
```

**For better accuracy (slower):**
```python
BATCH_SIZE = 16
D_MODEL = 512
NUM_LAYERS = 12
NUM_EPOCHS = 200
```

**For limited memory:**
```python
BATCH_SIZE = 2
USE_AMP = True  # Enable mixed precision
```

## 📁 Project Structure After Setup

```
transformer_model/
├── dataset_with_label/          # Your data (10 .mat files)
├── models/                      # Model architectures
├── utils/                       # Dataset utilities
├── configs/                     # Configuration
├── checkpoints/                 # Saved models (created during training)
├── logs/                        # TensorBoard logs (created during training)
├── results/                     # Evaluation results (created during evaluation)
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Single sample inference
├── test_setup.py               # Setup verification
├── requirements.txt             # Dependencies
├── README.md                    # Main documentation
└── SETUP_GUIDE.md              # This file
```

## 🐛 Troubleshooting

### Issue 1: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue 2: "CUDA out of memory"
**Solutions:**
1. Reduce batch size:
   ```bash
   python train.py --batch_size 2
   ```
2. Use CPU instead:
   Edit `configs/config.py` and set:
   ```python
   DEVICE = torch.device("cpu")
   ```

### Issue 3: "No .mat files found"
**Solution:**
Ensure your dataset is in `dataset_with_label/` directory:
```bash
ls dataset_with_label/*.mat
```

### Issue 4: Training is very slow
**Solutions:**
1. Enable mixed precision (if not already):
   ```python
   USE_AMP = True  # in configs/config.py
   ```
2. Reduce model size:
   ```python
   D_MODEL = 128
   NUM_LAYERS = 4
   ```
3. Use GPU if available

### Issue 5: "RuntimeError: Expected all tensors to be on the same device"
**Solution:**
This is usually handled automatically. If it occurs, ensure consistent device usage in `configs/config.py`.

## 📈 Expected Training Time

With your 10 samples:

| Hardware | Batch Size | Time per Epoch | Total (100 epochs) |
|----------|------------|----------------|-------------------|
| CPU (Intel i7) | 8 | ~2-3 min | ~3-5 hours |
| CPU (Apple M1/M2) | 8 | ~1-2 min | ~2-3 hours |
| GPU (RTX 3060) | 8 | ~20-30 sec | ~30-50 min |
| GPU (RTX 4090) | 8 | ~10-15 sec | ~15-25 min |

**Note:** With only 10 samples, the model may overfit. For production use, 1000+ samples are recommended.

## 🎓 Next Steps

1. **Run test setup:**
   ```bash
   python test_setup.py
   ```

2. **Start training:**
   ```bash
   python train.py --epochs 10
   ```

3. **Monitor training:**
   ```bash
   tensorboard --logdir logs
   ```

4. **Evaluate results:**
   ```bash
   python evaluate.py --checkpoint checkpoints/best_model.pt --visualize
   ```

## 📞 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify installation: `python test_setup.py`
3. Check TensorBoard logs: `tensorboard --logdir logs`
4. Review configuration: `configs/config.py`

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ `test_setup.py` passes all tests
2. ✅ Training starts without errors
3. ✅ Loss decreases over epochs
4. ✅ TensorBoard shows training curves
5. ✅ Evaluation produces metrics and visualizations

Good luck with your training! 🚀

