# ⚡ Quick Start Guide

## 🎯 What You Have

A complete **Transformer-based EEG source localization system** that:

- Maps EEG signals (500×75) → Brain source activity (500×994)
- Uses state-of-the-art Transformer architecture
- Includes training, evaluation, and inference scripts
- Has comprehensive documentation

---

## 🚀 Get Started in 3 Steps

### Step 1: Install (2 minutes)

```bash
# Navigate to project
cd /Users/pasindusankalpa/Documents/dataset_deepSIF/transformer_model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (choose based on your system)
# For CPU (macOS/Linux):
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### Step 2: Train (depends on data size)

```bash
cd src
python train.py --config ../configs/config.yaml
```

### Step 3: Evaluate

```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val \
    --visualize
```

---

## 📊 Monitor Training

Open a new terminal:

```bash
cd /Users/pasindusankalpa/Documents/dataset_deepSIF/transformer_model
source venv/bin/activate
tensorboard --logdir logs
```

Then open: http://localhost:6006

---

## 📁 What's Where

```
transformer_model/
├── 📖 README.md              ← Start here
├── 📖 QUICK_START.md         ← This file
├── 📖 INSTALL.md             ← Detailed installation
├── 📖 PROJECT_SUMMARY.md     ← Complete overview
├── 📖 ARCHITECTURE.md        ← Model architecture
├── 📖 WORKFLOW.md            ← Step-by-step workflows
│
├── ⚙️  configs/config.yaml    ← Configure everything here
├── 📦 requirements.txt       ← Dependencies
├── 🧪 test_setup.py          ← Verify installation
│
├── 📂 src/                   ← Source code
│   ├── models/transformer.py  (Model architecture)
│   ├── data/dataset.py        (Data loading)
│   ├── utils/helpers.py       (Utilities)
│   ├── train.py               (Training script)
│   ├── evaluate.py            (Evaluation script)
│   └── inference.py           (Inference script)
│
├── 📂 dataset_with_label/    ← Your data (10 samples)
├── 📂 checkpoints/           ← Saved models (after training)
├── 📂 logs/                  ← TensorBoard logs (after training)
└── 📂 results/               ← Evaluation results (after eval)
```

---

## ⚙️ Configuration

Edit `configs/config.yaml`:

```yaml
# Quick tweaks for common scenarios:

# 🎯 For better accuracy (slower training):
model:
  d_model: 512
  num_layers: 8

# ⚡ For faster training (may reduce accuracy):
model:
  d_model: 128
  num_layers: 4
batch_size: 16

# 🛡️ If overfitting (val loss > train loss):
model:
  dropout: 0.3
optimizer:
  weight_decay: 0.1

# 🔧 If underfitting (both losses high):
optimizer:
  lr: 0.0005  # Increase learning rate
num_epochs: 300  # Train longer
```

---

## 🎯 Common Tasks

### Train the model

```bash
cd src
python train.py --config ../configs/config.yaml
```

### Resume training from checkpoint

```bash
cd src
python train.py \
    --config ../configs/config.yaml \
    --resume ../checkpoints/checkpoint_epoch_50.pth
```

### Evaluate the model

```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val
```

### Make predictions (single file)

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label/sample_00000.mat \
    --output ../results/prediction.mat
```

### Make predictions (batch)

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label \
    --output ../results/predictions \
    --batch
```

---

## 🐛 Troubleshooting

### "No module named 'torch'"

```bash
pip install torch torchvision torchaudio
```

### "CUDA out of memory"

Reduce batch size in `configs/config.yaml`:

```yaml
batch_size: 4 # or smaller
```

### Training is slow

- Use GPU if available
- Increase batch_size (if memory allows)
- Reduce model size (d_model, num_layers)

### Model not learning

- Check learning rate (try 1e-4 to 1e-3)
- Train longer (increase num_epochs)
- Check TensorBoard for issues
- Verify data is normalized

### Import errors

Make sure you're in the right directory:

```bash
cd src  # Must be in src directory
python train.py --config ../configs/config.yaml
```

---

## 📚 Documentation Overview

| Document               | Purpose                | When to Read          |
| ---------------------- | ---------------------- | --------------------- |
| **QUICK_START.md**     | Get started fast       | First                 |
| **README.md**          | Main documentation     | After installation    |
| **INSTALL.md**         | Installation help      | If setup issues       |
| **PROJECT_SUMMARY.md** | Complete overview      | To understand project |
| **ARCHITECTURE.md**    | Model details          | To understand model   |
| **WORKFLOW.md**        | Step-by-step workflows | During development    |

---

## 🎓 Understanding Your Model

### Input

- **EEG data**: 500 time points × 75 channels
- **Format**: .mat file with 'eeg_data' field
- **Shape**: (500, 75)

### Output

- **Source activity**: 500 time points × 994 brain regions
- **Format**: .mat file with 'source_data' field
- **Shape**: (500, 994)

### Model

- **Architecture**: Transformer encoder
- **Parameters**: ~8.5M (default config)
- **Training**: Supervised learning with MSE loss
- **Metrics**: MSE, MAE, Correlation, R²

---

## 📊 Expected Performance

With default settings and adequate training data:

| Metric          | Target Value |
| --------------- | ------------ |
| Training Loss   | < 0.1        |
| Validation Loss | < 0.15       |
| Correlation     | > 0.85       |
| R² Score        | > 0.70       |

**Note**: You have 10 samples. For better results, add more data!

---

## 🔍 What Happens During Training

```
1. Load dataset → Split train/val (8/2 with 10 samples)
2. Create model → Initialize ~8.5M parameters
3. For each epoch:
   ├─ Forward pass: EEG → Model → Predictions
   ├─ Compute loss: MSE(predictions, ground_truth)
   ├─ Backward pass: Compute gradients
   ├─ Update weights: Optimizer step
   └─ Validate: Check performance on validation set
4. Save best model based on validation loss
5. Early stopping if no improvement for 30 epochs
```

---

## ⚡ Performance Tips

### To improve accuracy:

- Add more training data (most important!)
- Increase model size (d_model, num_layers)
- Train longer (more epochs)
- Tune hyperparameters

### To speed up training:

- Use GPU (if available)
- Increase batch size
- Reduce model size
- Use fewer epochs

### To reduce overfitting:

- Add more data
- Increase dropout
- Increase weight_decay
- Use data augmentation

---

## 🎯 Next Steps

1. ✅ **Install dependencies**

   ```bash
   pip install -r requirements.txt
   python test_setup.py
   ```

2. ✅ **Start training**

   ```bash
   cd src
   python train.py --config ../configs/config.yaml
   ```

3. ✅ **Monitor progress**

   ```bash
   tensorboard --logdir logs
   ```

4. ✅ **Evaluate results**

   ```bash
   python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth
   ```

5. ✅ **Make predictions**
   ```bash
   python inference.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --input ../dataset_with_label/sample_00000.mat
   ```

---

## 💡 Tips for Success

1. **Start small**: Use default config first
2. **Monitor training**: Always use TensorBoard
3. **Check validation**: Watch for overfitting
4. **Save checkpoints**: Don't lose trained models
5. **Document changes**: Keep track of experiments
6. **Add more data**: More data = better results!

---

## 🆘 Need Help?

1. **Installation issues** → Read `INSTALL.md`
2. **Understanding code** → Read `ARCHITECTURE.md`
3. **Step-by-step guide** → Read `WORKFLOW.md`
4. **Complete overview** → Read `PROJECT_SUMMARY.md`
5. **General usage** → Read `README.md`

---

## ✨ You're Ready!

Your transformer model is ready to train. The complete implementation is production-ready with:

✅ Professional code structure  
✅ Comprehensive documentation  
✅ Training pipeline  
✅ Evaluation tools  
✅ Inference scripts  
✅ Configuration management  
✅ Monitoring with TensorBoard

**Start training now:**

```bash
cd src && python train.py --config ../configs/config.yaml
```

Good luck! 🚀🧠
