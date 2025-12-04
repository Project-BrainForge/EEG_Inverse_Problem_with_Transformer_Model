# Workflow Guide: From Installation to Deployment

## 📋 Table of Contents
1. [Installation Workflow](#installation-workflow)
2. [Training Workflow](#training-workflow)
3. [Evaluation Workflow](#evaluation-workflow)
4. [Inference Workflow](#inference-workflow)
5. [Customization Workflow](#customization-workflow)

---

## 🔧 Installation Workflow

```
START
  │
  ├─→ Step 1: Create Virtual Environment
  │   $ python3 -m venv venv
  │   $ source venv/bin/activate
  │
  ├─→ Step 2: Install PyTorch
  │   $ pip install torch torchvision torchaudio
  │
  ├─→ Step 3: Install Dependencies
  │   $ pip install -r requirements.txt
  │
  ├─→ Step 4: Verify Installation
  │   $ python test_setup.py
  │
  └─→ ✅ Ready to Train!
```

### Quick Installation (Automated)
```bash
./quick_start.sh
```

---

## 🚀 Training Workflow

### Standard Training Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. PREPARE DATA                                         │
│    • Place .mat files in dataset_with_label/           │
│    • Ensure format: eeg_data (500,75)                  │
│                    source_data (500,994)               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. CONFIGURE MODEL                                      │
│    • Edit configs/config.yaml                          │
│    • Set hyperparameters (d_model, layers, etc.)      │
│    • Set training params (lr, batch_size, epochs)     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. START TRAINING                                       │
│    $ cd src                                            │
│    $ python train.py --config ../configs/config.yaml  │
│                                                        │
│    What happens:                                       │
│    ├─ Load dataset → split train/val                  │
│    ├─ Create model → initialize weights               │
│    ├─ Setup optimizer & scheduler                     │
│    └─ Start training loop                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. MONITOR TRAINING (parallel terminal)                │
│    $ tensorboard --logdir logs                         │
│    → Open http://localhost:6006                        │
│                                                        │
│    Watch:                                              │
│    ├─ Training loss (should decrease)                 │
│    ├─ Validation loss (should decrease)               │
│    ├─ Correlation (should increase)                   │
│    └─ Learning rate (should adjust automatically)     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. TRAINING COMPLETES                                   │
│    Outputs:                                            │
│    ├─ checkpoints/best_model.pth (best val loss)      │
│    ├─ checkpoints/checkpoint_epoch_*.pth (each epoch) │
│    └─ logs/ (TensorBoard logs)                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
              ✅ Model Trained!
```

### Training Commands

**Basic training:**
```bash
cd src
python train.py --config ../configs/config.yaml
```

**With custom data directory:**
```bash
python train.py --config ../configs/config.yaml --data_dir /path/to/data
```

**Resume from checkpoint:**
```bash
python train.py --config ../configs/config.yaml --resume ../checkpoints/checkpoint_epoch_50.pth
```

### What to Expect During Training

```
Epoch [1/200] Batch [0/8] Loss: 1.234567 Time: 0.123s
Epoch [1/200] Batch [10/8] Loss: 0.987654 Time: 0.115s

Validation - Epoch [1/200]
  Loss: 0.876543
  MAE: 0.765432
  Correlation: 0.7234

Checkpoint saved to checkpoints/checkpoint_epoch_1.pth
Best model saved to checkpoints/best_model.pth

Epoch [2/200] Batch [0/8] Loss: 0.856432 Time: 0.112s
...
```

---

## 📊 Evaluation Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. SELECT CHECKPOINT                                    │
│    • Usually: checkpoints/best_model.pth               │
│    • Or specific epoch: checkpoint_epoch_N.pth         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. RUN EVALUATION                                       │
│    $ cd src                                            │
│    $ python evaluate.py \                              │
│        --config ../configs/config.yaml \               │
│        --checkpoint ../checkpoints/best_model.pth \    │
│        --split val                                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. MODEL EVALUATION                                     │
│    Process:                                            │
│    ├─ Load trained model                              │
│    ├─ Load validation data                            │
│    ├─ Run inference on all samples                    │
│    ├─ Compute metrics (MSE, MAE, Corr, R²)           │
│    └─ Generate visualizations                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. RESULTS GENERATED                                    │
│    Outputs:                                            │
│    ├─ results/evaluation_val.npz (all predictions)    │
│    ├─ results/metrics_val.txt (metrics text)          │
│    └─ results/visualization_*.png (if --visualize)    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. ANALYZE RESULTS                                      │
│    Check:                                              │
│    ├─ Correlation > 0.85? ✅ Good                     │
│    ├─ R² > 0.70? ✅ Good                              │
│    ├─ MAE acceptable? (domain-specific)               │
│    └─ Visual inspection: predictions match targets?   │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
         Satisfied with results?
                 │
        ┌────────┴────────┐
        │                 │
      YES                NO
        │                 │
        ▼                 ▼
  Deploy Model    Improve Model
                  (see Customization)
```

### Evaluation Commands

**Basic evaluation:**
```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val
```

**With visualization:**
```bash
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val \
    --visualize \
    --sample_idx 0
```

**Evaluate on training set:**
```bash
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split train
```

### Expected Output

```
============================================================
Evaluating on val set (2 samples)
============================================================

Processed 1/1 batches

============================================================
Evaluation Results on val set:
============================================================
  MSE:             0.123456
  MAE:             0.234567
  RMSE:            0.345678
  Correlation:     0.8765
  R²:              0.7654
  Relative Error:  0.1234
============================================================

Results saved to results/evaluation_val.npz
Metrics saved to results/metrics_val.txt
```

---

## 🎯 Inference Workflow

### Single File Inference

```
┌─────────────────────────────────────────────────────────┐
│ 1. PREPARE INPUT                                        │
│    • Single .mat file with 'eeg_data' (500, 75)        │
│    • Example: sample_00000.mat                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. RUN INFERENCE                                        │
│    $ cd src                                            │
│    $ python inference.py \                             │
│        --config ../configs/config.yaml \               │
│        --checkpoint ../checkpoints/best_model.pth \    │
│        --input ../dataset_with_label/sample_00000.mat \│
│        --output ../results/prediction.mat              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. PREDICTION GENERATED                                 │
│    Output file: prediction.mat                         │
│    Contains:                                           │
│    ├─ source_data_predicted (500, 994)                │
│    ├─ eeg_data (original input)                       │
│    └─ source_data_true (if available)                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
              ✅ Done!
```

### Batch Inference

```
┌─────────────────────────────────────────────────────────┐
│ 1. PREPARE DIRECTORY                                    │
│    • Multiple .mat files in directory                  │
│    • Each with 'eeg_data' field                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. RUN BATCH INFERENCE                                  │
│    $ cd src                                            │
│    $ python inference.py \                             │
│        --config ../configs/config.yaml \               │
│        --checkpoint ../checkpoints/best_model.pth \    │
│        --input ../dataset_with_label \                 │
│        --output ../results/predictions \               │
│        --batch                                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. BATCH PROCESSING                                     │
│    For each file:                                      │
│    ├─ Load EEG data                                    │
│    ├─ Normalize using training stats                  │
│    ├─ Run model inference                             │
│    ├─ Denormalize predictions                         │
│    └─ Save to output directory                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. ALL PREDICTIONS SAVED                                │
│    results/predictions/                                │
│    ├─ predicted_sample_00000.mat                      │
│    ├─ predicted_sample_00001.mat                      │
│    └─ ...                                             │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
              ✅ Batch Complete!
```

### Inference Commands

**Single file:**
```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label/sample_00000.mat \
    --output ../results/prediction.mat
```

**Batch processing:**
```bash
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label \
    --output ../results/predictions \
    --batch
```

---

## 🔨 Customization Workflow

### Improving Model Performance

```
┌─────────────────────────────────────────────────────────┐
│ Model not performing well?                             │
│ Follow this decision tree:                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
         Is training loss high?
                 │
        ┌────────┴────────┐
        │                 │
      YES                NO
        │                 │
        │                 └─→ Model underfitting
        │                     ├─ Increase model size
        │                     │   (d_model, num_layers)
        │                     ├─ Train longer
        │                     └─ Lower learning rate
        │
        ▼
  Is validation loss >> training loss?
        │
        ├─ YES → Model overfitting
        │         ├─ Add more data
        │         ├─ Increase dropout
        │         ├─ Increase weight_decay
        │         └─ Reduce model size
        │
        └─ NO → Need better architecture
                  ├─ Try encoder-decoder
                  ├─ Adjust num_layers
                  └─ Tune hyperparameters
```

### Hyperparameter Tuning Guide

```
┌─────────────────────────────────────────────────────────┐
│ 1. BASELINE EXPERIMENT                                  │
│    • Run with default config                           │
│    • Record results                                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. TUNE LEARNING RATE                                   │
│    Try: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]                │
│    Pick the one with best val loss                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. TUNE MODEL SIZE                                      │
│    d_model: [128, 256, 512]                           │
│    num_layers: [4, 6, 8]                              │
│    Balance performance vs. speed                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. TUNE REGULARIZATION                                  │
│    dropout: [0.0, 0.1, 0.2, 0.3]                      │
│    weight_decay: [0.0, 0.01, 0.05, 0.1]               │
│    Find sweet spot                                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. FINE-TUNE OTHER PARAMS                               │
│    • batch_size                                        │
│    • num_heads                                         │
│    • dim_feedforward                                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
              ✅ Optimized!
```

### Configuration Examples

**Small/Fast Model:**
```yaml
model:
  d_model: 128
  num_layers: 4
  dim_feedforward: 512
  nhead: 4
batch_size: 16
```

**Large/Accurate Model:**
```yaml
model:
  d_model: 512
  num_layers: 8
  dim_feedforward: 2048
  nhead: 8
batch_size: 4
```

**Regularized Model (prevent overfitting):**
```yaml
model:
  dropout: 0.3
optimizer:
  weight_decay: 0.1
```

---

## 📈 Complete ML Pipeline

```
DATA PREPARATION
     ↓
DATA LOADING & PREPROCESSING
     ↓
MODEL ARCHITECTURE DESIGN
     ↓
TRAINING
     ↓
MONITORING (TensorBoard)
     ↓
EVALUATION
     ↓
┌────────────────┐
│ Good Results?  │
└────┬───────────┘
     │
  NO │ YES
     │  │
     │  └→ DEPLOYMENT
     │      ↓
     │   INFERENCE
     │      ↓
     │   PRODUCTION
     │
     └→ HYPERPARAMETER TUNING
         ↓
      RE-TRAINING
         ↓
      (back to TRAINING)
```

---

## 🚀 Production Deployment Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. EXPORT BEST MODEL                                    │
│    • Identify best checkpoint                          │
│    • Copy to deployment directory                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. OPTIMIZE MODEL (optional)                            │
│    • TorchScript compilation                           │
│    • ONNX export                                       │
│    • Quantization                                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. CREATE INFERENCE SERVICE                             │
│    • Flask/FastAPI REST API                            │
│    • gRPC service                                      │
│    • Batch processing script                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. DEPLOY                                               │
│    • Docker container                                  │
│    • Cloud service (AWS, GCP, Azure)                  │
│    • On-premise server                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
              ✅ In Production!
```

---

## 💡 Quick Reference Commands

### Installation
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch && pip install -r requirements.txt
python test_setup.py
```

### Training
```bash
cd src && python train.py --config ../configs/config.yaml
```

### Monitoring
```bash
tensorboard --logdir logs
```

### Evaluation
```bash
cd src && python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --split val
```

### Inference
```bash
cd src && python inference.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --input ../dataset_with_label/sample_00000.mat --output ../results/prediction.mat
```

---

## 📊 Success Checklist

### Before Training
- [ ] Dependencies installed
- [ ] test_setup.py passes
- [ ] Dataset in correct location
- [ ] Config file customized

### During Training
- [ ] TensorBoard running
- [ ] Training loss decreasing
- [ ] Validation loss decreasing
- [ ] No NaN values

### After Training
- [ ] Checkpoints saved
- [ ] Best model identified
- [ ] Evaluation metrics computed
- [ ] Results visualized

### Deployment Ready
- [ ] Model performance acceptable
- [ ] Inference script tested
- [ ] Documentation complete
- [ ] Production plan ready

---

You're all set! Follow these workflows to go from installation to deployment. 🚀

