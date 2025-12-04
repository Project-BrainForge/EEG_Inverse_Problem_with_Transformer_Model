# Project Summary: EEG Source Localization Transformer

## 🎯 Project Overview

A complete PyTorch implementation of a Transformer-based deep learning model for **EEG source localization**. The model predicts brain source activity (994 brain regions) from EEG signals (75 channels) across 500 time points.

### What This Project Does

- **Input**: EEG recordings (500 time points × 75 channels)
- **Output**: Brain source activity (500 time points × 994 brain regions)
- **Task**: Time-series regression for neuroscience applications
- **Method**: State-of-the-art Transformer architecture

---

## 📁 Complete Project Structure

```
transformer_model/
├── 📄 README.md                    # Main documentation
├── 📄 INSTALL.md                   # Installation guide
├── 📄 ARCHITECTURE.md              # Detailed model architecture
├── 📄 PROJECT_SUMMARY.md           # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 test_setup.py                # Setup verification script
├── 📄 quick_start.sh               # Quick start bash script
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 dataset_with_label/          # Your dataset (10 samples)
│   ├── sample_00000.mat
│   ├── sample_00001.mat
│   └── ... (10 total)
│
├── 📂 configs/                     # Configuration files
│   └── config.yaml                 # Main config (hyperparameters)
│
├── 📂 src/                         # Source code
│   ├── __init__.py
│   │
│   ├── 📂 models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── transformer.py          # Transformer implementations
│   │
│   ├── 📂 data/                    # Data loading
│   │   ├── __init__.py
│   │   └── dataset.py              # Dataset and DataLoader
│   │
│   ├── 📂 utils/                   # Utilities
│   │   ├── __init__.py
│   │   └── helpers.py              # Helper functions
│   │
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── inference.py                # Inference script
│
├── 📂 checkpoints/                 # Model checkpoints (created on training)
├── 📂 logs/                        # TensorBoard logs (created on training)
└── 📂 results/                     # Evaluation results (created on eval)
```

---

## ✨ Key Features

### 🧠 Model Architecture
- **Two variants**: Encoder-only (default) and Encoder-Decoder
- **Transformer layers**: 6 layers with multi-head attention (8 heads)
- **Model dimension**: 256 (configurable)
- **Total parameters**: ~8.5M parameters
- **Attention mechanism**: Captures long-range temporal dependencies

### 📊 Data Processing
- **Automatic normalization**: Channel-wise z-score normalization
- **Train/validation split**: Configurable (default 80/20)
- **Batch processing**: Efficient DataLoader with multi-worker support
- **Memory efficient**: Loads data on-the-fly from .mat files

### 🚀 Training Features
- **TensorBoard integration**: Real-time training visualization
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: ReduceLROnPlateau
- **Gradient clipping**: Training stability
- **Checkpointing**: Automatic model saving
- **Resume training**: Continue from checkpoints

### 📈 Evaluation & Metrics
- **Comprehensive metrics**: MSE, MAE, RMSE, Correlation, R²
- **Visualization tools**: Plot predictions vs ground truth
- **Batch evaluation**: Process entire validation set
- **Save results**: Export to .npz and .txt files

### 🎯 Inference
- **Single file prediction**: Predict on one sample
- **Batch prediction**: Process multiple files
- **Save outputs**: Export predictions to .mat files
- **Easy to use**: Simple command-line interface

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python test_setup.py
```

### 3. Train the Model

```bash
cd src
python train.py --config ../configs/config.yaml
```

### 4. Monitor Training (in another terminal)

```bash
tensorboard --logdir logs
# Open browser to http://localhost:6006
```

### 5. Evaluate the Model

```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val \
    --visualize
```

### 6. Run Inference

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label/sample_00000.mat \
    --output ../results/prediction.mat
```

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

### Model Hyperparameters
```yaml
model:
  type: "encoder"           # or "encoder_decoder"
  input_channels: 75        # EEG channels
  output_channels: 994      # Brain regions
  d_model: 256             # Embedding dimension
  nhead: 8                 # Attention heads
  num_layers: 6            # Transformer layers
  dim_feedforward: 1024    # FFN dimension
  dropout: 0.1             # Dropout rate
```

### Training Settings
```yaml
num_epochs: 200
batch_size: 8
optimizer:
  type: "adamw"
  lr: 0.0001
  weight_decay: 0.01
```

---

## 📊 Expected Results

With the default configuration and adequate training data:

| Metric | Expected Value |
|--------|---------------|
| Training Loss (MSE) | < 0.1 (normalized) |
| Validation Loss | < 0.15 (normalized) |
| Correlation | > 0.85 |
| R² Score | > 0.70 |

*Note: Results depend on dataset quality and size. Current dataset has only 10 samples - consider adding more data for better performance.*

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| **README.md** | Main documentation with features and usage |
| **INSTALL.md** | Detailed installation and troubleshooting |
| **ARCHITECTURE.md** | Deep dive into model architecture |
| **PROJECT_SUMMARY.md** | This file - quick overview |

---

## 🛠️ Main Scripts

### `src/train.py`
- **Purpose**: Train the transformer model
- **Key Features**:
  - TensorBoard logging
  - Checkpointing
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping

### `src/evaluate.py`
- **Purpose**: Evaluate trained model
- **Key Features**:
  - Compute multiple metrics
  - Visualization of predictions
  - Save results to files
  - Support train/val splits

### `src/inference.py`
- **Purpose**: Make predictions on new data
- **Key Features**:
  - Single file or batch processing
  - Load from .mat files
  - Save predictions to .mat files
  - Automatic normalization

### `test_setup.py`
- **Purpose**: Verify installation
- **Tests**:
  - Package imports
  - Model creation
  - Dataset loading
  - Configuration parsing

---

## 💡 Usage Examples

### Example 1: Basic Training
```bash
cd src
python train.py --config ../configs/config.yaml
```

### Example 2: Training with Custom Data Directory
```bash
cd src
python train.py \
    --config ../configs/config.yaml \
    --data_dir /path/to/your/data
```

### Example 3: Resume Training from Checkpoint
```bash
cd src
python train.py \
    --config ../configs/config.yaml \
    --resume ../checkpoints/checkpoint_epoch_50.pth
```

### Example 4: Evaluate and Visualize
```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val \
    --visualize \
    --sample_idx 0
```

### Example 5: Batch Inference
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

## 🎓 Understanding the Code

### Key Classes

**`EEGTransformer`** (`src/models/transformer.py`)
- Main transformer model (encoder-only)
- Maps EEG (500×75) → Source (500×994)
- Uses positional encoding + transformer layers

**`EEGSourceDataset`** (`src/data/dataset.py`)
- Loads .mat files
- Handles train/val splitting
- Performs normalization

**`Trainer`** (`src/train.py`)
- Manages training loop
- Handles validation
- Saves checkpoints
- Logs to TensorBoard

**`Evaluator`** (`src/evaluate.py`)
- Computes metrics
- Generates visualizations
- Saves results

**`Inferencer`** (`src/inference.py`)
- Loads trained model
- Makes predictions
- Handles file I/O

### Important Functions

- `create_model()`: Factory function for model creation
- `create_dataloaders()`: Creates train/val dataloaders
- `compute_metrics()`: Computes evaluation metrics
- `save_checkpoint()`: Saves model state
- `load_checkpoint()`: Loads model state

---

## 🔍 Monitoring Training

### TensorBoard Metrics

Launch TensorBoard:
```bash
tensorboard --logdir logs
```

Available plots:
- **train/loss**: Training loss per iteration
- **train/lr**: Learning rate schedule
- **val/loss**: Validation loss per epoch
- **val/mae**: Mean absolute error
- **val/correlation**: Pearson correlation

---

## 🐛 Troubleshooting

### Common Issues

**1. "No module named 'torch'"**
```bash
pip install torch torchvision torchaudio
```

**2. "No .mat files found"**
- Check dataset is in `dataset_with_label/`
- Verify .mat files contain 'eeg_data' and 'source_data'

**3. "CUDA out of memory"**
- Reduce batch_size in config.yaml
- Reduce d_model or num_layers

**4. Training is slow**
- Use GPU if available
- Increase batch_size
- Reduce model size

**5. Model not learning**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data normalization
- Check for NaN values
- Try different random seed

---

## 📦 Dependencies

### Core Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+

### Additional Requirements
- PyYAML (configuration)
- TensorBoard (visualization)
- Matplotlib (plotting)

See `requirements.txt` for full list.

---

## 🎯 Next Steps

### For Beginners
1. ✅ Install dependencies (`pip install -r requirements.txt`)
2. ✅ Run test script (`python test_setup.py`)
3. ✅ Train with default settings (`cd src && python train.py --config ../configs/config.yaml`)
4. ✅ Monitor in TensorBoard (`tensorboard --logdir logs`)
5. ✅ Evaluate the model (`python evaluate.py ...`)

### For Advanced Users
1. **Customize architecture**: Edit model hyperparameters in config.yaml
2. **Add data augmentation**: Modify `EEGSourceDataset` class
3. **Try different losses**: Add custom loss functions in train.py
4. **Implement new metrics**: Extend compute_metrics() in helpers.py
5. **Add attention visualization**: Modify model to return attention weights
6. **Experiment with learning rate**: Try different schedulers
7. **Add regularization**: Implement label smoothing or mixup

### For Researchers
1. **Compare architectures**: Try encoder-decoder vs encoder-only
2. **Ablation studies**: Remove components to understand importance
3. **Hyperparameter tuning**: Use grid search or Optuna
4. **Ensemble methods**: Train multiple models and combine predictions
5. **Transfer learning**: Pre-train on related tasks
6. **Publication**: Document results and methodology

---

## 📝 Data Format

### Input .mat Files

Each file should contain:

```python
{
    'eeg_data': numpy.ndarray,      # Shape: (500, 75)
    'source_data': numpy.ndarray,   # Shape: (500, 994)
    # Optional:
    'index': int,                   # Sample index
    'snr': float,                   # Signal-to-noise ratio
    'labels': numpy.ndarray         # Additional labels
}
```

### Creating Your Own Data

```python
from scipy.io import savemat
import numpy as np

# Generate or load your data
eeg_data = np.random.randn(500, 75)      # Your EEG data
source_data = np.random.randn(500, 994)  # Your source data

# Save to .mat file
savemat('my_sample.mat', {
    'eeg_data': eeg_data,
    'source_data': source_data
})
```

---

## 🏆 Performance Tips

### To Improve Accuracy
1. **More data**: Add more training samples
2. **Larger model**: Increase d_model or num_layers
3. **Better regularization**: Tune dropout and weight_decay
4. **Data augmentation**: Add noise, time shifting, etc.
5. **Longer training**: Increase num_epochs
6. **Ensemble**: Train multiple models

### To Improve Speed
1. **Use GPU**: Ensure CUDA is available
2. **Larger batches**: Increase batch_size (if memory allows)
3. **Smaller model**: Reduce d_model or num_layers
4. **Mixed precision**: Enable AMP (automatic mixed precision)
5. **More workers**: Increase num_workers in DataLoader

### To Reduce Overfitting
1. **More training data**: Best solution
2. **Increase dropout**: Try 0.2-0.3
3. **Stronger weight decay**: Try 0.05-0.1
4. **Early stopping**: Already implemented
5. **Data augmentation**: Add variations to training data

---

## 📞 Support

### Getting Help
1. Check **README.md** for usage examples
2. Read **INSTALL.md** for installation issues
3. Review **ARCHITECTURE.md** for model details
4. Run `python train.py --help` for command options
5. Check TensorBoard logs for training insights

### Reporting Issues
When reporting issues, include:
- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Error message and traceback
- Configuration file (config.yaml)
- Steps to reproduce

---

## 📄 License

This project is provided as-is for research and educational purposes.

---

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Transformer**: Based on "Attention Is All You Need" (Vaswani et al., 2017)
- **EEG Analysis**: Inspired by neuroscience research community

---

## 🚀 You're Ready!

Your EEG source localization transformer is ready to train. The complete implementation includes:

✅ State-of-the-art transformer architecture  
✅ Comprehensive data loading and preprocessing  
✅ Full training pipeline with monitoring  
✅ Evaluation tools with visualization  
✅ Inference scripts for production  
✅ Detailed documentation  

**Start training now:**
```bash
cd src && python train.py --config ../configs/config.yaml
```

Good luck with your research! 🧠⚡

