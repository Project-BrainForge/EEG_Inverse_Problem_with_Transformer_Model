# Installation Guide

This guide will help you set up the environment and start training the EEG source localization transformer.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Step-by-Step Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd /Users/pasindusankalpa/Documents/dataset_deepSIF/transformer_model

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install PyTorch

Install PyTorch based on your system. Visit [pytorch.org](https://pytorch.org) for specific instructions.

#### For CPU-only (macOS/Linux):
```bash
pip install torch torchvision torchaudio
```

#### For CUDA 11.8 (Linux/Windows with NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1 (Linux/Windows with NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Run the test script to verify everything is set up correctly:

```bash
python test_setup.py
```

You should see output indicating all tests passed:
```
✓ All tests passed! Setup is complete.
```

## Quick Start

### Option 1: Use the Quick Start Script

```bash
# Make script executable (if not already)
chmod +x quick_start.sh

# Run the script
./quick_start.sh
```

### Option 2: Manual Setup

```bash
# 1. Activate virtual environment (if not already active)
source venv/bin/activate

# 2. Test the setup
python test_setup.py

# 3. Train the model
cd src
python train.py --config ../configs/config.yaml
```

## Training Your Model

### Start Training

```bash
cd src
python train.py --config ../configs/config.yaml
```

### Monitor Training with TensorBoard

Open a new terminal and run:

```bash
# Activate virtual environment
source venv/bin/activate

# Start TensorBoard
tensorboard --logdir logs
```

Then open your browser to: `http://localhost:6006`

### Training Output

During training, you'll see:
- Training loss per batch
- Validation metrics (loss, MAE, correlation)
- Learning rate updates
- Checkpoint saving information

Example output:
```
Epoch [1/200] Batch [0/8] Loss: 1.234567 Time: 0.123s
Epoch [1/200] Batch [10/8] Loss: 1.123456 Time: 0.115s

Validation - Epoch [1/200]
  Loss: 1.098765
  MAE: 0.876543
  Correlation: 0.8234

Checkpoint saved to checkpoints/checkpoint_epoch_1.pth
Best model saved to checkpoints/best_model.pth
```

## Evaluating the Model

After training, evaluate your model:

```bash
cd src
python evaluate.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --split val \
    --visualize \
    --sample_idx 0
```

This will:
- Compute evaluation metrics (MSE, MAE, correlation, R²)
- Save results to `results/evaluation_val.npz`
- Generate visualization plots (if `--visualize` is used)

## Running Inference

### Single File Prediction

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label/sample_00000.mat \
    --output ../results/prediction.mat
```

### Batch Prediction

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../checkpoints/best_model.pth \
    --input ../dataset_with_label \
    --output ../results/predictions \
    --batch
```

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: PyTorch is not installed. Follow step 2 above.

### Issue: "No module named 'yaml'"

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in `configs/config.yaml`:
```yaml
batch_size: 4  # or smaller
```

### Issue: Training is very slow

**Solutions**:
1. Use GPU if available
2. Check GPU is being used:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Reduce model size:
   ```yaml
   model:
     d_model: 128
     num_layers: 4
   ```

### Issue: "No .mat files found"

**Solution**: Make sure your dataset is in the correct location:
```
transformer_model/
└── dataset_with_label/
    ├── sample_00000.mat
    ├── sample_00001.mat
    └── ...
```

### Issue: Import errors

**Solution**: Make sure you're running scripts from the correct directory:
```bash
cd src  # Must be in src directory
python train.py --config ../configs/config.yaml
```

## Directory Structure After Installation

```
transformer_model/
├── venv/                    # Virtual environment
├── checkpoints/             # Model checkpoints (created during training)
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── logs/                    # TensorBoard logs (created during training)
├── results/                 # Evaluation results (created during evaluation)
├── configs/
│   └── config.yaml
├── dataset_with_label/      # Your dataset
├── src/
│   ├── models/
│   ├── data/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── requirements.txt
├── README.md
└── INSTALL.md
```

## Next Steps

1. **Customize Configuration**: Edit `configs/config.yaml` to adjust model architecture and training parameters
2. **Monitor Training**: Use TensorBoard to track training progress
3. **Experiment**: Try different model architectures and hyperparameters
4. **Evaluate**: Use the evaluation script to assess model performance
5. **Deploy**: Use the inference script for production predictions

## Getting Help

- Check the README.md for detailed documentation
- Review the configuration in `configs/config.yaml`
- Run `python train.py --help` for command-line options
- Check TensorBoard logs for training insights

## Additional Resources

- PyTorch documentation: https://pytorch.org/docs/
- Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
- EEG processing: MNE-Python documentation

---

Happy training! 🚀

