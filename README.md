# EEG Source Localization Transformer

A PyTorch implementation of a Transformer-based model for EEG source localization. This model predicts brain source activity (994 regions) from EEG signals (75 channels) at 500 time points.

## Overview

This project uses transformer architecture to map EEG signals to brain source activity, which is a crucial task in neuroscience for understanding brain dynamics and localization of neural activity.

### Problem Statement

- **Input**: EEG data with shape `(500, 75)` - 500 time points with 75 channels
- **Output**: Source data with shape `(500, 994)` - corresponding brain activity of 994 brain regions

## Features

- 🧠 State-of-the-art Transformer architecture for time-series prediction
- 📊 Two model variants: Encoder-only and Encoder-Decoder
- 🔄 Automatic data normalization and preprocessing
- 📈 TensorBoard integration for training visualization
- ⚡ Early stopping and learning rate scheduling
- 💾 Model checkpointing and resumable training
- 📉 Comprehensive evaluation metrics (MSE, MAE, Correlation, R²)
- 🎨 Visualization tools for predictions

## Project Structure

```
transformer_model/
├── configs/
│   └── config.yaml              # Configuration file
├── dataset_with_label/          # Dataset directory (.mat files)
│   ├── sample_00000.mat
│   ├── sample_00001.mat
│   └── ...
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py       # Transformer model architectures
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Dataset and dataloader
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py          # Helper functions
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Inference script
├── checkpoints/                 # Model checkpoints
├── logs/                        # TensorBoard logs
├── results/                     # Evaluation results
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Verify installation:

```bash
python src/models/transformer.py  # Test model
python src/data/dataset.py        # Test dataset
```

## Usage

### 1. Training

Train the model using the default configuration:

```bash
cd src
python train.py --config ../configs/config.yaml
```

With custom data directory:

```bash
python train.py --config ../configs/config.yaml --data_dir ../dataset_with_label
```

Resume training from checkpoint:

```bash
python train.py --config ../configs/config.yaml --resume ../checkpoints/best_model.pth
```

### 2. Monitoring Training

Launch TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006`

### 3. Evaluation

Evaluate the trained model on validation set:

```bash
cd src
python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --split val
```

Evaluate on training set:

```bash
python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --split train
```

With visualization:

```bash
python evaluate.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --visualize --sample_idx 0
```

### 4. Inference

Predict on a single file:

```bash
cd src
python inference.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --input ../dataset_with_label/sample_00000.mat --output ../results/prediction.mat
```

Batch prediction on multiple files:

```bash
python inference.py --config ../configs/config.yaml --checkpoint ../checkpoints/best_model.pth --input ../dataset_with_label --output ../results/predictions --batch
```

## Configuration

Edit `configs/config.yaml` to customize:

### Model Parameters

- `d_model`: Transformer embedding dimension (default: 256)
- `nhead`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 6)
- `dim_feedforward`: Feedforward network dimension (default: 1024)
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters

- `num_epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 8)
- `lr`: Learning rate (default: 0.0001)
- `weight_decay`: Weight decay for regularization (default: 0.01)
- `early_stopping_patience`: Patience for early stopping (default: 30)

## Model Architecture

### Encoder-Only Transformer (Default)

```
Input EEG (500, 75)
    ↓
Linear Projection (75 → d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder (6 layers)
    ↓
Output Projection (d_model → 994)
    ↓
Predicted Source (500, 994)
```

### Key Components

1. **Input Projection**: Projects EEG channels to model dimension
2. **Positional Encoding**: Adds temporal position information
3. **Transformer Encoder**: Multi-head self-attention layers
4. **Output Projection**: Maps to brain regions

## Data Format

### Input .mat Files

Each `.mat` file should contain:

- `eeg_data`: NumPy array of shape `(500, 75)` - EEG signals
- `source_data`: NumPy array of shape `(500, 994)` - Brain source activity (ground truth)

### Example Loading

```python
from scipy.io import loadmat

data = loadmat('sample_00000.mat')
eeg = data['eeg_data']      # Shape: (500, 75)
source = data['source_data']  # Shape: (500, 994)
```

## Performance Metrics

The model is evaluated using:

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Correlation**: Pearson correlation coefficient
- **R²**: Coefficient of determination
- **Relative Error**: Mean relative error

## Tips for Better Performance

1. **Increase model capacity**: Increase `d_model` or `num_layers`
2. **More data**: The model benefits from more training samples
3. **Learning rate tuning**: Adjust `lr` and `scheduler` parameters
4. **Regularization**: Tune `dropout` and `weight_decay`
5. **Batch size**: Larger batches can improve stability (if memory allows)
6. **Gradient clipping**: Helps with training stability (default: 1.0)

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Reduce `d_model` or `num_layers`
- Enable gradient checkpointing (requires code modification)

### Training is too slow

- Increase `batch_size`
- Reduce `num_layers` or `dim_feedforward`
- Use GPU if available

### Model not converging

- Reduce learning rate
- Check data normalization
- Increase `grad_clip` value
- Try different optimizer (`adam` vs `adamw`)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eeg_transformer,
  title={EEG Source Localization Transformer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/eeg-transformer}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the deep learning framework
- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- EEG processing inspired by MNE-Python

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
