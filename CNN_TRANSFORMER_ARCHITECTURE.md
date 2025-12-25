# CNN-Transformer Hybrid Architecture

## Overview

This project now supports **two model architectures** for EEG source localization:

1. **Standard Linear Transformer** - Original architecture with linear input projection
2. **CNN-Transformer Hybrid** - New modular architecture with separated CNN spatial encoder and Transformer temporal encoder

## Architecture Comparison

### 1. Standard Linear Transformer (`EEGTransformer`)

```
Input (batch, 500, 75)
    ↓
Linear Projection (75 → 256)
    ↓
Positional Encoding
    ↓
Transformer Encoder (6 layers)
    ↓
Output Projection (256 → 994)
    ↓
Output (batch, 500, 994)
```

**Pros:**
- Simple and fast
- Fewer parameters
- No external dependencies

**Cons:**
- Doesn't capture spatial electrode topology
- Treats channels as arbitrary features

### 2. CNN-Transformer Hybrid (`CNNTransformerHybrid`)

```
Input (batch, 500, 75)
    ↓
Topological Converter (75 channels → 64×64 spatial map)
    ↓
CNN Spatial Encoder (3 conv layers: 32→64→128)
    ↓
Features (batch, 500, 256)
    ↓
Transformer Encoder (6 layers)
    ↓
Output Projection (256 → 994)
    ↓
Output (batch, 500, 994)
```

**Pros:**
- Preserves true scalp spatial topology
- CNN extracts hierarchical spatial features
- Modular design (swap CNN/Transformer independently)
- Can extract intermediate features for analysis

**Cons:**
- More parameters (+2.9%)
- Slower due to topological conversion (~240ms overhead per batch)
- Requires MNE library and electrode configuration

## Usage

### Option 1: Standard Transformer

```python
from src.models import EEGTransformer

model = EEGTransformer(
    input_channels=75,
    output_channels=994,
    d_model=256,
    nhead=8,
    num_layers=6,
    use_cnn_encoder=False  # Standard linear projection
)
```

### Option 2: CNN-Transformer Hybrid

```python
from src.models import CNNTransformerHybrid

model = CNNTransformerHybrid(
    eeg_channels=75,
    output_channels=994,
    d_model=256,
    nhead=8,
    num_layers=6,
    topo_image_size=64,
    electrode_file='anatomy/electrode_75.mat',
    cnn_channels=[32, 64, 128],
    cnn_type='standard'  # or 'light' for fewer parameters
)
```

### Option 3: Using Config

```python
from src.models import create_hybrid_model
from configs.config import Config

# Set in configs/config.py:
# USE_CNN_ENCODER = True

model = create_hybrid_model(Config)
```

## Training

Both architectures have **identical input/output interfaces**, so training code is the same:

```python
# Works with both architectures!
for eeg_data, source_data in train_loader:
    predictions = model(eeg_data)  # (batch, 500, 75) → (batch, 500, 994)
    loss = criterion(predictions, source_data)
    loss.backward()
    optimizer.step()
```

## Extracting Intermediate Features

The hybrid model allows feature visualization:

```python
features = model.get_intermediate_features(eeg_data)

# Returns dict with:
# - 'topological': Spatial maps (batch, 500, 64, 64)
# - 'cnn_features': CNN embeddings (batch, 500, 256)
# - 'output': Final predictions (batch, 500, 994)
```

## Model Components

### Topological Converter (`EEGTopologicalConverter`)
- Converts 75 EEG channels to 64×64 spatial topological maps
- Uses MNE library for electrode positioning
- Preserves true scalp geometry
- Applies cubic interpolation for smooth spatial representation

### CNN Encoder (`SpatialCNN` / `SpatialCNNLight`)
- Processes 2D topological maps per timestep
- Architecture: Conv2D → BatchNorm → ReLU → MaxPool
- Two variants:
  - **Standard**: 3 layers [32, 64, 128 channels] - better accuracy
  - **Light**: 2 layers [32, 64 channels] with depthwise separable convs - fewer parameters

### Transformer
- Standard transformer encoder with multi-head self-attention
- Processes temporal dependencies across time steps
- Same architecture for both model types

## Configuration

Add to `configs/config.py`:

```python
# CNN Encoder parameters
USE_CNN_ENCODER = False  # Set True to use hybrid model
TOPO_IMAGE_SIZE = 64     # Topological map resolution
ELECTRODE_FILE = "anatomy/electrode_75.mat"
CNN_CHANNELS = [32, 64, 128]  # CNN channel progression
CNN_KERNEL_SIZE = 3
CNN_TYPE = "standard"  # or "light"
```

## Performance

| Architecture | Parameters | Model Size | Forward Pass* | Throughput* |
|-------------|-----------|-----------|--------------|------------|
| Standard Linear | 6,040,546 | 23.04 MB | 166 ms | 48 samples/s |
| CNN-Hybrid | 6,212,802 | 23.70 MB | 4867 ms | 1.6 samples/s |

*Measured on CPU with batch_size=8, seq_len=500. GPU will be much faster.

**Note**: The CNN-Hybrid is slower due to on-the-fly topological conversion (scipy interpolation on CPU). This can be optimized by:
1. Pre-converting dataset offline
2. Caching converted maps
3. Using GPU-accelerated interpolation
4. Batch-optimized conversion

## Files

```
src/
├── models/
│   ├── transformer.py          # Standard transformer
│   ├── cnn_encoder.py          # CNN spatial encoders
│   ├── hybrid_model.py         # CNN-Transformer hybrid (NEW)
│   └── __init__.py
└── utils/
    └── topological_converter.py  # EEG→2D spatial mapping (NEW)

configs/
└── config.py                   # Configuration with CNN params

anatomy/
└── electrode_75.mat            # Electrode positions

test_cnn_transformer.py         # Comprehensive tests
examples_model_architectures.py # Usage examples
```

## Testing

Run comprehensive tests:

```bash
python test_cnn_transformer.py
```

See usage examples:

```bash
python examples_model_architectures.py
```

## When to Use Which?

**Use Standard Linear Transformer if:**
- You want fast training/inference
- Electrode spatial topology is not critical
- You have limited computational resources
- You're doing quick experiments

**Use CNN-Transformer Hybrid if:**
- Spatial electrode relationships are important
- You want to leverage 2D convolutions for feature extraction
- You're willing to trade speed for potentially better spatial feature learning
- You want modular architecture for research

## Future Improvements

1. **Pre-processing**: Convert entire dataset offline to avoid runtime overhead
2. **GPU interpolation**: Custom CUDA kernels for faster topological conversion
3. **Attention visualization**: Visualize what spatial patterns the model learns
4. **Transfer learning**: Pre-train CNN on large EEG datasets
5. **Adaptive resolution**: Dynamic topological map size based on frequency bands

## Citation

If you use the CNN-Transformer hybrid architecture, please cite both the original transformer work and the topological mapping approach (MNE library).

## Requirements

Additional requirement for CNN-Transformer Hybrid:

```
mne>=1.5.0  # For electrode positioning and topological mapping
```

Install with:
```bash
pip install mne>=1.5.0
```
