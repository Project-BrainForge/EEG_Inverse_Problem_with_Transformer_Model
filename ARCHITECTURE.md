# Model Architecture Documentation

## Overview

This document describes the architecture of the EEG source localization transformer model.

## Problem Definition

**Input**: EEG signals
- Shape: `(batch_size, 500, 75)`
- 500 time points
- 75 EEG channels

**Output**: Brain source activity
- Shape: `(batch_size, 500, 994)`
- 500 time points
- 994 brain regions

**Task**: Time-series regression to predict brain source activity from EEG signals

## Model Architecture

### 1. Encoder-Only Transformer (Default)

```
┌─────────────────────────────────────────────────────┐
│                  Input EEG Data                      │
│                  (batch, 500, 75)                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Input Projection                        │
│         Linear: 75 → d_model (256)                   │
│                (batch, 500, 256)                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Positional Encoding                        │
│        Add temporal position info                    │
│                (batch, 500, 256)                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         Transformer Encoder Layer 1                  │
│  ┌──────────────────────────────────────────────┐   │
│  │    Multi-Head Self-Attention (8 heads)       │   │
│  │           Attention dimension: 32            │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │ Residual + LayerNorm           │
│                     ▼                                │
│  ┌──────────────────────────────────────────────┐   │
│  │      Feed-Forward Network (1024 dim)         │   │
│  │            Linear → GELU → Linear            │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │ Residual + LayerNorm           │
│                     ▼                                │
│                (batch, 500, 256)                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
              (Repeat 5 more times)
┌─────────────────────────────────────────────────────┐
│         Transformer Encoder Layer 6                  │
│              (same structure)                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Layer Normalization                     │
│                (batch, 500, 256)                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│             Output Projection                        │
│  ┌──────────────────────────────────────────────┐   │
│  │         Linear: 256 → 1024                   │   │
│  │                 GELU                         │   │
│  │               Dropout                        │   │
│  │         Linear: 1024 → 994                   │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            Predicted Source Activity                 │
│                (batch, 500, 994)                     │
└─────────────────────────────────────────────────────┘
```

### 2. Encoder-Decoder Transformer (Alternative)

```
┌─────────────────────────────────────────────────────┐
│                 Input EEG Data                       │
│                 (batch, 500, 75)                     │
└────────────────────┬────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │ ENCODER                          │
    ▼                                  │
┌─────────────────────────────┐       │
│  Input Projection: 75→256   │       │
│  + Positional Encoding      │       │
│  + 6 Encoder Layers         │       │
│    (Multi-Head Attention    │       │
│     + Feed-Forward)         │       │
└──────────┬──────────────────┘       │
           │ Encoded Features          │
           │ (batch, 500, 256)         │
           │                           │
    ┌──────┴───────────────────────────┘
    │ DECODER
    ▼
┌─────────────────────────────┐
│  Target Projection          │
│  + Positional Encoding      │
│  + 6 Decoder Layers         │
│    (Masked Self-Attention   │
│     + Cross-Attention       │
│     + Feed-Forward)         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Output Projection: 256→994 │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Predicted Source Activity  │
│     (batch, 500, 994)       │
└─────────────────────────────┘
```

## Key Components

### 1. Input Projection
- **Purpose**: Map EEG channels to model dimension
- **Operation**: Linear transformation
- **Input**: (batch, time, 75)
- **Output**: (batch, time, d_model)

### 2. Positional Encoding
- **Purpose**: Add temporal position information
- **Type**: Sinusoidal positional encoding
- **Formula**:
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
- **Properties**: Allows model to learn temporal patterns

### 3. Multi-Head Self-Attention
- **Number of heads**: 8 (default)
- **Dimension per head**: d_model / nhead = 256 / 8 = 32
- **Purpose**: Learn dependencies across time points
- **Operation**:
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  ```

### 4. Feed-Forward Network
- **Structure**: Linear → GELU → Dropout → Linear
- **Dimensions**: d_model → dim_feedforward → d_model
- **Default**: 256 → 1024 → 256
- **Purpose**: Non-linear transformation

### 5. Output Projection
- **Structure**: Two-layer MLP with GELU activation
- **Purpose**: Map from model dimension to brain regions
- **Dimensions**: 256 → 1024 → 994

## Hyperparameters

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 256 | Model embedding dimension |
| nhead | 8 | Number of attention heads |
| num_layers | 6 | Number of transformer layers |
| dim_feedforward | 1024 | Feed-forward network dimension |
| dropout | 0.1 | Dropout rate |
| max_seq_len | 500 | Maximum sequence length |

### Model Size

**Encoder-only model with default settings**:
- Total parameters: ~8.5M
- Trainable parameters: ~8.5M
- Memory (fp32): ~34 MB

**Approximate breakdown**:
- Input projection: 75 × 256 = 19,200
- Output projection: 256 × 1024 + 1024 × 994 ≈ 1.3M
- Transformer layers (6×): ~7M
- Positional encoding: Non-trainable

## Training Strategy

### Loss Function
- **Primary**: Mean Squared Error (MSE)
- **Formula**: `L = (1/N) Σ(y_pred - y_true)²`

### Optimizer
- **Type**: AdamW
- **Learning rate**: 0.0001
- **Weight decay**: 0.01
- **Beta1**: 0.9
- **Beta2**: 0.999

### Learning Rate Scheduling
- **Type**: ReduceLROnPlateau
- **Factor**: 0.5 (reduce LR by half)
- **Patience**: 10 epochs
- **Min LR**: 1e-7

### Regularization
- **Dropout**: 0.1 (applied after each sub-layer)
- **Weight decay**: 0.01 (L2 regularization)
- **Gradient clipping**: Max norm = 1.0

### Data Augmentation
- **Normalization**: Channel-wise z-score normalization
- **Note**: No additional augmentation (can be added)

## Evaluation Metrics

1. **Mean Squared Error (MSE)**
   - Primary metric for training
   - Measures average squared difference

2. **Mean Absolute Error (MAE)**
   - Interpretable metric
   - Less sensitive to outliers

3. **Pearson Correlation**
   - Measures linear relationship
   - Range: [-1, 1], higher is better

4. **R² Score**
   - Proportion of variance explained
   - Range: (-∞, 1], closer to 1 is better

5. **Relative Error**
   - Normalized error metric
   - Accounts for scale differences

## Design Choices

### Why Encoder-Only (Default)?

1. **Simplicity**: Easier to train and tune
2. **Efficiency**: Fewer parameters than encoder-decoder
3. **Suitability**: No autoregressive generation needed
4. **Performance**: Often sufficient for regression tasks

### Why Transformer?

1. **Long-range dependencies**: Can capture temporal patterns across all 500 time points
2. **Parallelization**: Faster training than RNNs
3. **Scalability**: Easy to scale up/down
4. **Flexibility**: Can handle variable-length sequences

### Alternative Architectures

Consider these alternatives if needed:

1. **CNN-based**: For local temporal patterns
2. **LSTM/GRU**: For sequential processing
3. **Hybrid**: CNN + Transformer
4. **U-Net style**: With skip connections

## Performance Considerations

### Memory Usage

For batch_size=8, d_model=256, seq_len=500:
- Input: 8 × 500 × 75 × 4 bytes ≈ 1.2 MB
- Activations: ~50 MB (with gradient)
- Model: ~34 MB
- **Total**: ~85 MB per batch

### Training Speed

On a typical GPU (e.g., NVIDIA RTX 3080):
- Forward pass: ~10ms per batch
- Backward pass: ~30ms per batch
- **Total**: ~40ms per batch
- **Throughput**: ~25 batches/sec = 200 samples/sec

### Scaling Guidelines

To scale the model:

**Increase capacity** (better accuracy, slower):
```yaml
d_model: 512
num_layers: 8
dim_feedforward: 2048
```

**Decrease capacity** (faster, may reduce accuracy):
```yaml
d_model: 128
num_layers: 4
dim_feedforward: 512
```

## Future Improvements

1. **Attention visualization**: Add tools to visualize attention patterns
2. **Multi-scale modeling**: Incorporate different temporal scales
3. **Channel-wise attention**: Learn importance of EEG channels
4. **Temporal convolution**: Add 1D convolutions for local patterns
5. **Ensemble methods**: Combine multiple models
6. **Transfer learning**: Pre-train on larger datasets

## References

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
3. Wu et al. (2020). "Deep Transformer Models for Time Series Forecasting"

