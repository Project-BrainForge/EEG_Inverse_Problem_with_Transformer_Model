"""
Example: Using Different Model Architectures

Demonstrates how to use:
1. Standard Linear Transformer (original)
2. CNN-Transformer Hybrid (new, separated components)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.models import EEGTransformer, CNNTransformerHybrid
from configs.config import Config


def example_1_standard_transformer():
    """Example 1: Standard transformer with linear projection."""
    print("="*70)
    print("EXAMPLE 1: Standard Transformer (Linear Projection)")
    print("="*70)
    
    # Create model
    model = EEGTransformer(
        input_channels=Config.EEG_CHANNELS,
        output_channels=Config.SOURCE_REGIONS,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        use_cnn_encoder=False  # Standard linear projection
    )
    
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    batch_size = 8
    x = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print()


def example_2_hybrid_model():
    """Example 2: CNN-Transformer hybrid with separated components."""
    print("="*70)
    print("EXAMPLE 2: CNN-Transformer Hybrid (Separated Components)")
    print("="*70)
    
    # Create hybrid model
    model = CNNTransformerHybrid(
        eeg_channels=Config.EEG_CHANNELS,
        output_channels=Config.SOURCE_REGIONS,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        max_seq_len=Config.SEQ_LEN,
        topo_image_size=Config.TOPO_IMAGE_SIZE,
        electrode_file=Config.ELECTRODE_FILE,
        cnn_channels=Config.CNN_CHANNELS,
        cnn_kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_type=Config.CNN_TYPE
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    - CNN: {sum(p.numel() for p in model.cnn_encoder.parameters()):,}")
    print(f"    - Transformer: {sum(p.numel() for p in model.transformer.parameters()):,}")
    
    # Create dummy data
    batch_size = 8
    x = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    
    # Get intermediate features for analysis
    with torch.no_grad():
        features = model.get_intermediate_features(x)
    
    print(f"✓ Intermediate features extracted")
    print(f"  Topological maps: {features['topological'].shape}")
    print(f"  CNN features: {features['cnn_features'].shape}")
    print()


def example_3_comparison():
    """Example 3: Side-by-side comparison."""
    print("="*70)
    print("EXAMPLE 3: Architecture Comparison")
    print("="*70)
    
    models = {
        'Standard Linear': EEGTransformer(
            input_channels=Config.EEG_CHANNELS,
            output_channels=Config.SOURCE_REGIONS,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            use_cnn_encoder=False
        ),
        'CNN-Transformer Hybrid': CNNTransformerHybrid(
            eeg_channels=Config.EEG_CHANNELS,
            output_channels=Config.SOURCE_REGIONS,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            topo_image_size=Config.TOPO_IMAGE_SIZE,
            electrode_file=Config.ELECTRODE_FILE,
            cnn_channels=Config.CNN_CHANNELS
        )
    }
    
    print(f"\n{'Model':<30} {'Parameters':<15} {'Size (MB)':<12}")
    print("-"*70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"{name:<30} {params:<15,} {size_mb:<12.2f}")
    
    print()


def example_4_training_usage():
    """Example 4: How to use in training loop."""
    print("="*70)
    print("EXAMPLE 4: Training Loop Usage")
    print("="*70)
    
    print("""
# Choose your model architecture:

# Option 1: Standard Linear Transformer
from src.models import EEGTransformer

model = EEGTransformer(
    input_channels=75,
    output_channels=994,
    d_model=256,
    nhead=8,
    num_layers=6,
    use_cnn_encoder=False  # Standard
).to(device)

# Option 2: CNN-Transformer Hybrid
from src.models import CNNTransformerHybrid

model = CNNTransformerHybrid(
    eeg_channels=75,
    output_channels=994,
    d_model=256,
    nhead=8,
    num_layers=6,
    topo_image_size=64,
    electrode_file='anatomy/electrode_75.mat',
    cnn_channels=[32, 64, 128]
).to(device)

# Option 3: Using config factory
from src.models import create_hybrid_model
from configs.config import Config

model = create_hybrid_model(Config).to(device)

# Then train normally:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    for eeg_data, source_data in train_loader:
        eeg_data = eeg_data.to(device)
        source_data = source_data.to(device)
        
        # Forward pass (same for both architectures!)
        predictions = model(eeg_data)
        
        # Compute loss and backprop
        loss = criterion(predictions, source_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    """)


def main():
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE EXAMPLES")
    print("="*70 + "\n")
    
    # Run examples
    example_1_standard_transformer()
    example_2_hybrid_model()
    example_3_comparison()
    example_4_training_usage()
    
    print("="*70)
    print("✓ All examples completed successfully!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Both architectures have the same input/output interface")
    print("2. Hybrid model separates CNN and Transformer for better modularity")
    print("3. Can extract intermediate features from hybrid model")
    print("4. Training code is identical for both architectures")


if __name__ == "__main__":
    main()
