"""
Test script for CNN-Enhanced Transformer Model
Tests the end-to-end flow with topological spatial feature extraction.
"""

import sys
import os
from pathlib import Path
import time
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.transformer import EEGTransformer
from src.utils.topological_converter import EEGTopologicalConverter
from configs.config import Config


def test_topological_converter():
    """Test the topological converter standalone."""
    print("="*80)
    print("TEST 1: Topological Converter")
    print("="*80)
    
    converter = EEGTopologicalConverter(
        electrode_file=Config.ELECTRODE_FILE,
        image_size=(Config.TOPO_IMAGE_SIZE, Config.TOPO_IMAGE_SIZE),
        sphere='auto',
        normalize=True
    )
    
    print(f"\n✓ Converter initialized")
    print(f"  - Channels: {converter.num_channels}")
    print(f"  - Image size: {converter.image_size}")
    
    # Test single sample conversion
    batch_size = 8
    eeg_data = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS)
    
    start_time = time.time()
    topo_maps = converter.to_torch(eeg_data, device='cpu', verbose=False)
    conversion_time = time.time() - start_time
    
    print(f"\n✓ Conversion successful")
    print(f"  - Input shape: {eeg_data.shape}")
    print(f"  - Output shape: {topo_maps.shape}")
    print(f"  - Conversion time: {conversion_time*1000:.2f} ms")
    print(f"  - Time per sample: {conversion_time/batch_size*1000:.2f} ms")
    
    assert topo_maps.shape == (batch_size, Config.SEQ_LEN, 
                                Config.TOPO_IMAGE_SIZE, Config.TOPO_IMAGE_SIZE)
    
    return converter


def test_model_comparison():
    """Compare linear projection vs CNN encoder models."""
    print("\n" + "="*80)
    print("TEST 2: Model Comparison (Linear vs CNN)")
    print("="*80)
    
    batch_size = 8
    device = Config.DEVICE
    
    # Create test input
    x = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS).to(device)
    
    # Test 1: Linear projection model
    print("\n" + "-"*80)
    print("Model 1: Linear Projection (Original)")
    print("-"*80)
    
    model_linear = EEGTransformer(
        input_channels=Config.EEG_CHANNELS,
        output_channels=Config.SOURCE_REGIONS,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        use_cnn_encoder=False
    ).to(device)
    
    model_linear.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model_linear(x)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        output_linear = model_linear(x)
    linear_time = time.time() - start_time
    
    linear_params = sum(p.numel() for p in model_linear.parameters())
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output_linear.shape}")
    print(f"  - Parameters: {linear_params:,}")
    print(f"  - Forward time: {linear_time*1000:.2f} ms")
    print(f"  - Throughput: {batch_size/linear_time:.2f} samples/sec")
    
    # Test 2: CNN encoder model
    print("\n" + "-"*80)
    print("Model 2: CNN Spatial Encoder (New)")
    print("-"*80)
    
    model_cnn = EEGTransformer(
        input_channels=Config.EEG_CHANNELS,
        output_channels=Config.SOURCE_REGIONS,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        use_cnn_encoder=True,
        topo_image_size=Config.TOPO_IMAGE_SIZE,
        electrode_file=Config.ELECTRODE_FILE,
        cnn_channels=Config.CNN_CHANNELS,
        cnn_kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_type=Config.CNN_TYPE
    ).to(device)
    
    model_cnn.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model_cnn(x)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        output_cnn = model_cnn(x)
    cnn_time = time.time() - start_time
    
    cnn_params = sum(p.numel() for p in model_cnn.parameters())
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output_cnn.shape}")
    print(f"  - Parameters: {cnn_params:,}")
    print(f"  - Forward time: {cnn_time*1000:.2f} ms")
    print(f"  - Throughput: {batch_size/cnn_time:.2f} samples/sec")
    
    # Comparison
    print("\n" + "-"*80)
    print("Comparison Summary")
    print("-"*80)
    print(f"Parameter difference: {cnn_params - linear_params:+,} ({(cnn_params/linear_params - 1)*100:+.1f}%)")
    print(f"Speed difference: {cnn_time - linear_time:+.4f}s ({(cnn_time/linear_time - 1)*100:+.1f}%)")
    print(f"Overhead per sample: {(cnn_time - linear_time)/batch_size*1000:.2f} ms")
    
    return model_linear, model_cnn


def test_gradient_flow():
    """Test that gradients flow correctly through the CNN encoder."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow")
    print("="*80)
    
    batch_size = 4
    device = Config.DEVICE
    
    model = EEGTransformer(
        input_channels=Config.EEG_CHANNELS,
        output_channels=Config.SOURCE_REGIONS,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=2,  # Fewer layers for faster test
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        use_cnn_encoder=True,
        topo_image_size=Config.TOPO_IMAGE_SIZE,
        electrode_file=Config.ELECTRODE_FILE,
        cnn_channels=Config.CNN_CHANNELS,
        cnn_kernel_size=Config.CNN_KERNEL_SIZE
    ).to(device)
    
    model.train()
    
    # Create dummy data
    x = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS, 
                   requires_grad=True).to(device)
    target = torch.randn(batch_size, Config.SEQ_LEN, Config.SOURCE_REGIONS).to(device)
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    print(f"\n✓ Gradient flow test passed")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Gradients computed: {has_gradients}")
    print(f"  - All parameters have gradients: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")
    
    assert has_gradients, "No gradients computed!"
    
    return model


def test_memory_usage():
    """Test memory usage of both models."""
    print("\n" + "="*80)
    print("TEST 4: Memory Usage")
    print("="*80)
    
    batch_size = 8
    device = 'cpu'  # Use CPU for consistent memory measurement
    
    models_config = [
        ('Linear Projection', False),
        ('CNN Encoder', True)
    ]
    
    for name, use_cnn in models_config:
        print(f"\n{name}:")
        
        model = EEGTransformer(
            input_channels=Config.EEG_CHANNELS,
            output_channels=Config.SOURCE_REGIONS,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            use_cnn_encoder=use_cnn,
            topo_image_size=Config.TOPO_IMAGE_SIZE,
            electrode_file=Config.ELECTRODE_FILE,
            cnn_channels=Config.CNN_CHANNELS,
            cnn_kernel_size=Config.CNN_KERNEL_SIZE
        ).to(device)
        
        # Calculate model size
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        # Calculate activation size (approximate)
        x = torch.randn(batch_size, Config.SEQ_LEN, Config.EEG_CHANNELS).to(device)
        with torch.no_grad():
            output = model(x)
        
        activation_size_mb = (x.element_size() * x.nelement() + 
                             output.element_size() * output.nelement()) / 1024**2
        
        print(f"  - Model size: {model_size_mb:.2f} MB")
        print(f"  - Activation size (I/O): {activation_size_mb:.2f} MB")
        print(f"  - Total: {model_size_mb + activation_size_mb:.2f} MB")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CNN-ENHANCED TRANSFORMER MODEL TESTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Device: {Config.DEVICE}")
    print(f"  - EEG Channels: {Config.EEG_CHANNELS}")
    print(f"  - Source Regions: {Config.SOURCE_REGIONS}")
    print(f"  - Sequence Length: {Config.SEQ_LEN}")
    print(f"  - Batch Size: {Config.BATCH_SIZE}")
    print(f"  - Topological Image Size: {Config.TOPO_IMAGE_SIZE}x{Config.TOPO_IMAGE_SIZE}")
    print(f"  - CNN Channels: {Config.CNN_CHANNELS}")
    
    try:
        # Run tests
        test_topological_converter()
        test_model_comparison()
        test_gradient_flow()
        test_memory_usage()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe CNN-enhanced transformer is ready for training.")
        print("To enable CNN encoder, set USE_CNN_ENCODER=True in configs/config.py")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
