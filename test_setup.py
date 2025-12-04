"""
Test script to verify the setup is correct
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        import torch

        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import numpy as np

        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from scipy.io import loadmat
        import scipy

        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy import failed: {e}")
        return False

    try:
        import yaml

        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False

    return True


def test_modules():
    """Test that custom modules work"""
    print("\nTesting custom modules...")

    try:
        from models.transformer import EEGTransformer, create_model

        print("  ✓ Transformer model module")
    except ImportError as e:
        print(f"  ✗ Transformer model import failed: {e}")
        return False

    try:
        from data.dataset import EEGSourceDataset

        print("  ✓ Dataset module")
    except ImportError as e:
        print(f"  ✗ Dataset module import failed: {e}")
        return False

    try:
        from utils.helpers import AverageMeter, EarlyStopping

        print("  ✓ Utils module")
    except ImportError as e:
        print(f"  ✗ Utils module import failed: {e}")
        return False

    return True


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")

    try:
        import torch
        from models.transformer import create_model

        # Create encoder model
        model = create_model(
            model_type="encoder",
            input_channels=75,
            output_channels=994,
            d_model=128,
            nhead=4,
            num_layers=2,
        )

        # Test forward pass
        x = torch.randn(2, 500, 75)
        output = model(x)

        assert output.shape == (2, 500, 994), (
            f"Expected shape (2, 500, 994), got {output.shape}"
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully ({n_params:,} parameters)")
        print(f"  ✓ Forward pass works (input: {x.shape} → output: {output.shape})")

        return True

    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset loading...")

    try:
        from data.dataset import EEGSourceDataset
        from pathlib import Path

        # Check if dataset directory exists
        data_dir = Path(__file__).parent / "dataset_with_label"

        if not data_dir.exists():
            print(f"  ⚠ Dataset directory not found at {data_dir}")
            print("  ℹ Dataset test skipped (this is OK if you haven't added data yet)")
            return True

        # Try to load dataset
        dataset = EEGSourceDataset(str(data_dir), split="all", normalize=False)

        if len(dataset) == 0:
            print("  ⚠ No samples found in dataset directory")
            return True

        # Load a sample
        eeg, source = dataset[0]

        assert eeg.shape == (500, 75), f"Expected EEG shape (500, 75), got {eeg.shape}"
        assert source.shape == (500, 994), (
            f"Expected source shape (500, 994), got {source.shape}"
        )

        print(f"  ✓ Dataset loaded ({len(dataset)} samples)")
        print(f"  ✓ Sample shapes correct (EEG: {eeg.shape}, Source: {source.shape})")

        return True

    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")

    try:
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent / "configs" / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required fields
        required_fields = ["model", "num_epochs", "batch_size", "optimizer"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

        print(f"  ✓ Configuration loaded successfully")
        print(f"  ✓ Model type: {config['model']['type']}")
        print(f"  ✓ Batch size: {config['batch_size']}")
        print(f"  ✓ Learning rate: {config['optimizer']['lr']}")

        return True

    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("EEG Source Localization Transformer - Setup Test")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_imports()
    all_passed &= test_modules()
    all_passed &= test_model_creation()
    all_passed &= test_dataset()
    all_passed &= test_config()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Setup is complete.")
        print("\nYou can now train the model with:")
        print("  cd src")
        print("  python train.py --config ../configs/config.yaml")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
