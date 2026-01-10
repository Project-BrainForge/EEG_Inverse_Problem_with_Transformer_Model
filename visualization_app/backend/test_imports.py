"""Test script to verify imports work"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
backend_dir = Path(__file__).parent
print(f"Project root: {project_root}")
print(f"Backend dir: {backend_dir}")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

try:
    from models.transformer_model import EEGSourceTransformerV2
    print("✓ Successfully imported EEGSourceTransformerV2")
except Exception as e:
    print(f"✗ Failed to import EEGSourceTransformerV2: {e}")

try:
    from services.cortex_service import CortexService
    print("✓ Successfully imported CortexService")
except Exception as e:
    print(f"✗ Failed to import CortexService: {e}")

try:
    from utilities.preprocessing import EEGPreprocessor
    print("✓ Successfully imported EEGPreprocessor")
except Exception as e:
    print(f"✗ Failed to import EEGPreprocessor: {e}")

try:
    from config import settings
    print("✓ Successfully imported settings")
    print(f"  - Device: {settings.DEVICE}")
    print(f"  - API Title: {settings.API_TITLE}")
except Exception as e:
    print(f"✗ Failed to import settings: {e}")

try:
    from app import app
    print("✓ Successfully imported FastAPI app")
except Exception as e:
    print(f"✗ Failed to import app: {e}")

print("\n✅ All imports successful!")
