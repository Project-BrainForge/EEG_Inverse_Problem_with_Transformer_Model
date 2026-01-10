"""
Configuration settings for the EEG Visualization API
Centralizes all configuration in one place
"""
from pathlib import Path
import torch

class Settings:
    """Application settings"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    ANATOMY_DIR: Path = BASE_DIR / "anatomy"
    SOURCE_DIR: Path = BASE_DIR / "source"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
    
    # Model settings
    DEFAULT_CHECKPOINT: str = "checkpoints/best_model.pt"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # API settings
    API_TITLE: str = "EEG Visualization API"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    
    # Cache settings
    ENABLE_CACHE: bool = True
    MAX_CACHE_SIZE: int = 100

settings = Settings()
