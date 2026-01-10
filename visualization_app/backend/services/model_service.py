"""
Model loading and inference service
Handles transformer model operations
"""
import sys
from pathlib import Path
import torch
from typing import Optional
from fastapi import HTTPException
import logging

# Add parent directory to path for model imports
# Navigate from services/ -> backend/ -> visualization_app/ -> project root
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.transformer_model import EEGSourceTransformerV2

logger = logging.getLogger(__name__)


class ModelService:
    """Manages model loading and inference"""
    
    def __init__(self, base_dir: Path, device: str):
        """
        Initialize model service
        
        Args:
            base_dir: Base directory for resolving relative paths
            device: Device to load model on ('cuda' or 'cpu')
        """
        self.base_dir = base_dir
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
        self._checkpoint_path: Optional[Path] = None
    
    def load_model(self, checkpoint_path: str):
        """
        Load model from checkpoint with caching
        
        Args:
            checkpoint_path: Path to model checkpoint (absolute or relative)
            
        Returns:
            Loaded model
            
        Raises:
            HTTPException: If checkpoint not found or loading fails
        """
        # Resolve path
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_absolute():
            checkpoint_path_obj = self.base_dir / checkpoint_path_obj
        
        # Return cached model if same checkpoint
        if self._model is not None and self._checkpoint_path == checkpoint_path_obj:
            logger.info(f"Returning cached model from: {checkpoint_path_obj}")
            return self._model
        
        if not checkpoint_path_obj.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {checkpoint_path_obj}"
            )
        
        try:
            logger.info(f"Loading model from: {checkpoint_path_obj}")
            checkpoint = torch.load(str(checkpoint_path_obj), map_location=self.device)
            
            # Initialize model with config
            model = self._create_model(checkpoint.get('config'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self._model = model
            self._checkpoint_path = checkpoint_path_obj
            
            epoch = checkpoint.get('epoch', 'unknown')
            logger.info(f"Loaded model from epoch {epoch}")
            return self._model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )
    
    def _create_model(self, config: Optional[dict]) -> torch.nn.Module:
        """
        Create model instance from config
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Initialized model
        """
        if config:
            logger.info("Creating model with saved config")
            return EEGSourceTransformerV2(
                eeg_channels=config['EEG_CHANNELS'],
                source_regions=config['SOURCE_REGIONS'],
                d_model=config['D_MODEL'],
                nhead=config['NHEAD'],
                num_layers=config['NUM_LAYERS'],
                dim_feedforward=config['DIM_FEEDFORWARD'],
                dropout=config['DROPOUT']
            ).to(self.device)
        else:
            # Default configuration
            logger.info("Creating model with default config")
            return EEGSourceTransformerV2(
                eeg_channels=75,
                source_regions=994,
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.1
            ).to(self.device)
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run inference on data
        
        Args:
            data: Input tensor
            
        Returns:
            Model predictions
            
        Raises:
            HTTPException: If model not loaded
        """
        if self._model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not loaded. Call load_model() first."
            )
        
        logger.info(f"Running inference on data shape: {data.shape}")
        with torch.no_grad():
            predictions = self._model(data)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
    
    def clear_cache(self):
        """Clear cached model"""
        logger.info("Clearing model cache")
        self._model = None
        self._checkpoint_path = None
