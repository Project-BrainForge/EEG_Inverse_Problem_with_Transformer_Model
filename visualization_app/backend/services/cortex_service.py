"""
Cortex mesh data service
Handles loading and caching of cortex mesh data
"""
from pathlib import Path
import numpy as np
from typing import Dict, Optional
from fastapi import HTTPException
from scipy.io import loadmat
import logging

logger = logging.getLogger(__name__)


class CortexService:
    """Manages cortex mesh data loading and caching"""
    
    def __init__(self, anatomy_dir: Path):
        """
        Initialize cortex service
        
        Args:
            anatomy_dir: Path to anatomy data directory
        """
        self.anatomy_dir = anatomy_dir
        self._cache: Optional[Dict] = None
    
    def load_cortex_mesh(self) -> Dict[str, np.ndarray]:
        """
        Load cortex mesh data with caching
        
        Returns:
            Dictionary with 'vertices' and 'faces' keys
            
        Raises:
            HTTPException: If file not found or loading fails
        """
        if self._cache is not None:
            logger.info("Returning cached cortex mesh")
            return self._cache
        
        cortex_path = self.anatomy_dir / "fs_cortex_20k.mat"
        
        if not cortex_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cortex mesh file not found"
            )
        
        try:
            logger.info(f"Loading cortex mesh from: {cortex_path}")
            mat_data = loadmat(str(cortex_path))
            
            self._cache = {
                'vertices': mat_data['pos'],  # (N, 3) array
                'faces': mat_data['tri'] - 1  # MATLAB 1-based to 0-based indexing
            }
            
            logger.info(
                f"Loaded cortex mesh: {self._cache['vertices'].shape[0]} vertices, "
                f"{self._cache['faces'].shape[0]} faces"
            )
            return self._cache
            
        except KeyError as e:
            logger.error(f"Missing expected key in cortex file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid cortex mesh file format. Missing key: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error loading cortex mesh: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading cortex mesh: {str(e)}"
            )
    
    def get_vertices(self) -> np.ndarray:
        """Get cortex vertices"""
        return self.load_cortex_mesh()['vertices']
    
    def get_faces(self) -> np.ndarray:
        """Get cortex faces"""
        return self.load_cortex_mesh()['faces']
    
    def clear_cache(self):
        """Clear cached cortex data"""
        logger.info("Clearing cortex mesh cache")
        self._cache = None
