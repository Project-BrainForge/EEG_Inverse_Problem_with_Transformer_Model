"""
MAT file loading utilities
Eliminates duplicated file loading logic
"""
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import List
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class MATFileLoader:
    """Utility for loading MAT files with consistent error handling"""
    
    POSSIBLE_DATA_KEYS = ['data', 'eeg_data', 'eeg', 'EEG']
    
    @staticmethod
    def load_mat_file(file_path: Path) -> dict:
        """
        Load MAT file with error handling
        
        Args:
            file_path: Path to the MAT file
            
        Returns:
            Dictionary containing MAT file contents
            
        Raises:
            HTTPException: If file not found or loading fails
        """
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"File not found: {file_path}"
            )
        
        try:
            logger.info(f"Loading MAT file: {file_path}")
            return loadmat(str(file_path))
        except Exception as e:
            logger.error(f"Error loading MAT file {file_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading MAT file: {str(e)}"
            )
    
    @classmethod
    def extract_eeg_data(cls, mat_data: dict) -> np.ndarray:
        """
        Extract EEG data from MAT file with flexible key matching
        
        Args:
            mat_data: Dictionary from loadmat
            
        Returns:
            EEG data array
            
        Raises:
            HTTPException: If no valid data key found
        """
        for key in cls.POSSIBLE_DATA_KEYS:
            if key in mat_data:
                logger.info(f"Found EEG data under key: '{key}'")
                return mat_data[key]
        
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise HTTPException(
            status_code=400,
            detail=f"Could not find EEG data. Expected one of: {cls.POSSIBLE_DATA_KEYS}. "
                   f"Found: {available_keys}"
        )
    
    @staticmethod
    def extract_file_names(mat_data: dict) -> List[str]:
        """
        Extract file names from MAT file
        
        Args:
            mat_data: Dictionary from loadmat
            
        Returns:
            List of file names (may be empty)
        """
        file_names_raw = mat_data.get('file_names', [])
        if len(file_names_raw) == 0:
            return []
        
        file_names = []
        for f in file_names_raw:
            if isinstance(f, np.ndarray):
                # MATLAB cell array
                if f.size > 0:
                    file_names.append(
                        str(f.flat[0]) if hasattr(f, 'flat') else str(f)
                    )
            else:
                file_names.append(str(f))
        
        logger.info(f"Extracted {len(file_names)} file names")
        return file_names
