"""
EEG data preprocessing utilities
Eliminates code duplication across endpoints
"""
import numpy as np
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Handles EEG data preprocessing and validation"""
    
    EXPECTED_CHANNELS = 75
    TARGET_TIME_POINTS = 500
    
    @classmethod
    def validate_and_reshape(cls, data: np.ndarray) -> np.ndarray:
        """Ensure data has correct shape (time, channels)"""
        if data.shape[1] == cls.EXPECTED_CHANNELS and data.shape[0] != cls.EXPECTED_CHANNELS:
            return data  # Already correct (time, channels)
        
        elif data.shape[0] == cls.EXPECTED_CHANNELS and data.shape[1] != cls.EXPECTED_CHANNELS:
            logger.info(f"Transposing data from {data.shape} to (time, channels)")
            return data.T  # Transpose to (time, channels)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Expected data with {cls.EXPECTED_CHANNELS} channels. "
                       f"Got shape: {data.shape}"
            )
    
    @classmethod
    def pad_or_truncate(cls, data: np.ndarray) -> np.ndarray:
        """Ensure data has exactly TARGET_TIME_POINTS time points"""
        original_points = data.shape[0]
        
        if original_points == cls.TARGET_TIME_POINTS:
            return data
        
        if original_points > cls.TARGET_TIME_POINTS:
            logger.info(f"Truncating data from {original_points} to {cls.TARGET_TIME_POINTS} time points")
            return data[:cls.TARGET_TIME_POINTS, :]
        
        # Pad with zeros
        logger.info(f"Padding data from {original_points} to {cls.TARGET_TIME_POINTS} time points")
        padded = np.zeros((cls.TARGET_TIME_POINTS, data.shape[1]))
        padded[:original_points, :] = data
        return padded
    
    @staticmethod
    def center_data(data: np.ndarray) -> np.ndarray:
        """Remove mean from data (row-wise and column-wise)"""
        data = data - np.mean(data, axis=0, keepdims=True)
        data = data - np.mean(data, axis=1, keepdims=True)
        return data
    
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data by max absolute value"""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data
    
    @classmethod
    def preprocess(cls, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Input EEG data
            normalize: Whether to normalize by max absolute value
            
        Returns:
            Preprocessed data ready for model inference
        """
        logger.info(f"Preprocessing data with shape: {data.shape}")
        
        data = cls.validate_and_reshape(data)
        data = cls.pad_or_truncate(data)
        data = cls.center_data(data)
        
        if normalize:
            data = cls.normalize(data)
        
        logger.info(f"Preprocessing complete. Final shape: {data.shape}")
        return data
