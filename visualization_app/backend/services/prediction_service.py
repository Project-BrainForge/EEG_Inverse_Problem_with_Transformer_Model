"""
Prediction operations service
Handles prediction-related calculations and formatting
"""
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Handles prediction-related operations"""
    
    @staticmethod
    def calculate_statistics(predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction statistics
        
        Args:
            predictions: Prediction array
            
        Returns:
            Dictionary with min, max, mean, std statistics
        """
        return {
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions))
        }
    
    @staticmethod
    def reshape_predictions(predictions: np.ndarray) -> np.ndarray:
        """
        Handle different prediction formats
        
        Args:
            predictions: Prediction array with varying dimensions
            
        Returns:
            Reshaped predictions in (samples, sources) format
            
        Raises:
            ValueError: If prediction shape is unexpected
        """
        if len(predictions.shape) == 3:
            # (batch, time, sources) -> (batch * time, sources)
            batch_size, time_points, num_sources = predictions.shape
            reshaped = predictions.reshape(-1, num_sources)
            logger.info(
                f"Reshaped 3D predictions from {predictions.shape} to {reshaped.shape}"
            )
            return reshaped
        
        elif len(predictions.shape) == 2:
            # Already correct format (samples, sources)
            return predictions
        
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    @classmethod
    def build_response(
        cls,
        predictions: np.ndarray,
        file_names: Optional[List[str]] = None
    ) -> dict:
        """
        Build prediction response dictionary
        
        Args:
            predictions: Prediction array
            file_names: Optional list of file names
            
        Returns:
            Dictionary with predictions, metadata, and statistics
        """
        predictions = cls.reshape_predictions(predictions)
        stats = cls.calculate_statistics(predictions)
        
        logger.info(
            f"Built response for {predictions.shape[0]} samples, "
            f"{predictions.shape[1]} sources"
        )
        
        return {
            "predictions": predictions.tolist(),
            "num_samples": predictions.shape[0],
            "num_sources": predictions.shape[1],
            "file_names": file_names,
            "statistics": stats
        }
    
    @staticmethod
    def filter_sample(
        predictions: np.ndarray,
        file_names: List[str],
        sample_idx: int
    ) -> tuple:
        """
        Filter predictions to specific sample index
        
        Args:
            predictions: Full prediction array
            file_names: List of file names
            sample_idx: Index of sample to extract
            
        Returns:
            Tuple of (filtered_predictions, filtered_file_names)
        """
        if sample_idx < 0 or sample_idx >= predictions.shape[0]:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sample index: {sample_idx}. "
                       f"Valid range: 0-{predictions.shape[0]-1}"
            )
        
        filtered_preds = predictions[sample_idx:sample_idx+1]
        filtered_names = [file_names[sample_idx]] if file_names and sample_idx < len(file_names) else []
        
        logger.info(f"Filtered to sample {sample_idx}")
        return filtered_preds, filtered_names
