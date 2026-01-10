"""
Pydantic models for API request/response validation
"""
from typing import List, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[List[float]]
    num_samples: int
    num_sources: int
    file_names: Optional[List[str]] = None
    statistics: dict


class CortexMeshResponse(BaseModel):
    """Response model for cortex mesh"""
    vertices: List[List[float]]
    faces: List[List[int]]
    num_vertices: int
    num_faces: int


class SubjectInfo(BaseModel):
    """Information about available subjects"""
    name: str
    num_files: int
    has_predictions: bool
