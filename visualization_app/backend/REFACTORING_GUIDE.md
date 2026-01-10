# Backend Refactoring Guide

## Overview
This document outlines refactoring opportunities for the FastAPI backend to improve code quality, maintainability, and adherence to KISS (Keep It Simple, Stupid) and DRY (Don't Repeat Yourself) principles.

## Current Structure Issues

### 1. **Single Monolithic File (598 lines)**
- **Problem**: All code in `app.py` makes it hard to navigate and maintain
- **Impact**: Difficult to test, understand, and modify specific functionality

### 2. **Code Duplication**
- Data loading logic repeated in multiple places
- Data preprocessing duplicated across endpoints
- Error handling patterns repeated
- MAT file field searching (`data`, `eeg_data`, `eeg`, `EEG`) appears multiple times

### 3. **Mixed Concerns**
- Business logic, data processing, API routes, and configuration all in one file
- Global state management mixed with request handling
- Model loading logic mixed with API endpoints

### 4. **Poor Error Handling**
- Generic exception catching in some places
- Inconsistent error messages
- Debug print statements instead of proper logging

---

## Recommended Refactoring Structure

```
backend/
├── app.py                      # FastAPI app initialization & routes only
├── requirements.txt
├── config.py                   # Configuration and constants
├── schemas/
│   ├── __init__.py
│   └── schemas.py              # Pydantic models (renamed to avoid conflict)
├── services/
│   ├── __init__.py
│   ├── cortex_service.py       # Cortex mesh operations
│   ├── model_service.py        # Model loading & inference
│   ├── prediction_service.py   # Prediction operations
│   └── data_loader.py          # MAT file loading utilities
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing
│   └── file_utils.py           # File operations
├── middleware/
│   ├── __init__.py
│   └── cache.py                # Caching logic
└── tests/
    ├── __init__.py
    ├── test_services.py
    └── test_preprocessing.py
```

---

## Specific Refactoring Recommendations

### 1. **Extract Configuration** ✅ KISS Principle

**Current Problem:**
```python
# Scattered throughout code
BASE_DIR = Path(__file__).parent.parent.parent
ANATOMY_DIR = BASE_DIR / "anatomy"
SOURCE_DIR = BASE_DIR / "source"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Recommended: `config.py`**
```python
from pathlib import Path
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
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
    
    class Config:
        case_sensitive = True

settings = Settings()
```

**Benefits:**
- Single source of truth
- Easy to modify without touching business logic
- Testable configuration

---

### 2. **Extract Data Loading** ✅ DRY Principle

**Current Problem:** MAT file loading duplicated 3+ times

**Recommended: `services/data_loader.py`**
```python
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import Optional, List
from fastapi import HTTPException

class MATFileLoader:
    """Utility for loading MAT files with consistent error handling"""
    
    POSSIBLE_DATA_KEYS = ['data', 'eeg_data', 'eeg', 'EEG']
    
    @staticmethod
    def load_mat_file(file_path: Path) -> dict:
        """Load MAT file with error handling"""
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"File not found: {file_path}"
            )
        
        try:
            return loadmat(str(file_path))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading MAT file: {str(e)}"
            )
    
    @classmethod
    def extract_eeg_data(cls, mat_data: dict) -> np.ndarray:
        """Extract EEG data from MAT file with flexible key matching"""
        for key in cls.POSSIBLE_DATA_KEYS:
            if key in mat_data:
                return mat_data[key]
        
        raise HTTPException(
            status_code=400,
            detail=f"Could not find EEG data. Expected one of: {cls.POSSIBLE_DATA_KEYS}. "
                   f"Found: {list(mat_data.keys())}"
        )
    
    @staticmethod
    def extract_file_names(mat_data: dict) -> List[str]:
        """Extract file names from MAT file"""
        file_names_raw = mat_data.get('file_names', [])
        if len(file_names_raw) == 0:
            return []
        
        file_names = []
        for f in file_names_raw:
            if isinstance(f, np.ndarray):
                if f.size > 0:
                    file_names.append(
                        str(f.flat[0]) if hasattr(f, 'flat') else str(f)
                    )
            else:
                file_names.append(str(f))
        
        return file_names
```

**Benefits:**
- Eliminates 3 copies of similar code
- Consistent error messages
- Single place to fix bugs
- Easier to test

---

### 3. **Extract Data Preprocessing** ✅ DRY Principle

**Current Problem:** Preprocessing logic duplicated

**Recommended: `utils/preprocessing.py`**
```python
import numpy as np
from typing import Tuple
from fastapi import HTTPException

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
        if data.shape[0] == cls.TARGET_TIME_POINTS:
            return data
        
        if data.shape[0] > cls.TARGET_TIME_POINTS:
            return data[:cls.TARGET_TIME_POINTS, :]
        
        # Pad with zeros
        padded = np.zeros((cls.TARGET_TIME_POINTS, data.shape[1]))
        padded[:data.shape[0], :] = data
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
        """Complete preprocessing pipeline"""
        data = cls.validate_and_reshape(data)
        data = cls.pad_or_truncate(data)
        data = cls.center_data(data)
        
        if normalize:
            data = cls.normalize(data)
        
        return data
```

**Benefits:**
- Single preprocessing pipeline
- Easy to modify preprocessing steps
- Testable in isolation
- Clear documentation

---

### 4. **Extract Services** ✅ KISS + DRY Principles

**Recommended: `services/cortex_service.py`**
```python
from pathlib import Path
import numpy as np
from typing import Dict, Optional
from fastapi import HTTPException
from scipy.io import loadmat

class CortexService:
    """Manages cortex mesh data loading and caching"""
    
    def __init__(self, anatomy_dir: Path):
        self.anatomy_dir = anatomy_dir
        self._cache: Optional[Dict] = None
    
    def load_cortex_mesh(self) -> Dict[str, np.ndarray]:
        """Load cortex mesh data with caching"""
        if self._cache is not None:
            return self._cache
        
        cortex_path = self.anatomy_dir / "fs_cortex_20k.mat"
        
        if not cortex_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Cortex mesh file not found"
            )
        
        try:
            mat_data = loadmat(str(cortex_path))
            
            self._cache = {
                'vertices': mat_data['pos'],
                'faces': mat_data['tri'] - 1  # MATLAB 1-based to 0-based
            }
            
            return self._cache
            
        except Exception as e:
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
```

**Recommended: `services/model_service.py`**
```python
import torch
from pathlib import Path
from typing import Optional
from fastapi import HTTPException

class ModelService:
    """Manages model loading and inference"""
    
    def __init__(self, base_dir: Path, device: str):
        self.base_dir = base_dir
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint with caching"""
        if self._model is not None:
            return self._model
        
        # Resolve path
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.base_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {checkpoint_path}"
            )
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            
            # Initialize model with config
            model = self._create_model(checkpoint.get('config'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self._model = model
            return self._model
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )
    
    def _create_model(self, config: Optional[dict]):
        """Create model instance from config"""
        from models.transformer_model import EEGSourceTransformerV2
        
        if config:
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
        """Run inference on data"""
        if self._model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not loaded"
            )
        
        with torch.no_grad():
            return self._model(data)
```

---

### 5. **Extract Response Builders** ✅ KISS Principle

**Recommended: `services/prediction_service.py`**
```python
import numpy as np
from typing import List, Optional, Dict

class PredictionService:
    """Handles prediction-related operations"""
    
    @staticmethod
    def calculate_statistics(predictions: np.ndarray) -> Dict[str, float]:
        """Calculate prediction statistics"""
        return {
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions))
        }
    
    @staticmethod
    def reshape_predictions(predictions: np.ndarray) -> np.ndarray:
        """Handle different prediction formats"""
        if len(predictions.shape) == 3:
            # (batch, time, sources) -> (batch * time, sources)
            batch_size, time_points, num_sources = predictions.shape
            return predictions.reshape(-1, num_sources)
        
        elif len(predictions.shape) == 2:
            # Already correct format
            return predictions
        
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    @classmethod
    def build_response(
        cls,
        predictions: np.ndarray,
        file_names: Optional[List[str]] = None
    ) -> dict:
        """Build prediction response"""
        predictions = cls.reshape_predictions(predictions)
        stats = cls.calculate_statistics(predictions)
        
        return {
            "predictions": predictions.tolist(),
            "num_samples": predictions.shape[0],
            "num_sources": predictions.shape[1],
            "file_names": file_names,
            "statistics": stats
        }
```

---

### 6. **Simplified API Routes** ✅ KISS Principle

**Recommended: `app.py` (refactored)**
```python
"""
FastAPI backend for EEG Source Localization Visualization
"""
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.schemas import PredictionResponse, CortexMeshResponse, SubjectInfo
from services.cortex_service import CortexService
from services.model_service import ModelService
from services.prediction_service import PredictionService
from services.data_loader import MATFileLoader
from utils.preprocessing import EEGPreprocessor

# Initialize services
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)
cortex_service = CortexService(settings.ANATOMY_DIR)
model_service = ModelService(settings.BASE_DIR, settings.DEVICE)
prediction_service = PredictionService()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "endpoints": {
            "cortex_mesh": "/api/cortex-mesh",
            "subjects": "/api/subjects",
            "predictions": "/api/predictions/{subject}",
            "predict": "/api/predict/{subject}",
            "upload": "/api/upload-and-predict",
            "health": "/api/health"
        }
    }


@app.get("/api/cortex-mesh", response_model=CortexMeshResponse)
async def get_cortex_mesh():
    """Get cortex mesh data for visualization"""
    cortex_data = cortex_service.load_cortex_mesh()
    
    return CortexMeshResponse(
        vertices=cortex_data['vertices'].tolist(),
        faces=cortex_data['faces'].tolist(),
        num_vertices=len(cortex_data['vertices']),
        num_faces=len(cortex_data['faces'])
    )


@app.post("/api/upload-and-predict")
async def upload_and_predict(
    file: UploadFile = File(...),
    checkpoint: str = Query(settings.DEFAULT_CHECKPOINT),
    normalize: bool = Query(True)
):
    """Upload MAT file and get predictions"""
    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="File must be a .mat file")
    
    # Load and validate data
    mat_data = MATFileLoader.load_mat_file(file)
    eeg_data = MATFileLoader.extract_eeg_data(mat_data)
    
    # Preprocess
    processed_data = EEGPreprocessor.preprocess(eeg_data, normalize=normalize)
    
    # Load model and predict
    model = model_service.load_model(checkpoint)
    data_tensor = torch.from_numpy(processed_data[np.newaxis, :, :]).to(
        model_service.device, torch.float
    )
    predictions = model_service.predict(data_tensor)
    
    # Build response
    return prediction_service.build_response(
        predictions.cpu().numpy(),
        file_names=[file.filename]
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(model_service.device),
        "model_loaded": model_service._model is not None,
        "cortex_loaded": cortex_service._cache is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
```

---

## Additional Improvements

### 7. **Replace Print Statements with Logging** ✅ Best Practice

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Usage
logger.info(f"Loaded cortex mesh: {vertices.shape[0]} vertices")
logger.warning("Model not found in cache, loading from disk")
logger.error(f"Error loading predictions: {str(e)}")
```

### 8. **Add Type Hints** ✅ Code Quality

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def process_data(
    data: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process data and return results with statistics"""
    # Implementation
    pass
```

### 9. **Add Unit Tests** ✅ Quality Assurance

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from utils.preprocessing import EEGPreprocessor

def test_validate_and_reshape():
    # Test correct shape
    data = np.random.rand(500, 75)
    result = EEGPreprocessor.validate_and_reshape(data)
    assert result.shape == (500, 75)
    
    # Test transpose
    data = np.random.rand(75, 500)
    result = EEGPreprocessor.validate_and_reshape(data)
    assert result.shape == (500, 75)
    
    # Test invalid shape
    data = np.random.rand(100, 100)
    with pytest.raises(HTTPException):
        EEGPreprocessor.validate_and_reshape(data)
```

### 10. **Add Dependency Injection** ✅ Testability

```python
from fastapi import Depends

def get_cortex_service() -> CortexService:
    """Dependency injection for cortex service"""
    return cortex_service

@app.get("/api/cortex-mesh")
async def get_cortex_mesh(
    service: CortexService = Depends(get_cortex_service)
):
    """Get cortex mesh with injected dependency"""
    cortex_data = service.load_cortex_mesh()
    # ...
```

---

## Migration Strategy

### Phase 1: Extract Utilities (Low Risk)
1. Create `config.py`
2. Create `utils/preprocessing.py`
3. Create `services/data_loader.py`
4. Update imports in `app.py`
5. Test endpoints

### Phase 2: Extract Services (Medium Risk)
1. Create `services/cortex_service.py`
2. Create `services/model_service.py`
3. Create `services/prediction_service.py`
4. Update `app.py` to use services
5. Test endpoints

### Phase 3: Clean Up Routes (Low Risk)
1. Simplify route handlers
2. Add logging
3. Remove print statements
4. Add type hints

### Phase 4: Add Tests (Quality)
1. Add unit tests for utilities
2. Add integration tests for services
3. Add API tests for routes

---

## Expected Benefits

### Code Quality
- **Lines per file**: 598 → ~50-150 per file
- **Duplication**: 3+ copies → 1 copy
- **Testability**: Hard → Easy
- **Maintainability**: Low → High

### Development Speed
- **Bug fixes**: Easier to locate and fix
- **New features**: Faster to add
- **Onboarding**: New developers understand faster

### Reliability
- **Error handling**: Consistent across all endpoints
- **Testing**: More comprehensive
- **Debugging**: Easier with proper logging

---

## Key Principles Applied

### KISS (Keep It Simple, Stupid)
- ✅ Each service has one responsibility
- ✅ Clear naming conventions
- ✅ Simple, focused functions
- ✅ No complex inheritance hierarchies

### DRY (Don't Repeat Yourself)
- ✅ MAT file loading in one place
- ✅ Preprocessing in one place
- ✅ Error handling patterns reused
- ✅ Configuration centralized

### Additional Principles
- **Separation of Concerns**: Routes, business logic, and data access separated
- **Single Responsibility**: Each class/function has one clear purpose
- **Dependency Injection**: Makes testing easier
- **Fail Fast**: Validate inputs early
- **Type Safety**: Type hints throughout

---

## Conclusion

This refactoring will transform a 598-line monolithic file into a well-structured, maintainable codebase that follows industry best practices. The modular structure makes it easier to:

1. Test individual components
2. Fix bugs quickly
3. Add new features
4. Onboard new developers
5. Scale the application

**Start with Phase 1** (utilities) as it's low-risk and provides immediate benefits.
