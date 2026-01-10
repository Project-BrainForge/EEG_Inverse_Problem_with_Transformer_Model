"""
FastAPI backend for EEG Source Localization Visualization
Serves model predictions and cortex mesh data
"""
import os
import sys
from pathlib import Path

# CRITICAL: Add paths before any local imports
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent.parent

# Add both backend and project root to path
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

# Debug: Print paths
print(f"Backend dir: {backend_dir}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

from typing import List, Optional
import numpy as np
from scipy.io import savemat, loadmat
import torch
import shutil
import logging

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from config import settings

# Import schemas
from schemas.schemas import PredictionResponse, CortexMeshResponse, SubjectInfo

# Import services
from services.cortex_service import CortexService
from services.model_service import ModelService
from services.prediction_service import PredictionService
from services.data_loader import MATFileLoader

# Import utilities
from utilities.preprocessing import EEGPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
cortex_service = CortexService(settings.ANATOMY_DIR)
model_service = ModelService(settings.BASE_DIR, settings.DEVICE)
prediction_service = PredictionService()

# Predictions cache
PREDICTIONS_CACHE = {}


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
    try:
        cortex_data = cortex_service.load_cortex_mesh()
        
        return CortexMeshResponse(
            vertices=cortex_data['vertices'].tolist(),
            faces=cortex_data['faces'].tolist(),
            num_vertices=len(cortex_data['vertices']),
            num_faces=len(cortex_data['faces'])
        )
    except Exception as e:
        logger.error(f"Error in get_cortex_mesh: {str(e)}")
        raise


@app.get("/api/subjects", response_model=List[SubjectInfo])
async def get_subjects():
    """Get list of available subjects"""
    subjects = []
    
    if not settings.SOURCE_DIR.exists():
        return subjects
    
    for subject_dir in settings.SOURCE_DIR.iterdir():
        if subject_dir.is_dir() and subject_dir.name != "nmm_spikes":
            # Count data files
            data_files = list(subject_dir.glob("data*.mat"))
            
            # Check for predictions
            pred_files = list(subject_dir.glob("transformer_predictions_*.mat"))
            
            subjects.append(SubjectInfo(
                name=subject_dir.name,
                num_files=len(data_files),
                has_predictions=len(pred_files) > 0
            ))
    
    return subjects


@app.get("/api/predictions/{subject}", response_model=PredictionResponse)
async def get_predictions(
    subject: str,
    sample_idx: Optional[int] = Query(None, description="Specific sample index to return")
):
    """Get pre-computed predictions for a subject"""
    
    # Check cache
    cache_key = f"{subject}_{sample_idx}"
    if cache_key in PREDICTIONS_CACHE:
        logger.info(f"Returning cached predictions for {cache_key}")
        return PREDICTIONS_CACHE[cache_key]
    
    subject_dir = settings.SOURCE_DIR / subject
    
    if not subject_dir.exists():
        raise HTTPException(status_code=404, detail=f"Subject not found: {subject}")
    
    # Find prediction file
    pred_files = list(subject_dir.glob("transformer_predictions_*.mat"))
    
    if len(pred_files) == 0:
        raise HTTPException(status_code=404, detail=f"No predictions found for {subject}")
    
    # Load most recent prediction file
    pred_file = sorted(pred_files)[-1]
    
    try:
        logger.info(f"Loading predictions from: {pred_file}")
        mat_data = MATFileLoader.load_mat_file(pred_file)
        
        predictions = mat_data['all_out']
        
        # Reshape if needed
        predictions = prediction_service.reshape_predictions(predictions)
        
        # Extract file names
        file_names = MATFileLoader.extract_file_names(mat_data)
        
        # Get specific sample or all samples
        if sample_idx is not None:
            predictions, file_names = prediction_service.filter_sample(
                predictions, file_names, sample_idx
            )
        
        # Build response
        response = PredictionResponse(
            **prediction_service.build_response(predictions, file_names)
        )
        
        # Cache the response
        if settings.ENABLE_CACHE:
            PREDICTIONS_CACHE[cache_key] = response
        
        return response
        
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")


@app.post("/api/predict/{subject}")
async def predict_subject(
    subject: str,
    checkpoint: str = Query(settings.DEFAULT_CHECKPOINT, description="Path to model checkpoint"),
    normalize: bool = Query(True, description="Normalize input data")
):
    """
    Run inference on subject data
    This will process the raw EEG data and generate predictions
    """
    subject_dir = settings.SOURCE_DIR / subject
    
    if not subject_dir.exists():
        raise HTTPException(status_code=404, detail=f"Subject not found: {subject}")
    
    # Load model
    try:
        model_service.load_model(checkpoint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Find data files
    data_files = sorted(subject_dir.glob("data*.mat"), 
                       key=lambda x: int(x.stem.replace('data', '')))
    
    if len(data_files) == 0:
        raise HTTPException(status_code=404, detail=f"No data files found for {subject}")
    
    # Load and process data
    try:
        test_data = []
        file_names = []
        
        for data_file in data_files:
            mat_data = MATFileLoader.load_mat_file(data_file)
            
            try:
                data = MATFileLoader.extract_eeg_data(mat_data)
            except:
                continue
            
            # Preprocess
            data = EEGPreprocessor.preprocess(data, normalize=normalize)
            
            test_data.append(data)
            file_names.append(data_file.name)
        
        if len(test_data) == 0:
            raise HTTPException(status_code=400, detail="No valid data loaded")
        
        # Convert to tensor and run inference
        data_tensor = torch.from_numpy(np.array(test_data)).to(
            torch.device(settings.DEVICE), torch.float
        )
        
        predictions = model_service.predict(data_tensor)
        all_out = predictions.cpu().numpy()
        
        # Build response
        return PredictionResponse(
            **prediction_service.build_response(all_out, file_names)
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/api/upload-and-predict")
async def upload_and_predict(
    file: UploadFile = File(...),
    checkpoint: str = Query(settings.DEFAULT_CHECKPOINT, description="Path to model checkpoint"),
    normalize: bool = Query(True, description="Normalize input data")
):
    """
    Upload a MAT file with EEG data and get predictions
    Saves the file to VEP folder and generates predictions
    
    Expected MAT file format:
    - 'data' or 'eeg_data' or 'eeg': (time_points, channels) array
    - Should have 75 channels and any number of time points
    """
    
    # Verify file is a MAT file
    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="File must be a .mat file")
    
    # Create VEP folder if it doesn't exist
    vep_folder = settings.SOURCE_DIR / "VEP"
    vep_folder.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file to VEP folder
    uploaded_file_path = vep_folder / file.filename
    
    try:
        # Save to VEP folder
        with open(uploaded_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Saved uploaded file to: {uploaded_file_path}")
        
        # Load the MAT file
        mat_data = MATFileLoader.load_mat_file(uploaded_file_path)
        
        # Extract EEG data
        data = MATFileLoader.extract_eeg_data(mat_data)
        
        logger.info(f"Loaded uploaded file: {data.shape}")
        
        # Preprocess
        data = EEGPreprocessor.preprocess(data, normalize=normalize)
        
        # Load model
        logger.info("Loading model...")
        model_service.load_model(checkpoint)
        
        # Convert to tensor and run inference
        # Add batch dimension: (1, time_points, channels)
        data_tensor = torch.from_numpy(data[np.newaxis, :, :]).to(
            torch.device(settings.DEVICE), torch.float
        )
        
        logger.info(f"Running inference on data shape: {data_tensor.shape}")
        
        predictions = model_service.predict(data_tensor)
        all_out = predictions.cpu().numpy()
        
        logger.info(f"Predictions shape: {all_out.shape}")
        
        # Reshape if needed
        all_out = prediction_service.reshape_predictions(all_out)
        
        # Save predictions to VEP folder
        pred_filename = f"transformer_predictions_{file.filename.replace('.mat', '')}_uploaded.mat"
        pred_file_path = vep_folder / pred_filename
        
        savemat(str(pred_file_path), {
            'all_out': all_out,
            'file_names': [file.filename],
            'checkpoint': checkpoint,
            'num_samples': all_out.shape[0],
            'uploaded_file': str(uploaded_file_path)
        })
        
        logger.info(f"Saved predictions to: {pred_file_path}")
        
        # Build response
        return PredictionResponse(
            **prediction_service.build_response(all_out, [file.filename])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # Clean up uploaded file on error
        if uploaded_file_path.exists():
            try:
                os.unlink(uploaded_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(settings.DEVICE),
        "model_loaded": model_service.is_loaded(),
        "cortex_loaded": cortex_service._cache is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
