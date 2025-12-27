"""
FastAPI backend for EEG Source Localization Visualization
Serves model predictions and cortex mesh data
"""
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from scipy.io import loadmat, savemat
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import tempfile
import shutil

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.transformer_model import EEGSourceTransformerV2

app = FastAPI(title="EEG Visualization API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
CORTEX_DATA = None
PREDICTIONS_CACHE = {}
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
ANATOMY_DIR = BASE_DIR / "anatomy"
SOURCE_DIR = BASE_DIR / "source"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


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


def load_cortex_mesh():
    """Load cortex mesh data"""
    global CORTEX_DATA
    
    print("CORTEX_DATA is", CORTEX_DATA)
    if CORTEX_DATA is not None:
        return CORTEX_DATA
    
    cortex_path = ANATOMY_DIR / "fs_cortex_20k.mat"
    
    if not cortex_path.exists():
        raise HTTPException(status_code=404, detail="Cortex mesh file not found")
    
    try:

        print("Loading cortex mesh...")
        mat_data = loadmat(str(cortex_path))

        print("mat_data is", mat_data)
        # Extract vertices and faces
        vertices = mat_data['pos']  # Should be (N, 3)
        faces = mat_data['tri'] - 1  # MATLAB uses 1-based indexing
        
        CORTEX_DATA = {
            'vertices': vertices,
            'faces': faces
        }
        
        print(f"Loaded cortex mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        return CORTEX_DATA
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cortex mesh: {str(e)}")


def load_model(checkpoint_path: str):
    """Load trained model"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    print("Loading model...")
    print("checkpoint_path input:", checkpoint_path)
    
    # Convert to Path and resolve relative to BASE_DIR
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path
    
    print("Resolved checkpoint_path:", checkpoint_path)
    print("File exists:", checkpoint_path.exists())

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
        
        # Get model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = EEGSourceTransformerV2(
                eeg_channels=config['EEG_CHANNELS'],
                source_regions=config['SOURCE_REGIONS'],
                d_model=config['D_MODEL'],
                nhead=config['NHEAD'],
                num_layers=config['NUM_LAYERS'],
                dim_feedforward=config['DIM_FEEDFORWARD'],
                dropout=config['DROPOUT']
            ).to(DEVICE)
        else:
            # Use default config
            model = EEGSourceTransformerV2(
                eeg_channels=75,
                source_regions=994,
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.1
            ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        MODEL = model
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        return MODEL
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EEG Source Localization Visualization API",
        "version": "1.0.0",
        "endpoints": {
            "cortex_mesh": "/api/cortex-mesh",
            "subjects": "/api/subjects",
            "predictions": "/api/predictions/{subject}",
            "predict": "/api/predict/{subject}"
        }
    }


@app.get("/api/cortex-mesh", response_model=CortexMeshResponse)
async def get_cortex_mesh():
    """Get cortex mesh data for visualization"""
    try:
        cortex_data = load_cortex_mesh()
        
        vertices = cortex_data['vertices'].tolist()
        faces = cortex_data['faces'].tolist()
        
        return CortexMeshResponse(
            vertices=vertices,
            faces=faces,
            num_vertices=len(vertices),
            num_faces=len(faces)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subjects", response_model=List[SubjectInfo])
async def get_subjects():
    """Get list of available subjects"""
    subjects = []
    
    if not SOURCE_DIR.exists():
        return subjects
    
    for subject_dir in SOURCE_DIR.iterdir():
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
        return PREDICTIONS_CACHE[cache_key]
    
    subject_dir = SOURCE_DIR / subject
    
    if not subject_dir.exists():
        raise HTTPException(status_code=404, detail=f"Subject not found: {subject}")
    
    # Find prediction file
    pred_files = list(subject_dir.glob("transformer_predictions_*.mat"))
    
    if len(pred_files) == 0:
        raise HTTPException(status_code=404, detail=f"No predictions found for {subject}")
    
    # Load most recent prediction file
    pred_file = sorted(pred_files)[-1]
    
    try:
        mat_data = loadmat(str(pred_file))
        
        predictions = mat_data['all_out']  # Shape: (batch, time, sources) or (num_samples, num_sources)
        
        # Handle different prediction formats
        if len(predictions.shape) == 3:
            # Format: (batch, time_points, sources)
            # Treat each time point as a separate "sample" for visualization
            batch_size, time_points, num_sources = predictions.shape
            # Reshape to (batch * time_points, num_sources)
            predictions = predictions.reshape(-1, num_sources)
            print(f"Reshaped 3D predictions from {(batch_size, time_points, num_sources)} to {predictions.shape}")
        elif len(predictions.shape) == 2:
            # Format: (num_samples, num_sources) - already correct
            pass
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        # Process file names
        file_names_raw = mat_data.get('file_names', [])
        if len(file_names_raw) > 0:
            file_names = []
            for f in file_names_raw:
                if isinstance(f, np.ndarray):
                    # MATLAB cell array
                    if f.size > 0:
                        file_names.append(str(f.flat[0]) if hasattr(f, 'flat') else str(f))
                else:
                    file_names.append(str(f))
        else:
            file_names = []
        
        # Get specific sample or all samples
        if sample_idx is not None:
            if sample_idx < 0 or sample_idx >= predictions.shape[0]:
                raise HTTPException(status_code=400, detail="Invalid sample index")
            predictions = predictions[sample_idx:sample_idx+1]
            if file_names and sample_idx < len(file_names):
                file_names = [file_names[sample_idx]]
        
        # Calculate statistics
        stats = {
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions))
        }
        
        response = PredictionResponse(
            predictions=predictions.tolist(),
            num_samples=predictions.shape[0],
            num_sources=predictions.shape[1],
            file_names=file_names if file_names else None,
            statistics=stats
        )
        
        # Cache the response
        PREDICTIONS_CACHE[cache_key] = response
        
        return response
        
    except Exception as e:
        import traceback
        error_details = f"Error loading predictions: {str(e)}\n{traceback.format_exc()}"
        print(error_details)  # Log to console
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")


@app.post("/api/predict/{subject}")
async def predict_subject(
    subject: str,
    checkpoint: str = Query("checkpoints/best_model.pt", description="Path to model checkpoint"),
    normalize: bool = Query(True, description="Normalize input data")
):
    """
    Run inference on subject data
    This will process the raw EEG data and generate predictions
    """
    subject_dir = SOURCE_DIR / subject
    
    if not subject_dir.exists():
        raise HTTPException(status_code=404, detail=f"Subject not found: {subject}")
    
    # Load model
    try:
        model = load_model(checkpoint)
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
            mat_data = loadmat(str(data_file))
            
            # Try different field names
            data = None
            for key in ['data', 'eeg_data', 'eeg', 'EEG']:
                if key in mat_data:
                    data = mat_data[key]
                    break
            
            if data is None:
                continue
            
            # Ensure correct shape (time, channels)
            if data.shape[1] == 75 and data.shape[0] != 75:
                pass  # Already correct
            elif data.shape[0] == 75 and data.shape[1] != 75:
                data = data.T
            else:
                continue
            
            # Ensure 500 time points
            if data.shape[0] != 500:
                if data.shape[0] > 500:
                    data = data[:500, :]
                else:
                    padded = np.zeros((500, data.shape[1]))
                    padded[:data.shape[0], :] = data
                    data = padded
            
            # Preprocess
            data = data - np.mean(data, axis=0, keepdims=True)
            data = data - np.mean(data, axis=1, keepdims=True)
            
            if normalize:
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val
            
            test_data.append(data)
            file_names.append(data_file.name)
        
        if len(test_data) == 0:
            raise HTTPException(status_code=400, detail="No valid data loaded")
        
        # Convert to tensor and run inference
        data_tensor = torch.from_numpy(np.array(test_data)).to(DEVICE, torch.float)
        
        with torch.no_grad():
            predictions = model(data_tensor)
            all_out = predictions.cpu().numpy()
        
        # Calculate statistics
        stats = {
            "min": float(np.min(all_out)),
            "max": float(np.max(all_out)),
            "mean": float(np.mean(all_out)),
            "std": float(np.std(all_out))
        }
        
        return PredictionResponse(
            predictions=all_out.tolist(),
            num_samples=all_out.shape[0],
            num_sources=all_out.shape[1],
            file_names=file_names,
            statistics=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/api/upload-and-predict")
async def upload_and_predict(
    file: UploadFile = File(...),
    checkpoint: str = Query("checkpoints\best_model.pt", description="Path to model checkpoint"),
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
    vep_folder = SOURCE_DIR / "VEP"
    vep_folder.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file to VEP folder
    uploaded_file_path = vep_folder / file.filename
    temp_file = None
    
    try:
        # Save to VEP folder
        with open(uploaded_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        print(f"Saved uploaded file to: {uploaded_file_path}")
        temp_file_path = uploaded_file_path
        
        # Load the MAT file
        mat_data = loadmat(temp_file_path)
        
        # Try different field names for EEG data
        data = None
        for key in ['data', 'eeg_data', 'eeg', 'EEG']:
            if key in mat_data:
                data = mat_data[key]
                break
        
        if data is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not find EEG data. Expected one of: 'data', 'eeg_data', 'eeg', 'EEG'. Found: {list(mat_data.keys())}"
            )
        
        # Ensure correct shape (time, channels)
        if data.shape[1] == 75 and data.shape[0] != 75:
            pass  # Already correct
        elif data.shape[0] == 75 and data.shape[1] != 75:
            data = data.T  # Transpose
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Expected data with 75 channels. Got shape: {data.shape}"
            )
        
        print(f"Loaded uploaded file: {data.shape}")
        
        # Ensure 500 time points (or use what's available)
        original_time_points = data.shape[0]
        print("original_time_points is", original_time_points)
        if data.shape[0] != 500:
            if data.shape[0] > 500:
                data = data[:500, :]
                print(f"Truncated from {original_time_points} to 500 time points")
            else:
                # Pad with zeros
                padded = np.zeros((500, data.shape[1]))
                padded[:data.shape[0], :] = data
                data = padded
                print(f"Padded from {original_time_points} to 500 time points")
        
        # Preprocess
        print("Preprocessing data...")
        data = data - np.mean(data, axis=0, keepdims=True)
        data = data - np.mean(data, axis=1, keepdims=True)
        print("Data preprocessed")

        print("Normalizing data...")
        if normalize:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
        print("Data normalized")
        # Load model
        print("Loading model...")
        model = load_model(checkpoint)
        print("Model loaded")
        # Convert to tensor and run inference
        # Add batch dimension: (1, time_points, channels)
        data_tensor = torch.from_numpy(data[np.newaxis, :, :]).to(DEVICE, torch.float)
        
        print(f"Running inference on data shape: {data_tensor.shape}")
        
        with torch.no_grad():
            predictions = model(data_tensor)
            all_out = predictions.cpu().numpy()
        
        print(f"Predictions shape: {all_out.shape}")
        
        # Handle different output shapes
        if len(all_out.shape) == 3:
            # Format: (batch, time_points, sources)
            batch_size, time_points, num_sources = all_out.shape
            # Reshape to (time_points, num_sources) for visualization
            all_out = all_out.reshape(-1, num_sources)
            print(f"Reshaped predictions to: {all_out.shape}")
        
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
        
        print(f"Saved predictions to: {pred_file_path}")
        
        # Calculate statistics
        stats = {
            "min": float(np.min(all_out)),
            "max": float(np.max(all_out)),
            "mean": float(np.mean(all_out)),
            "std": float(np.std(all_out))
        }
        
        return PredictionResponse(
            predictions=all_out.tolist(),
            num_samples=all_out.shape[0],
            num_sources=all_out.shape[1],
            file_names=[file.filename],
            statistics=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        print(error_details)
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
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "cortex_loaded": CORTEX_DATA is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

