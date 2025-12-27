# EEG Source Localization Visualization App

A modern web application for visualizing EEG source localization predictions from the Transformer model. Features interactive 3D cortex visualization with real-time activation mapping.

## Features

- 🧠 **3D Cortex Visualization**: Interactive 3D rendering of brain cortex with source activations
- 🎨 **Hot Colormap**: Intuitive color mapping from black (inactive) to red/yellow (active)
- 🎛️ **Interactive Controls**: 
  - Navigate between samples
  - Adjust activation threshold
  - Toggle normalization
  - Rotate, zoom, and pan the 3D view
- 📊 **Real-time Statistics**: View activation statistics for each sample
- ⚡ **Fast API Backend**: Efficient data serving with caching

## Architecture

### Backend (FastAPI)
- Serves cortex mesh data
- Loads and serves model predictions
- Supports on-demand inference
- Caches data for performance

### Frontend (React + Three.js)
- Modern React UI with hooks
- Three.js for 3D rendering via React Three Fiber
- Responsive design
- Real-time interaction

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
cd visualization_app/backend

# Install Python dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

```bash
cd visualization_app/frontend

# Install Node dependencies
npm install

# Start development server
npm start
```

The frontend will start on `http://localhost:3000`

## Usage

1. **Start the Backend**: Run the FastAPI server first
2. **Start the Frontend**: Launch the React development server
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Select Subject**: Choose a subject from the dropdown (e.g., VEP)
5. **Explore**: Use the controls to navigate samples and adjust visualization

### Controls

- **Subject Selection**: Choose which subject's predictions to visualize
- **Sample Navigation**: Use arrows or slider to move between samples
- **Threshold**: Filter out weak activations (0.0 - 1.0)
- **Normalize**: Toggle activation normalization
- **3D View**:
  - Left mouse drag: Rotate
  - Mouse wheel: Zoom
  - Right mouse drag: Pan

## API Endpoints

### GET `/api/cortex-mesh`
Returns the cortex mesh geometry (vertices and faces)

### GET `/api/subjects`
Lists available subjects with their data

### GET `/api/predictions/{subject}`
Get predictions for a subject
- Query param: `sample_idx` (optional) - specific sample index

### POST `/api/predict/{subject}`
Run inference on subject data
- Query params: 
  - `checkpoint`: path to model checkpoint
  - `normalize`: whether to normalize input

### GET `/api/health`
Health check endpoint

## Project Structure

```
visualization_app/
├── backend/
│   ├── app.py              # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── src/
│   │   ├── api/
│   │   │   └── api.js      # API client
│   │   ├── components/
│   │   │   ├── CortexVisualization.js    # 3D visualization
│   │   │   ├── ControlPanel.js           # Control UI
│   │   │   └── StatsPanel.js             # Statistics display
│   │   ├── App.js          # Main application
│   │   ├── App.css         # Main styles
│   │   └── index.js        # Entry point
│   └── package.json        # Node dependencies
└── README.md               # This file
```

## Configuration

### Backend Configuration

Edit `visualization_app/backend/app.py`:

```python
# Paths
BASE_DIR = Path(__file__).parent.parent.parent
ANATOMY_DIR = BASE_DIR / "anatomy"
SOURCE_DIR = BASE_DIR / "source"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
```

### Frontend Configuration

Create `.env` file in `frontend/` directory:

```
REACT_APP_API_URL=http://localhost:8000
```

## Data Requirements

The application expects the following data structure:

```
project_root/
├── anatomy/
│   └── fs_cortex_20k.mat          # Cortex mesh (pos, tri)
├── source/
│   └── VEP/                        # Subject folder
│       ├── data*.mat               # EEG data files
│       └── transformer_predictions_*.mat  # Predictions
└── checkpoints/
    └── best_model.pt               # Trained model
```

### MAT File Format

**Cortex Mesh** (`fs_cortex_20k.mat`):
- `pos`: (N, 3) array of vertex positions
- `tri`: (M, 3) array of triangle face indices

**Predictions** (`transformer_predictions_*.mat`):
- `all_out`: (num_samples, num_sources) array of predictions
- `file_names`: list of source file names (optional)

**EEG Data** (`data*.mat`):
- `data` or `eeg_data`: (time_points, channels) array

## Troubleshooting

### Backend Issues

**Port already in use**:
```bash
# Change port in app.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**CORS errors**:
- Check that backend is running
- Verify `REACT_APP_API_URL` in frontend `.env`

**Model not loading**:
- Verify checkpoint path exists
- Check model configuration matches

### Frontend Issues

**Module not found**:
```bash
cd visualization_app/frontend
rm -rf node_modules package-lock.json
npm install
```

**Blank screen**:
- Check browser console for errors
- Verify backend is running and accessible
- Check that data files exist

**3D rendering issues**:
- Update graphics drivers
- Try a different browser (Chrome recommended)
- Check WebGL support: https://get.webgl.org/

## Performance Tips

1. **Use smaller batch sizes** for inference
2. **Cache predictions** by running `eval_real.py` first
3. **Reduce mesh resolution** if rendering is slow
4. **Close other applications** to free GPU memory

## Development

### Adding New Features

1. **Backend**: Add endpoints in `app.py`
2. **Frontend**: Add API calls in `src/api/api.js`
3. **UI**: Create new components in `src/components/`

### Building for Production

```bash
# Frontend
cd visualization_app/frontend
npm run build

# Serve with backend
# Copy build/ folder to backend/static/
```

## License

This project is part of the EEG Source Localization research project.

## Acknowledgments

- Three.js for 3D rendering
- React Three Fiber for React integration
- FastAPI for the backend framework

