# EEG Source Localization Visualization App

## 📋 Overview

I've created a complete web application for visualizing your EEG transformer model predictions! This app provides an interactive 3D visualization of brain cortex activations predicted by your model.

## 🎯 Features

### Backend (FastAPI)
- ✅ Serves cortex mesh geometry (20k vertices)
- ✅ Loads and serves model predictions from MAT files
- ✅ Supports multiple subjects (VEP, etc.)
- ✅ Real-time inference capability
- ✅ Caching for performance
- ✅ RESTful API with automatic documentation

### Frontend (React + Three.js)
- ✅ Interactive 3D cortex visualization
- ✅ Hot colormap (black → red → orange → yellow)
- ✅ Subject selection dropdown
- ✅ Sample navigation (slider + buttons)
- ✅ Adjustable activation threshold
- ✅ Normalization toggle
- ✅ Real-time statistics display
- ✅ Smooth 3D controls (rotate, zoom, pan)
- ✅ Modern, responsive UI

## 📁 Project Structure

```
visualization_app/
├── backend/
│   ├── app.py                    # FastAPI server
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── api/
│   │   │   └── api.js           # API client
│   │   ├── components/
│   │   │   ├── CortexVisualization.js    # 3D rendering
│   │   │   ├── CortexVisualization.css
│   │   │   ├── ControlPanel.js           # UI controls
│   │   │   ├── ControlPanel.css
│   │   │   ├── StatsPanel.js             # Statistics
│   │   │   └── StatsPanel.css
│   │   ├── App.js               # Main app
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── .env                      # Configuration
├── start_app.bat                 # Start both servers (Windows)
├── start_backend.bat             # Start backend only
├── start_frontend.bat            # Start frontend only
├── test_backend.py               # Backend test suite
├── README.md                     # Full documentation
├── QUICK_START.md                # Quick start guide
└── SETUP_GUIDE.md                # Detailed setup guide
```

## 🚀 Quick Start

### Option 1: One-Click Start (Windows)

```bash
# From project root
visualization_app\start_app.bat
```

This starts both backend and frontend automatically!

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd visualization_app/backend
pip install -r requirements.txt
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd visualization_app/frontend
npm install
npm start
```

Then open http://localhost:3000 in your browser.

## 📚 Documentation

- **[QUICK_START.md](visualization_app/QUICK_START.md)** - Get started in 3 steps
- **[SETUP_GUIDE.md](visualization_app/SETUP_GUIDE.md)** - Detailed setup instructions
- **[README.md](visualization_app/README.md)** - Complete documentation

## 🎨 Screenshots

### Main Interface
- Left panel: Controls (subject selection, sample navigation, threshold)
- Center: 3D interactive cortex with color-coded activations
- Statistics panel: Real-time activation statistics

### Color Scheme
- **Dark gray/blue**: No activation (below threshold)
- **Red**: Low activation
- **Orange**: Medium activation  
- **Yellow**: High activation

## 🔧 API Endpoints

### GET `/api/cortex-mesh`
Returns cortex mesh geometry (vertices and faces)

### GET `/api/subjects`
Lists available subjects with metadata

### GET `/api/predictions/{subject}`
Get predictions for a subject
- Optional query param: `sample_idx`

### POST `/api/predict/{subject}`
Run real-time inference on subject data

### GET `/api/health`
Health check endpoint

Full API docs available at: http://localhost:8000/docs (when running)

## 💡 Usage Tips

1. **Navigate Samples**: Use arrow buttons or slider to move between predictions
2. **Adjust Threshold**: Move slider to filter weak activations (0.0 - 1.0)
3. **Normalize**: Toggle to normalize activations to [0, 1] range
4. **3D Controls**:
   - Left mouse drag: Rotate cortex
   - Mouse wheel: Zoom in/out
   - Right mouse drag: Pan view
5. **View Statistics**: Check the stats panel for min/max/mean/std

## 🧪 Testing

Test the backend:
```bash
python visualization_app/test_backend.py
```

This will verify all API endpoints are working correctly.

## 📊 Data Requirements

The app expects:

1. **Cortex Mesh**: `anatomy/fs_cortex_20k.mat`
   - Contains: `pos` (vertices), `tri` (faces)

2. **Predictions**: `source/{subject}/transformer_predictions_*.mat`
   - Contains: `all_out` (predictions), `file_names` (optional)

3. **Model** (optional): `checkpoints/best_model.pt`
   - Only needed for real-time inference

Generate predictions if needed:
```bash
python eval_real.py --subjects VEP --checkpoint checkpoints/best_model.pt
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **NumPy/SciPy**: Data processing
- **PyTorch**: Model inference

### Frontend
- **React**: UI framework
- **Three.js**: 3D rendering
- **React Three Fiber**: React bindings for Three.js
- **Axios**: HTTP client

## 🎯 Key Features Explained

### 3D Visualization
- Uses WebGL for hardware-accelerated rendering
- Supports meshes with 20k+ vertices
- Real-time color mapping based on activations
- Smooth camera controls with damping

### Performance
- Backend caching for fast repeated access
- Efficient geometry processing
- Batch inference support
- Progressive loading

### Interactivity
- Real-time threshold adjustment
- Sample navigation
- Multiple subjects support
- Responsive design

## 🔄 Workflow

1. **Load Data**: App loads cortex mesh and predictions
2. **Select Subject**: Choose from available subjects (e.g., VEP)
3. **Navigate**: Browse through prediction samples
4. **Adjust**: Fine-tune threshold and normalization
5. **Explore**: Interact with 3D view to examine activations
6. **Analyze**: Review statistics for each sample

## 🐛 Troubleshooting

### Backend Issues
- **Port in use**: Change port in `app.py`
- **Module not found**: Run `pip install -r requirements.txt`
- **Data not found**: Check file paths in `app.py`

### Frontend Issues
- **Won't start**: Delete `node_modules`, run `npm install`
- **Can't connect**: Check `.env` has correct API URL
- **Blank screen**: Check browser console (F12) for errors

### 3D Rendering Issues
- **No display**: Verify WebGL support at https://get.webgl.org/
- **Slow performance**: Increase threshold, reduce mesh resolution
- **Colors wrong**: Check normalization setting

## 📈 Future Enhancements

Potential additions:
- [ ] Multiple view angles (left, right, top, bottom)
- [ ] Animation of temporal activations
- [ ] Export visualizations as images/videos
- [ ] Comparison view (multiple samples side-by-side)
- [ ] Region of interest (ROI) selection
- [ ] Custom colormap selection
- [ ] Overlay anatomical labels
- [ ] Real-time EEG data streaming

## 🤝 Contributing

To modify the app:

1. **Backend**: Edit `visualization_app/backend/app.py`
2. **Frontend**: Edit files in `visualization_app/frontend/src/`
3. **Styling**: Modify `.css` files
4. **API**: Add endpoints in `app.py` and update `api.js`

## 📝 Notes

- The app is designed for the VEP dataset but works with any subject
- Predictions must be pre-computed or model checkpoint available
- Supports both normalized and raw activation values
- Threshold filtering happens in real-time on the frontend
- All data stays local (no external servers)

## 🎓 Credits

This visualization app integrates with your EEG Source Localization Transformer project and visualizes predictions similar to the MATLAB `visualize_result.m` script, but with modern web technologies for better interactivity.

## ✅ Checklist

Before using:
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Dependencies installed (backend and frontend)
- [ ] Data files present (cortex mesh, predictions)
- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000

## 🚀 Ready to Use!

Your visualization app is complete and ready to use. Start both servers and open http://localhost:3000 to begin exploring your EEG predictions in 3D!

For detailed instructions, see:
- Quick start: `visualization_app/QUICK_START.md`
- Full setup: `visualization_app/SETUP_GUIDE.md`
- API docs: http://localhost:8000/docs (when running)

