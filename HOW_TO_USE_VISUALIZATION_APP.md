# How to Use the EEG Visualization App

## 🎯 Overview

I've created a complete web application for visualizing your EEG transformer model predictions! This app provides an interactive 3D visualization of brain cortex activations.

## 📦 What's Included

### Complete Application Structure
```
visualization_app/
├── backend/                    # FastAPI server
│   ├── app.py                 # Main server code
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React web app
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── api/              # API client
│   │   └── App.js            # Main app
│   ├── public/
│   └── package.json          # Node dependencies
├── start_app.bat              # One-click start (Windows)
├── start_backend.bat          # Start backend only
├── start_frontend.bat         # Start frontend only
├── check_setup.py             # Verify installation
├── test_backend.py            # Test API endpoints
├── README.md                  # Full documentation
├── QUICK_START.md             # Quick start guide
├── SETUP_GUIDE.md             # Detailed setup
└── FEATURES.md                # Feature overview
```

## 🚀 Quick Start (3 Steps)

### Step 1: Verify Setup
```bash
python visualization_app\check_setup.py
```

This checks if all dependencies and data files are ready.

### Step 2: Start the Application

**Easiest way (Windows):**
```bash
visualization_app\start_app.bat
```

This starts both backend and frontend automatically!

**Or start manually:**

Terminal 1 - Backend:
```bash
cd visualization_app\backend
python app.py
```

Terminal 2 - Frontend:
```bash
cd visualization_app\frontend
npm install  # First time only
npm start
```

### Step 3: Open Browser

The app will automatically open at: **http://localhost:3000**

## 🎮 Using the App

### Main Interface

The app has three main sections:

1. **Left Panel**: Controls and statistics
2. **Center**: 3D interactive cortex visualization
3. **Bottom**: Navigation hints

### Controls Explained

#### 1. Subject Selection
- **Dropdown menu** at the top
- Shows available subjects (e.g., "VEP")
- Displays number of data files
- Select to load that subject's predictions

#### 2. Sample Navigation
- **Current sample indicator**: Shows "1 / 50" (current/total)
- **Previous/Next buttons**: Navigate one sample at a time
- **Slider**: Jump to any sample quickly
- Predictions update instantly when you change samples

#### 3. Threshold Control
- **Slider**: Ranges from 0.0 to 1.0
- **Purpose**: Filter out weak activations
- **Lower values**: Show more regions (including weak activations)
- **Higher values**: Show only strong activations
- **Recommended**: Start at 0.1, adjust as needed
- Updates in real-time as you move the slider

#### 4. Normalization Toggle
- **Checkbox**: Enable/disable normalization
- **When ON**: Scales all activations to [0, 1] range
- **When OFF**: Uses raw activation values
- **Use case**: Enable for consistent visualization across samples

#### 5. 3D View Controls
- **Rotate**: Left-click and drag
- **Zoom**: Mouse wheel (scroll up/down)
- **Pan**: Right-click and drag
- **Reset**: Refresh the page (auto-reset planned)

### Understanding the Visualization

#### Color Coding
The cortex is colored based on activation levels:

- **Dark gray/blue**: No activation (below threshold)
- **Dark red**: Low activation
- **Red**: Moderate activation
- **Orange**: High activation
- **Yellow/White**: Very high activation

#### Statistics Panel
Shows real-time statistics for the current sample:

- **Total Samples**: Number of predictions available
- **Source Regions**: Number of brain regions (994)
- **Current Sample**: Which sample you're viewing
- **File Name**: Original data file (if available)
- **Min/Max/Mean/Std**: Activation statistics

## 📊 Typical Workflow

### Workflow 1: Exploring Predictions

1. **Start the app**
2. **Select subject** (e.g., VEP)
3. **Browse samples** using the slider
4. **Adjust threshold** to focus on strong activations
5. **Rotate the view** to see all regions
6. **Check statistics** to understand the data

### Workflow 2: Quality Control

1. **Load predictions**
2. **Go through each sample** systematically
3. **Look for anomalies** (unusual patterns)
4. **Check statistics** for outliers
5. **Verify activations** look reasonable

### Workflow 3: Presentation

1. **Find interesting samples**
2. **Adjust threshold** for best visualization
3. **Set optimal view angle**
4. **Take screenshots** (browser screenshot tool)
5. **Use for papers/presentations**

## 🔧 Advanced Features

### Running Real-time Inference

Instead of using pre-computed predictions, you can run inference on-demand:

1. Ensure model checkpoint exists: `checkpoints/best_model.pt`
2. Place raw EEG data in: `source/{subject}/data*.mat`
3. The app will automatically offer to run inference
4. Or use the API endpoint: `POST /api/predict/{subject}`

### Using Multiple Subjects

To add more subjects:

1. Create folder: `source/NewSubject/`
2. Add data files: `data1.mat`, `data2.mat`, etc.
3. Generate predictions:
   ```bash
   python eval_real.py --subjects NewSubject
   ```
4. Refresh the app - new subject appears in dropdown

### API Access

The backend provides a REST API for programmatic access:

**API Documentation**: http://localhost:8000/docs (when running)

Example endpoints:
- `GET /api/cortex-mesh` - Get brain mesh
- `GET /api/predictions/VEP` - Get VEP predictions
- `GET /api/subjects` - List all subjects

## 🐛 Troubleshooting

### Problem: App won't start

**Check 1: Backend**
```bash
cd visualization_app\backend
python app.py
```
- Should see: "Uvicorn running on http://0.0.0.0:8000"
- If error, check Python packages: `pip install -r requirements.txt`

**Check 2: Frontend**
```bash
cd visualization_app\frontend
npm install
npm start
```
- Should open browser automatically
- If error, check Node.js is installed: `node --version`

### Problem: No data showing

**Solution 1: Check predictions exist**
```bash
dir source\VEP\transformer_predictions_*.mat
```
- If not found, generate predictions:
```bash
python eval_real.py --subjects VEP
```

**Solution 2: Check cortex mesh**
```bash
dir anatomy\fs_cortex_20k.mat
```
- Should exist (required file)

### Problem: 3D view is blank

**Solutions:**
1. Check browser console (F12) for errors
2. Verify WebGL support: https://get.webgl.org/
3. Try different browser (Chrome recommended)
4. Update graphics drivers
5. Check that data loaded (see statistics panel)

### Problem: Slow performance

**Solutions:**
1. Increase threshold (shows fewer vertices)
2. Close other applications
3. Use hardware acceleration in browser
4. Reduce browser zoom level
5. Use Chrome for best performance

### Problem: Can't connect to backend

**Solutions:**
1. Verify backend is running: http://localhost:8000
2. Check `.env` file in frontend:
   ```
   REACT_APP_API_URL=http://localhost:8000
   ```
3. Restart both servers
4. Check firewall settings

## 💡 Tips & Best Practices

### Visualization Tips

1. **Start with low threshold (0.1)** to see overall pattern
2. **Increase threshold gradually** to focus on strong regions
3. **Use normalization** when comparing different samples
4. **Rotate the view** to see both hemispheres
5. **Check statistics** to understand the data range

### Performance Tips

1. **Pre-compute predictions** using `eval_real.py`
2. **Close unused browser tabs**
3. **Use Chrome** for best WebGL performance
4. **Clear cache** if experiencing issues
5. **Restart servers** if they become slow

### Analysis Tips

1. **Compare multiple samples** to find patterns
2. **Note the file names** of interesting samples
3. **Take screenshots** for documentation
4. **Check statistics** for outliers
5. **Adjust view angle** for different perspectives

## 📚 Documentation

### Quick References
- **Quick Start**: `visualization_app/QUICK_START.md`
- **Setup Guide**: `visualization_app/SETUP_GUIDE.md`
- **Features**: `visualization_app/FEATURES.md`
- **Full README**: `visualization_app/README.md`

### Testing
- **Check setup**: `python visualization_app/check_setup.py`
- **Test backend**: `python visualization_app/test_backend.py`
- **API docs**: http://localhost:8000/docs

## 🎓 Understanding the Technology

### Backend (FastAPI)
- Modern Python web framework
- Fast and efficient
- Automatic API documentation
- Handles data loading and serving

### Frontend (React + Three.js)
- React for UI components
- Three.js for 3D rendering
- WebGL for hardware acceleration
- Responsive and interactive

### Data Flow
```
MAT Files → Backend (FastAPI) → API → Frontend (React) → Three.js → 3D View
```

## ✅ Checklist

Before using the app, ensure:

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Cortex mesh exists (`anatomy/fs_cortex_20k.mat`)
- [ ] Predictions exist (`source/VEP/transformer_predictions_*.mat`)
- [ ] Backend running (port 8000)
- [ ] Frontend running (port 3000)
- [ ] Browser opened to http://localhost:3000

## 🎉 You're Ready!

Your visualization app is complete and ready to use. It provides:

✅ Interactive 3D brain visualization
✅ Real-time control over display parameters
✅ Statistics and analysis tools
✅ Support for multiple subjects
✅ Professional, modern interface
✅ Fast and responsive performance

## 🆘 Getting Help

If you encounter issues:

1. **Check documentation** in `visualization_app/` folder
2. **Run setup check**: `python visualization_app/check_setup.py`
3. **Test backend**: `python visualization_app/test_backend.py`
4. **Check browser console** (F12) for errors
5. **Review terminal output** for error messages

## 🚀 Next Steps

1. ✅ App is built and ready
2. → Run `check_setup.py` to verify installation
3. → Start the app with `start_app.bat`
4. → Open http://localhost:3000
5. → Select VEP subject
6. → Explore your predictions!

Enjoy visualizing your EEG data! 🧠✨

