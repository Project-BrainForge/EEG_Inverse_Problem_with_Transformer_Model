# Complete Setup Guide

This guide will walk you through setting up the EEG Source Localization Visualization App from scratch.

## Prerequisites

### Required Software

1. **Python 3.8 or higher**
   - Check: `python --version`
   - Download: https://www.python.org/downloads/

2. **Node.js 16 or higher**
   - Check: `node --version`
   - Download: https://nodejs.org/

3. **npm (comes with Node.js)**
   - Check: `npm --version`

### Required Data Files

Ensure these files exist in your project:

```
D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\
├── anatomy/
│   └── fs_cortex_20k.mat          # Cortex mesh geometry
├── source/
│   └── VEP/                        # Subject data
│       ├── data3.mat               # EEG data (optional)
│       └── transformer_predictions_best_model.mat  # Model predictions
├── checkpoints/
│   └── best_model.pt               # Trained model (optional, for inference)
└── models/
    └── transformer_model.py        # Model definition
```

## Installation Steps

### Step 1: Install Backend Dependencies

```bash
# Navigate to project root
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model

# Activate your virtual environment (if using one)
venv\Scripts\activate

# Install backend requirements
pip install fastapi uvicorn numpy scipy torch pydantic python-multipart

# Or use the requirements file
pip install -r visualization_app/backend/requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd visualization_app/frontend

# Install Node packages (this may take a few minutes)
npm install
```

This will install:
- React
- Three.js
- React Three Fiber
- Axios
- And other dependencies

### Step 3: Configure Environment

Create `.env` file in `visualization_app/frontend/`:

```bash
# Create .env file
echo REACT_APP_API_URL=http://localhost:8000 > .env
```

Or manually create `visualization_app/frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8000
```

## Running the Application

### Method 1: Using Batch Files (Windows - Easiest)

**Option A: Start both at once**
```bash
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model
visualization_app\start_app.bat
```

**Option B: Start separately**
```bash
# Terminal 1: Backend
visualization_app\start_backend.bat

# Terminal 2: Frontend
visualization_app\start_frontend.bat
```

### Method 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model
python visualization_app/backend/app.py
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

**Terminal 2 - Frontend:**
```bash
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\visualization_app\frontend
npm start
```

Wait for browser to open at `http://localhost:3000`

## Verification

### Test Backend

1. Open browser to http://localhost:8000
2. You should see API information
3. Try http://localhost:8000/api/health
4. Or run the test script:
   ```bash
   python visualization_app/test_backend.py
   ```

### Test Frontend

1. Browser should open automatically to http://localhost:3000
2. You should see the EEG Visualization interface
3. Select "VEP" from the subject dropdown
4. The 3D cortex should load with colored activations

## Common Issues and Solutions

### Issue 1: Backend won't start

**Error: "Module not found"**
```bash
# Solution: Install missing packages
pip install fastapi uvicorn numpy scipy torch pydantic
```

**Error: "Port 8000 already in use"**
```bash
# Solution: Kill the process or change port
# Edit visualization_app/backend/app.py, line at bottom:
uvicorn.run(app, host="0.0.0.0", port=8001)

# Then update frontend .env:
REACT_APP_API_URL=http://localhost:8001
```

**Error: "Checkpoint not found"**
```bash
# Solution: Check checkpoint path
# The app will still work for viewing existing predictions
# Only needed for on-demand inference
```

### Issue 2: Frontend won't start

**Error: "Node.js not found"**
```bash
# Solution: Install Node.js from https://nodejs.org/
# Restart terminal after installation
```

**Error: "npm install fails"**
```bash
# Solution: Clear cache and retry
cd visualization_app/frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Error: "Port 3000 already in use"**
```bash
# Solution: Use different port
set PORT=3001
npm start
```

### Issue 3: Can't connect to backend

**Error: "Network Error" or "Failed to fetch"**

1. Check backend is running:
   - Open http://localhost:8000 in browser
   - Should see API info

2. Check CORS settings:
   - Backend should have CORS enabled (already configured)

3. Check .env file:
   - Should have: `REACT_APP_API_URL=http://localhost:8000`
   - Restart frontend after changing .env

### Issue 4: No data showing

**Error: "No predictions found"**

```bash
# Solution: Generate predictions first
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model
python eval_real.py --subjects VEP --checkpoint checkpoints/best_model.pt
```

**Error: "Cortex mesh not found"**
```bash
# Solution: Check file exists
# File: anatomy/fs_cortex_20k.mat
# Should contain 'pos' and 'tri' fields
```

### Issue 5: 3D visualization issues

**Problem: Blank screen or no 3D rendering**

1. Check browser console (F12) for errors
2. Verify WebGL support: https://get.webgl.org/
3. Update graphics drivers
4. Try different browser (Chrome recommended)
5. Check that data loaded successfully

**Problem: Slow rendering**

1. Reduce mesh resolution (use different cortex file)
2. Close other applications
3. Use hardware acceleration in browser
4. Increase threshold to show fewer vertices

## Advanced Configuration

### Custom Cortex Mesh

To use a different cortex mesh:

1. Place your `.mat` file in `anatomy/` folder
2. File must contain:
   - `pos`: (N, 3) array of vertex positions
   - `tri`: (M, 3) array of face indices (1-indexed)
3. Update `visualization_app/backend/app.py`:
   ```python
   cortex_path = ANATOMY_DIR / "your_cortex_file.mat"
   ```

### Multiple Subjects

To add more subjects:

1. Create folder in `source/` directory
2. Add data files: `data*.mat`
3. Generate predictions:
   ```bash
   python eval_real.py --subjects YourSubject --checkpoint checkpoints/best_model.pt
   ```
4. Refresh frontend - new subject will appear in dropdown

### Custom Color Scheme

Edit `visualization_app/frontend/src/components/CortexVisualization.js`:

```javascript
function getColorForActivation(value) {
  // Modify color mapping here
  // Current: hot colormap (black -> red -> yellow)
  // Example: cool colormap (blue -> cyan -> white)
  if (value <= 0) {
    return { r: 0, g: 0, b: 0.2 };
  }
  // ... customize colors
}
```

## Performance Optimization

### Backend Optimization

1. **Pre-compute predictions**: Run `eval_real.py` first
2. **Increase cache size**: Modify `PREDICTIONS_CACHE` in `app.py`
3. **Use GPU**: Ensure CUDA is available for inference

### Frontend Optimization

1. **Reduce mesh resolution**: Use smaller cortex file
2. **Increase threshold**: Filter more vertices
3. **Disable anti-aliasing**: Modify Canvas props
4. **Use production build**:
   ```bash
   cd visualization_app/frontend
   npm run build
   # Serve with a static file server
   ```

## Development Tips

### Hot Reload

Both frontend and backend support hot reload:
- **Frontend**: Saves automatically trigger reload
- **Backend**: Use `uvicorn --reload` for auto-restart

### Debugging

**Backend:**
```python
# Add print statements in app.py
print(f"Debug: {variable}")

# Or use Python debugger
import pdb; pdb.set_trace()
```

**Frontend:**
```javascript
// Use console.log
console.log('Debug:', variable);

// Or React DevTools (browser extension)
```

### API Documentation

When backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Next Steps

1. ✓ Setup complete
2. ✓ Application running
3. → Explore the visualization
4. → Try different subjects
5. → Adjust visualization parameters
6. → Generate new predictions
7. → Customize the interface

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review error messages in:
   - Backend terminal
   - Frontend terminal
   - Browser console (F12)
3. Check file paths and permissions
4. Verify all dependencies installed
5. Try restarting both servers

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] .env file created
- [ ] Data files present
- [ ] Backend starts successfully
- [ ] Frontend starts successfully
- [ ] Can access http://localhost:3000
- [ ] Can see 3D visualization
- [ ] Can navigate samples

If all items checked, you're ready to use the app! 🎉

