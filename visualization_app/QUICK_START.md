# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Start the Backend

**Option A - Using the batch file (Windows):**
```bash
# Double-click or run:
start_backend.bat
```

**Option B - Manual:**
```bash
cd visualization_app/backend
pip install -r requirements.txt
python app.py
```

Wait for the message: `Uvicorn running on http://0.0.0.0:8000`

### Step 2: Start the Frontend

**Open a NEW terminal/command prompt**

**Option A - Using the batch file (Windows):**
```bash
# Double-click or run:
start_frontend.bat
```

**Option B - Manual:**
```bash
cd visualization_app/frontend
npm install  # Only needed first time
npm start
```

Wait for the browser to open automatically at `http://localhost:3000`

### Step 3: Use the App

1. **Select Subject**: Choose "VEP" from the dropdown
2. **View Predictions**: The 3D cortex will load with activations
3. **Navigate**: Use the sample slider to view different predictions
4. **Adjust Threshold**: Move the threshold slider to filter activations
5. **Interact with 3D**:
   - Drag to rotate
   - Scroll to zoom
   - Right-click and drag to pan

## 🎯 All-in-One Start (Windows)

For the easiest experience, just run:
```bash
start_app.bat
```

This will start both servers in separate windows!

## ⚠️ Troubleshooting

### Backend won't start
- Make sure you're in the project root directory
- Check that Python is installed: `python --version`
- Activate virtual environment if needed
- Install dependencies: `pip install -r visualization_app/backend/requirements.txt`

### Frontend won't start
- Check that Node.js is installed: `node --version`
- If Node.js is not installed, download from: https://nodejs.org/
- Delete `node_modules` and run `npm install` again

### Can't connect to backend
- Verify backend is running on port 8000
- Check firewall settings
- Try accessing http://localhost:8000 directly in browser

### 3D visualization not showing
- Wait for data to load (check browser console)
- Make sure prediction files exist in `source/VEP/`
- Check that cortex mesh exists in `anatomy/fs_cortex_20k.mat`

## 📁 Required Files

Before running, ensure these files exist:

```
project_root/
├── anatomy/
│   └── fs_cortex_20k.mat          ✓ Required
├── source/
│   └── VEP/
│       └── transformer_predictions_best_model.mat  ✓ Required
└── checkpoints/
    └── best_model.pt               ✓ Optional (for inference)
```

If predictions don't exist, generate them first:
```bash
python eval_real.py --subjects VEP --checkpoint checkpoints/best_model.pt
```

## 🎨 What You'll See

1. **Left Panel**: Controls and statistics
2. **Main View**: 3D interactive cortex with color-coded activations
3. **Color Scale**: 
   - Dark gray/blue = No activation
   - Red = Low activation
   - Orange/Yellow = High activation

## 🔧 Configuration

### Change Backend Port
Edit `visualization_app/backend/app.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
```

Then update frontend `.env`:
```
REACT_APP_API_URL=http://localhost:8001
```

### Change Frontend Port
Set environment variable before starting:
```bash
# Windows
set PORT=3001
npm start

# Linux/Mac
PORT=3001 npm start
```

## 💡 Tips

1. **Generate predictions first** using `eval_real.py` for faster loading
2. **Adjust threshold** to focus on strong activations
3. **Use normalization** for consistent visualization across samples
4. **Try different viewing angles** to see all cortex regions
5. **Check the stats panel** for activation statistics

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the API at http://localhost:8000/docs (when backend is running)
- Customize the visualization in `frontend/src/components/CortexVisualization.js`

## 🆘 Need Help?

Check the browser console (F12) and backend terminal for error messages.

