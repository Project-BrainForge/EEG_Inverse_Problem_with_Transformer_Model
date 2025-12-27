# New Features Guide

## 🎉 What's New

Your visualization app now includes two powerful new features:

### 1. 📤 File Upload
Upload your own EEG data files directly in the browser!

### 2. 🎬 Animation/Video Playback
Automatically play through time points like a video with speed controls!

---

## 📤 File Upload Feature

### How to Use

1. **Locate the Upload Section**
   - Scroll down in the left panel
   - Look for "📁 Upload EEG Data" section

2. **Upload Your File**
   - **Option A**: Click the drop zone and select a .mat file
   - **Option B**: Drag and drop your .mat file onto the drop zone

3. **Click "Upload & Predict"**
   - The app will:
     - Upload your file to the server
     - Run the transformer model
     - Generate predictions
     - Display results immediately

### File Requirements

Your MAT file should contain:
- **Field name**: `data`, `eeg_data`, `eeg`, or `EEG`
- **Shape**: `(time_points, 75 channels)` or `(75 channels, time_points)`
- **Format**: Standard MATLAB .mat file

### What Happens Automatically

- ✅ **Auto-transpose**: If channels are in the first dimension, automatically transposes
- ✅ **Auto-resize**: Pads or truncates to 500 time points
- ✅ **Auto-normalize**: Centers and normalizes the data
- ✅ **Real-time inference**: Runs prediction using the trained model

### Example Files

Works with files like:
- `data3.mat` (your current VEP data)
- Any EEG recording with 75 channels
- Visual evoked potentials (VEP)
- Auditory evoked potentials (AEP)
- Motor-related potentials

---

## 🎬 Animation/Video Playback Feature

### How to Use

1. **Load Data First**
   - Either select a subject (VEP) OR upload a file
   - Wait for predictions to load

2. **Start Animation**
   - Click the **▶ Play** button
   - Watch the brain activity animate through time!

3. **Control Playback**
   - **⏸ Pause**: Click Play button again to pause
   - **⏹ Stop**: Reset to the beginning
   - **Speed**: Choose from 0.5x to 8x speed

### Animation Controls

#### Play/Pause (▶/⏸)
- Starts or pauses the animation
- Animation automatically loops back to start when finished

#### Stop (⏹)
- Stops playback
- Resets to time point 1
- Clears current position

#### Speed Options
- **0.5x**: Slower (2000ms per frame) - Best for detailed observation
- **1x**: Normal (1000ms per frame) - Standard speed
- **2x**: Fast (500ms per frame) - Quick overview
- **4x**: Very Fast (250ms per frame) - Rapid scan
- **8x**: Ultra Fast (125ms per frame) - Very quick preview

### What You'll See

The animation shows:
- **Brain activation changing over time**
- **Temporal dynamics** of the EEG response
- **Progression** from time point 1 to 500
- **Real-time color updates** on the 3D cortex

### Use Cases

#### Research Analysis
- Observe how brain activation propagates over time
- Identify peak activation periods
- Study temporal patterns in evoked potentials

#### Presentations
- Create dynamic demonstrations of brain activity
- Show real-time brain responses
- Engage audiences with animated visualizations

#### Quality Control
- Quickly scan through all time points
- Identify artifacts or anomalies
- Verify model predictions

---

## 🎯 Combined Workflow

### Complete Usage Example

1. **Upload Your Data**
   ```
   - Drag your_eeg_data.mat to the upload zone
   - Click "Upload & Predict"
   - Wait 5-10 seconds for processing
   ```

2. **Start Animation**
   ```
   - Click ▶ Play
   - Select 2x speed for a quick preview
   - Watch the brain activity unfold!
   ```

3. **Detailed Analysis**
   ```
   - Pause at interesting moments
   - Adjust threshold to focus on strong activations
   - Use sample slider for manual control
   ```

4. **Try Different Files**
   ```
   - Upload another .mat file
   - Compare different recordings
   - Analyze various conditions
   ```

---

## 🎨 UI Overview

### Left Panel Layout (Top to Bottom)

1. **Controls Section**
   - Subject selection (for pre-loaded data)
   - Sample navigation (manual control)
   - Threshold adjustment
   - Normalization toggle

2. **Animation Controls** (NEW!)
   - Time display (current / total)
   - Play/Pause button
   - Stop button
   - Speed selection

3. **Statistics Panel**
   - Activation statistics
   - File information
   - Color legend

4. **File Upload** (NEW!)
   - Drag & drop zone
   - Upload button
   - File requirements

---

## 🚀 Performance Tips

### For Best Animation Experience

1. **Use appropriate speed**
   - 1x for normal viewing
   - 2x-4x for quick scans
   - 0.5x for detailed analysis

2. **Adjust threshold**
   - Higher threshold (0.3-0.5) = faster rendering
   - Lower threshold (0.1) = more detail, slower

3. **Hardware acceleration**
   - Use Chrome for best WebGL performance
   - Close other browser tabs
   - Enable GPU acceleration in browser settings

### For File Upload

1. **File size**
   - Smaller files upload faster
   - Typical size: 200KB - 2MB
   - Maximum recommended: 10MB

2. **Internet connection**
   - Local server = instant
   - Remote server = depends on bandwidth

---

## 🐛 Troubleshooting

### Animation Issues

**Problem**: Animation is choppy
- **Solution**: Reduce threshold or use faster speed

**Problem**: Animation won't start
- **Solution**: Make sure predictions are loaded first

**Problem**: Animation stops abruptly
- **Solution**: Check browser console (F12) for errors

### Upload Issues

**Problem**: "File must be a .mat file"
- **Solution**: Ensure file extension is `.mat`

**Problem**: "Expected data with 75 channels"
- **Solution**: Verify your EEG data has exactly 75 channels

**Problem**: Upload takes too long
- **Solution**: 
  - Check file size (should be < 10MB)
  - Verify backend is running
  - Check network connection

**Problem**: "Error processing file"
- **Solution**:
  - Verify MAT file contains 'data' field
  - Check data shape is (time, 75) or (75, time)
  - Ensure file isn't corrupted

---

## 💡 Tips & Tricks

### Animation Tips

1. **Loop viewing**: Animation automatically loops - perfect for continuous monitoring
2. **Sync with events**: Note time points of interest while playing
3. **Speed comparison**: Try different speeds to see different aspects
4. **Threshold animation**: Adjust threshold during playback for different views

### Upload Tips

1. **Batch analysis**: Upload multiple files one after another
2. **Quick preview**: Upload → Play at 8x → Identify interesting periods
3. **Compare conditions**: Upload different experimental conditions
4. **Save predictions**: Backend caches results for quick re-access

### Combined Features

1. **Upload + Auto-play**: Upload a file and immediately hit play
2. **Speed + Threshold**: Adjust both for optimal viewing
3. **Manual + Auto**: Use slider to jump, then play from there
4. **Multiple uploads**: Try different files with different animation speeds

---

## 📊 Technical Details

### Backend Processing

When you upload a file:
1. File saved to temporary location
2. MAT file loaded with scipy
3. Data validated and preprocessed
4. Model inference runs on GPU/CPU
5. Predictions returned as JSON
6. Temporary file cleaned up

### Animation Mechanism

How auto-play works:
1. Uses JavaScript `setInterval`
2. Updates currentSample every N milliseconds
3. React re-renders visualization
4. Three.js updates colors in real-time
5. Smooth transitions between frames

### Performance

- **Upload**: ~5-10 seconds (depending on file size)
- **Inference**: ~1-3 seconds per file
- **Animation**: 60 FPS (smooth)
- **Frame update**: < 16ms per frame

---

## 🎓 Example Workflows

### Workflow 1: Quick Analysis

```
1. Upload file
2. Start at 4x speed
3. Pause at interesting moments
4. Adjust threshold
5. Resume at 1x speed
```

### Workflow 2: Detailed Study

```
1. Upload file
2. Start at 0.5x speed
3. Observe full sequence
4. Stop and manually navigate
5. Screenshot interesting frames
```

### Workflow 3: Comparison

```
1. Upload condition A
2. Play through at 2x
3. Note patterns
4. Upload condition B
5. Compare animations
```

### Workflow 4: Presentation

```
1. Upload demo file
2. Adjust threshold for clarity
3. Set speed to 1x
4. Let it loop continuously
5. Explain while it plays
```

---

## 🎉 Summary

You now have:

✅ **File Upload**: Direct upload of EEG .mat files
✅ **Auto Prediction**: Automatic model inference
✅ **Animation Controls**: Play/Pause/Stop with speed control
✅ **Video Playback**: Smooth progression through time points
✅ **Speed Options**: 5 different playback speeds
✅ **Loop Playback**: Automatic loop at end
✅ **Real-time Updates**: Instant visualization updates

**Your visualization app is now a complete analysis and presentation tool!**

Enjoy exploring brain dynamics in real-time! 🧠✨

