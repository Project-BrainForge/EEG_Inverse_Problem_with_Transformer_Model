# File Upload Guide

## 📤 How File Upload Works

When you upload a .mat file through the web interface, here's what happens:

### 1. File is Saved to VEP Folder
- Your uploaded file is **permanently saved** to `source/VEP/[your_filename].mat`
- This allows you to:
  - Access the file later
  - Re-run predictions without re-uploading
  - Keep a record of all analyzed files

### 2. Predictions are Generated and Saved
- The transformer model runs inference on your data
- Predictions are saved as: `source/VEP/transformer_predictions_[your_filename]_uploaded.mat`
- This file contains:
  - `all_out`: The prediction data (time_points × sources)
  - `file_names`: Original filename
  - `checkpoint`: Model checkpoint used
  - `num_samples`: Number of time points
  - `uploaded_file`: Path to the original uploaded file

### 3. Results are Displayed
- The 3D visualization immediately shows your results
- You can use the animation controls to play through time points
- Adjust threshold and other settings as needed

## 📁 File Structure After Upload

```
source/
└── VEP/
    ├── data3.mat (original file)
    ├── your_uploaded_file.mat (your upload)
    ├── transformer_predictions_best_model.mat (original predictions)
    └── transformer_predictions_your_uploaded_file_uploaded.mat (your predictions)
```

## ✅ Benefits of This Approach

### 1. **Persistence**
- Files aren't deleted after processing
- You can close the app and files remain
- Re-select VEP subject to see all predictions

### 2. **Reproducibility**
- Keep original data alongside predictions
- Know which model checkpoint was used
- Can re-analyze if needed

### 3. **Organization**
- All VEP-related files in one folder
- Easy to manage and backup
- Clear naming convention

## 🚀 Usage Example

### Upload and Analyze
1. **Drag & drop** your_data.mat to the upload zone
2. Click **"Upload & Predict"**
3. **Wait** 5-10 seconds for processing
4. Files are saved:
   - `source/VEP/your_data.mat`
   - `source/VEP/transformer_predictions_your_data_uploaded.mat`
5. **Visualization** appears immediately
6. Click **▶ Play** to animate through time

### Access Later
1. Refresh the page
2. Select **"VEP"** from subject dropdown
3. Your uploaded predictions appear in the list
4. Navigate through samples to see your uploaded data

## 📋 File Requirements

Your uploaded MAT file must contain:

### Required Fields
- **Field name**: `data`, `eeg_data`, `eeg`, or `EEG`
- **Data type**: Numerical array (float/double)
- **Dimensions**: (time_points, 75) or (75, time_points)

### Supported Formats
- Standard MATLAB .mat files (v5, v7, v7.3)
- Single or double precision
- Any number of time points (will be adjusted to 500)

### Example Data Structure
```matlab
% In MATLAB:
data = randn(500, 75);  % 500 time points, 75 channels
save('my_eeg_data.mat', 'data');
```

```python
# In Python with scipy:
import numpy as np
from scipy.io import savemat

data = np.random.randn(500, 75)  # 500 time points, 75 channels
savemat('my_eeg_data.mat', {'data': data})
```

## 🔧 What Happens to Your Data

### Automatic Preprocessing
1. **Validation**: Checks for 75 channels
2. **Transpose**: If needed (channels × time → time × channels)
3. **Resizing**: 
   - If > 500 time points: Truncates to 500
   - If < 500 time points: Pads with zeros
4. **Centering**: 
   - Removes mean across time
   - Removes mean across channels
5. **Normalization** (if enabled):
   - Scales to maximum absolute value

### Model Inference
- Uses your trained transformer model
- Runs on CPU or GPU (depending on availability)
- Generates predictions for all time points
- Returns source activations for 994 brain regions

## 🎯 Best Practices

### 1. File Naming
- Use descriptive names: `VEP_subject01_condition1.mat`
- Avoid special characters: Use letters, numbers, underscores
- Keep it short: < 50 characters recommended

### 2. Data Quality
- Use clean, preprocessed EEG data
- Remove artifacts before uploading
- Ensure 75 channels (standard 10-20 system)
- Typical sampling: 1000 Hz, ~500ms epochs

### 3. Organization
- Upload related files together
- Note experimental conditions in filename
- Keep backup of original raw data

### 4. Analysis Workflow
```
1. Preprocess EEG → 2. Save as .mat → 3. Upload to app
↓
4. Review predictions → 5. Animate playback → 6. Adjust threshold
↓
7. Analyze patterns → 8. Screenshot/record → 9. Document findings
```

## ⚠️ Troubleshooting

### "File must be a .mat file"
**Problem**: Wrong file format
**Solution**: Ensure file has .mat extension

### "Expected data with 75 channels"
**Problem**: Incorrect number of channels
**Solution**: 
- Check your data: `print(data.shape)` in Python
- Verify it's (time, 75) or (75, time)
- Use 75-channel montage

### "Could not find EEG data"
**Problem**: Wrong field name in MAT file
**Solution**:
- Use field name: `data`, `eeg_data`, `eeg`, or `EEG`
- Check with: `scipy.io.loadmat('file.mat').keys()`

### Upload takes too long
**Problem**: File size or network issue
**Solution**:
- Check file size (should be < 10 MB)
- Verify backend is running locally
- Close unnecessary browser tabs

### Predictions don't appear
**Problem**: Error during inference
**Solution**:
- Check backend terminal for error messages
- Verify model checkpoint exists
- Ensure data is valid numerical array

## 💾 Storage Considerations

### Disk Space
- Uploaded .mat file: ~200 KB - 2 MB
- Prediction file: ~2 MB - 5 MB
- Total per upload: ~2 MB - 7 MB

### Cleanup
To remove old uploads:
```bash
# Delete specific file
del source\VEP\old_file.mat
del source\VEP\transformer_predictions_old_file_uploaded.mat

# Or delete all uploaded predictions
del source\VEP\transformer_predictions_*_uploaded.mat
```

## 🎬 Complete Upload Workflow

### Step-by-Step
1. ✅ Prepare your .mat file with 75-channel EEG data
2. ✅ Open the visualization app (http://localhost:3000)
3. ✅ Scroll to "📁 Upload EEG Data" section
4. ✅ Drag & drop or click to select file
5. ✅ Click "🚀 Upload & Predict"
6. ✅ Wait for "Processing..." (5-10 seconds)
7. ✅ File saved to `source/VEP/`
8. ✅ Predictions saved alongside
9. ✅ Visualization appears
10. ✅ Use animation controls to explore

### What You'll See
- ✅ Progress indicator during upload
- ✅ Success message when complete
- ✅ 3D cortex with color-coded activations
- ✅ Animation controls ready to use
- ✅ Statistics panel with data info

## 📚 Related Documentation

- **[NEW_FEATURES.md](NEW_FEATURES.md)** - Overview of upload and animation features
- **[README.md](README.md)** - Complete app documentation
- **[QUICK_START.md](QUICK_START.md)** - Getting started guide

## 🎉 Summary

The file upload feature:
- ✅ **Saves** your files permanently to VEP folder
- ✅ **Generates** predictions automatically
- ✅ **Stores** predictions alongside original data
- ✅ **Enables** immediate visualization
- ✅ **Supports** animation playback
- ✅ **Maintains** reproducibility

**Your uploaded files are safe, organized, and ready for analysis!** 🧠✨

