# 🧠 EEG Transformer Visualization - START HERE

Welcome! This guide will help you visualize your EEG source localization predictions on a 3D brain surface.

## 🎯 What You Want to Do

You have run `eval_real.py` and generated predictions. Now you want to see them visualized on a 3D brain.

## ⚡ Quick Start (3 Steps)

Open MATLAB and run:

```matlab
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\misc_scripts
test_visualization
visualize_transformer_predictions
```

**That's it!** You should see your predictions on a 3D brain surface.

## 🕐 Have Temporal Data (500 timepoints)?

If your predictions have shape `(num_samples, 500, 994)`, visualize all timepoints:

```matlab
cd misc_scripts
visualize_timepoints(1, 16)              % Grid view (16 timepoints)
visualize_selected_timepoints(1, 1:50:500)  % Every 50th timepoint
visualize_all_timepoints_animation(1)    % Full animation
```

See `TEMPORAL_VISUALIZATION_SUMMARY.txt` for complete guide!

## 📚 Documentation Guide

### 🆕 New to This?

**Read in this order:**

1. **`VISUALIZATION_SUMMARY.txt`** (5 min)
   - Overview of what was created
   - What each script does
   - Quick start guide

2. **`HOW_TO_VISUALIZE.md`** (15 min)
   - Complete step-by-step tutorial
   - Understanding the visualization
   - Troubleshooting guide
   - **Most comprehensive guide**

3. **`QUICK_REFERENCE.txt`** (keep handy)
   - Quick command reference
   - Common parameters
   - Tips and tricks

### 🔄 Returning User?

**Quick references:**

- **`QUICK_REFERENCE.txt`** - Commands and parameters
- **`VISUALIZATION_QUICKSTART.txt`** - Quick lookup
- **`misc_scripts/VISUALIZATION_GUIDE.md`** - Advanced examples

### 🎨 Visual Learner?

**See diagrams:**

- **`misc_scripts/WORKFLOW_DIAGRAM.txt`** - ASCII workflow diagrams

### 🔧 Need Advanced Features?

**Detailed documentation:**

- **`misc_scripts/VISUALIZATION_GUIDE.md`** - Comprehensive guide with examples
- **`misc_scripts/README_VISUALIZATION.md`** - API reference

## 📁 What Was Created

### MATLAB Scripts (in `misc_scripts/`)

#### Regular Visualization (Single Timepoint)

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_visualization.m` | Test setup | `test_visualization` |
| `visualize_transformer_predictions.m` | Main visualization | `visualize_transformer_predictions` |
| `visualize_single_prediction.m` | Single sample | `visualize_single_prediction(1)` |
| `compare_predictions.m` | Compare models | `compare_predictions` |
| `inspect_anatomy_data.m` | Explore data | `inspect_anatomy_data` |

#### Temporal Visualization (500 Timepoints) 🆕

| Script | Purpose | Usage |
|--------|---------|-------|
| `visualize_timepoints.m` | Grid view | `visualize_timepoints(1, 16)` |
| `visualize_selected_timepoints.m` | Custom selection | `visualize_selected_timepoints(1, 1:50:500)` |
| `visualize_all_timepoints_animation.m` | Animation | `visualize_all_timepoints_animation(1)` |

### Documentation Files

#### Regular Visualization

| File | Purpose | Read When |
|------|---------|-----------|
| `VISUALIZATION_SUMMARY.txt` | Complete overview | First time |
| `HOW_TO_VISUALIZE.md` | Step-by-step tutorial | Learning |
| `QUICK_REFERENCE.txt` | Quick lookup | Daily use |
| `VISUALIZATION_QUICKSTART.txt` | Quick reference | Daily use |
| `FILES_CREATED.md` | File index | Reference |
| `misc_scripts/VISUALIZATION_GUIDE.md` | Comprehensive guide | Advanced |
| `misc_scripts/README_VISUALIZATION.md` | API reference | Development |
| `misc_scripts/WORKFLOW_DIAGRAM.txt` | Visual workflow | Understanding |

#### Temporal Visualization 🆕

| File | Purpose | Read When |
|------|---------|-----------|
| `TEMPORAL_VISUALIZATION_SUMMARY.txt` | Temporal overview | First time |
| `misc_scripts/TEMPORAL_VISUALIZATION_GUIDE.md` | Detailed temporal guide | Learning |
| `misc_scripts/TEMPORAL_QUICKSTART.txt` | Quick temporal reference | Daily use |

## 🚀 Recommended Workflow

### First Time Setup

```matlab
% Step 1: Navigate to scripts folder
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\misc_scripts

% Step 2: Test your setup
test_visualization

% Step 3: Explore your data (optional)
inspect_anatomy_data

% Step 4: Visualize predictions
visualize_transformer_predictions
```

### Daily Use

```matlab
cd misc_scripts
visualize_single_prediction(1)  % Quick view
```

### Model Comparison

```matlab
cd misc_scripts
compare_predictions  % After editing file
```

## 🎓 Learning Path

### Beginner Path (30 minutes)

1. ✅ Read `VISUALIZATION_SUMMARY.txt` (5 min)
2. ✅ Read `HOW_TO_VISUALIZE.md` (15 min)
3. ✅ Run `test_visualization` in MATLAB (2 min)
4. ✅ Run `visualize_transformer_predictions` (2 min)
5. ✅ Experiment with parameters (5 min)

### Intermediate Path (1 hour)

1. ✅ Complete Beginner Path
2. ✅ Read `VISUALIZATION_GUIDE.md` (20 min)
3. ✅ Try `compare_predictions.m` (10 min)
4. ✅ Customize visualization parameters (10 min)

### Advanced Path (2 hours)

1. ✅ Complete Intermediate Path
2. ✅ Read `README_VISUALIZATION.md` (30 min)
3. ✅ Create custom visualization scripts (30 min)
4. ✅ Batch export all samples (15 min)

## 🔍 What You're Visualizing

### Data Flow

```
Python (eval_real.py)
  ↓
Predictions: 994 brain regions
  ↓
MATLAB (region mapping)
  ↓
Vertex values: 20,484 vertices
  ↓
3D Brain Visualization
```

### Key Numbers

- **75** = EEG channels (input)
- **994** = Brain regions (model output)
- **20,484** = Cortex vertices (visualization)

## 🛠️ Common Tasks

### View a Single Sample

```matlab
visualize_single_prediction(1)
```

### View Multiple Samples

```matlab
visualize_transformer_predictions
```

### Save a Figure

```matlab
saveas(gcf, 'my_result.png')
```

### Change View Angle

```matlab
view([-86, 17])  % Left
view([86, 17])   % Right
view([0, 90])    % Top
```

### Adjust Threshold

Edit `visualize_transformer_predictions.m` line 27:
```matlab
threshold = 0.2;  % Show only strong activations
```

## ❓ Troubleshooting

### Problem: Script doesn't work

**Solution:**
```matlab
cd misc_scripts
test_visualization  % This will diagnose the issue
```

### Problem: "Cannot find file"

**Solution:**
```matlab
% Make sure you're in the right directory
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\misc_scripts
```

### Problem: No prediction files

**Solution:**
Run `eval_real.py` first:
```bash
python eval_real.py --checkpoint checkpoints/best_model.pt
```

### More Help

See `HOW_TO_VISUALIZE.md` troubleshooting section for detailed solutions.

## 📊 Customization

### Change Prediction File

Edit `visualize_transformer_predictions.m` line 18:
```matlab
prediction_file = '../source/VEP/YOUR_FILE.mat';
```

### Adjust Visualization

Edit `visualize_transformer_predictions.m` lines 24-29:
```matlab
num_samples_to_show = 10;   % Show more samples
face_alpha = 0.9;           % Less transparent
threshold = 0.2;            % Higher threshold
view_angle = [90, 0];       # Different angle
```

## 🎯 Next Steps

After basic visualization:

1. **Compare Models** - Use `compare_predictions.m`
2. **Batch Export** - Save all samples as images
3. **Custom Views** - Create multiple view angles
4. **Animations** - Create video of predictions
5. **Publications** - Export high-resolution figures

See `VISUALIZATION_GUIDE.md` for examples of all these tasks.

## 📖 Full Documentation Index

### Quick Start
- ⭐ `START_HERE.md` (this file)
- 📋 `VISUALIZATION_SUMMARY.txt`
- ⚡ `QUICK_REFERENCE.txt`

### Tutorials
- 📚 `HOW_TO_VISUALIZE.md` (comprehensive tutorial)
- 🎓 `VISUALIZATION_QUICKSTART.txt`

### Reference
- 📖 `misc_scripts/VISUALIZATION_GUIDE.md`
- 🔧 `misc_scripts/README_VISUALIZATION.md`
- 📊 `misc_scripts/WORKFLOW_DIAGRAM.txt`

### Index
- 📁 `FILES_CREATED.md`

## 💡 Tips

- ✅ Start with `test_visualization.m`
- ✅ Use `visualize_single_prediction(1)` for quick tests
- ✅ Adjust threshold to focus on strong activations
- ✅ Rotate view interactively in MATLAB
- ✅ Save both `.png` and `.fig` formats
- ✅ Compare different model checkpoints

## 🎉 You're Ready!

Everything is set up and ready to use. Just run:

```matlab
cd misc_scripts
test_visualization
visualize_transformer_predictions
```

Enjoy visualizing your EEG transformer predictions! 🧠✨

---

**Need help?** Read `HOW_TO_VISUALIZE.md` for detailed instructions.

**Quick lookup?** See `QUICK_REFERENCE.txt` for commands.

**Advanced features?** Check `VISUALIZATION_GUIDE.md` for examples.
