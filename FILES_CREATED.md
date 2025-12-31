# Files Created for Visualization

This document lists all files created to help you visualize your EEG transformer predictions.

## Summary

**Total Files Created: 13**
- 5 MATLAB scripts
- 8 documentation files

## MATLAB Scripts (in `misc_scripts/`)

### 1. `visualize_transformer_predictions.m`
**Purpose:** Main visualization script for multiple samples  
**Usage:** `visualize_transformer_predictions`  
**Features:**
- Visualizes first 5 samples in a subplot
- Converts 994 regions to 20,484 vertices
- Configurable parameters (threshold, transparency, view angle)
- Prints statistics
- Can be customized for different prediction files

### 2. `visualize_single_prediction.m`
**Purpose:** Quick function to visualize a single sample  
**Usage:** `visualize_single_prediction(sample_idx)`  
**Features:**
- Fast single-sample visualization
- Interactive exploration
- Good for testing and quick checks

### 3. `compare_predictions.m`
**Purpose:** Compare predictions from multiple models  
**Usage:** `compare_predictions` (after editing file)  
**Features:**
- Side-by-side comparison
- Difference maps
- Correlation analysis
- Good for model evaluation

### 4. `inspect_anatomy_data.m`
**Purpose:** Explore anatomy data structure  
**Usage:** `inspect_anatomy_data`  
**Features:**
- Shows cortex from 4 views
- Displays region mapping
- Prints data statistics
- Helps understand data structure

### 5. `test_visualization.m`
**Purpose:** Test visualization setup  
**Usage:** `test_visualization`  
**Features:**
- Checks if all required files exist
- Tests data loading
- Verifies visualization function works
- Reports any issues
- **Run this first!**

## Documentation Files

### Root Level Documentation

#### 6. `HOW_TO_VISUALIZE.md`
**Purpose:** Complete step-by-step tutorial  
**Contents:**
- Step-by-step guide for first-time users
- Understanding the visualization
- Advanced usage examples
- Comprehensive troubleshooting
- Tips and best practices
- **START HERE if you're new!**

#### 7. `VISUALIZATION_QUICKSTART.txt`
**Purpose:** Quick reference card  
**Contents:**
- Quick start (3 steps)
- Common commands
- Key parameters
- View angles
- File locations
- Troubleshooting tips
- **Keep this handy for quick lookups**

#### 8. `VISUALIZATION_SUMMARY.txt`
**Purpose:** Complete overview of visualization system  
**Contents:**
- What was created
- What each script does
- Data flow explanation
- Required files
- Customization guide
- Common tasks
- Next steps

#### 9. `QUICK_REFERENCE.txt`
**Purpose:** Formatted quick reference card  
**Contents:**
- Main commands
- Quick examples
- Key parameters
- Common view angles
- File locations
- Data dimensions
- Tips and workflow

#### 10. `FILES_CREATED.md`
**Purpose:** This file - index of all created files  
**Contents:**
- Complete list of files
- Purpose of each file
- Usage instructions
- File organization

### misc_scripts Documentation

#### 11. `misc_scripts/VISUALIZATION_GUIDE.md`
**Purpose:** Comprehensive visualization guide  
**Contents:**
- Detailed parameter explanations
- Multiple examples
- Advanced usage patterns
- Batch processing
- Custom colormaps
- Animation creation
- **Read this for advanced usage**

#### 12. `misc_scripts/README_VISUALIZATION.md`
**Purpose:** Script documentation and API reference  
**Contents:**
- Script overview
- Data flow
- Configuration options
- Examples for each script
- Troubleshooting
- Advanced usage
- File structure
- **Reference for developers**

#### 13. `misc_scripts/WORKFLOW_DIAGRAM.txt`
**Purpose:** Visual workflow diagram  
**Contents:**
- ASCII art diagrams
- Step-by-step workflow
- Data dimensions
- Conversion process
- Troubleshooting flow
- **Visual learners will love this**

## File Organization

```
EEG_Inverse_Problem_with_Transformer_Model/
│
├── misc_scripts/                              # MATLAB scripts folder
│   ├── visualize_transformer_predictions.m   # Main visualization
│   ├── visualize_single_prediction.m         # Single sample viewer
│   ├── compare_predictions.m                 # Model comparison
│   ├── inspect_anatomy_data.m                # Data explorer
│   ├── test_visualization.m                  # Setup tester
│   ├── visualize_result.m                    # Core function (existed)
│   ├── VISUALIZATION_GUIDE.md                # Comprehensive guide
│   ├── README_VISUALIZATION.md               # Script documentation
│   └── WORKFLOW_DIAGRAM.txt                  # Visual workflow
│
├── HOW_TO_VISUALIZE.md                       # Step-by-step tutorial
├── VISUALIZATION_QUICKSTART.txt              # Quick reference
├── VISUALIZATION_SUMMARY.txt                 # Complete overview
├── QUICK_REFERENCE.txt                       # Formatted reference
├── FILES_CREATED.md                          # This file
│
├── anatomy/                                   # Anatomy data (existing)
│   ├── fs_cortex_20k.mat                     # Cortex geometry
│   └── fs_cortex_20k_region_mapping.mat      # Region mapping
│
└── source/VEP/                               # Predictions (existing)
    └── transformer_predictions_*.mat         # Model outputs
```

## Which File Should I Read?

### For First-Time Users
1. **Start:** `VISUALIZATION_SUMMARY.txt` - Get overview
2. **Then:** `HOW_TO_VISUALIZE.md` - Step-by-step tutorial
3. **Keep handy:** `QUICK_REFERENCE.txt` - Quick lookups

### For Quick Reference
- `VISUALIZATION_QUICKSTART.txt` - Commands and parameters
- `QUICK_REFERENCE.txt` - Formatted reference card

### For Advanced Usage
- `misc_scripts/VISUALIZATION_GUIDE.md` - Detailed examples
- `misc_scripts/README_VISUALIZATION.md` - API reference

### For Visual Learners
- `misc_scripts/WORKFLOW_DIAGRAM.txt` - ASCII diagrams

### For Troubleshooting
- `HOW_TO_VISUALIZE.md` - Troubleshooting section
- Run `test_visualization.m` in MATLAB

## Quick Start Guide

### Absolute Beginner (Never used MATLAB for this)

1. Read: `VISUALIZATION_SUMMARY.txt` (5 minutes)
2. Read: `HOW_TO_VISUALIZE.md` (10 minutes)
3. Run in MATLAB:
   ```matlab
   cd misc_scripts
   test_visualization
   visualize_transformer_predictions
   ```

### Experienced User (Familiar with MATLAB)

1. Glance at: `QUICK_REFERENCE.txt` (2 minutes)
2. Run in MATLAB:
   ```matlab
   cd misc_scripts
   visualize_transformer_predictions
   ```
3. Customize as needed using `VISUALIZATION_GUIDE.md`

### Just Want to See Results

```matlab
cd misc_scripts
visualize_single_prediction(1)
```

## File Dependencies

### MATLAB Scripts Dependencies

```
visualize_transformer_predictions.m
  ├── Requires: visualize_result.m
  ├── Loads: fs_cortex_20k.mat
  ├── Loads: fs_cortex_20k_region_mapping.mat
  └── Loads: transformer_predictions_*.mat

visualize_single_prediction.m
  ├── Requires: visualize_result.m
  ├── Loads: fs_cortex_20k.mat
  ├── Loads: fs_cortex_20k_region_mapping.mat
  └── Loads: transformer_predictions_*.mat

compare_predictions.m
  ├── Requires: visualize_result.m
  ├── Loads: fs_cortex_20k.mat
  ├── Loads: fs_cortex_20k_region_mapping.mat
  └── Loads: multiple transformer_predictions_*.mat files

inspect_anatomy_data.m
  ├── Loads: fs_cortex_20k.mat
  ├── Loads: fs_cortex_20k_region_mapping.mat
  ├── Loads: leadfield_75_20k.mat (optional)
  └── Loads: electrode_75.mat (optional)

test_visualization.m
  ├── Requires: visualize_result.m
  ├── Loads: fs_cortex_20k.mat
  ├── Loads: fs_cortex_20k_region_mapping.mat
  └── Loads: transformer_predictions_*.mat
```

### Documentation Dependencies

- All documentation files are standalone
- Cross-references between files for different detail levels
- No external dependencies

## Maintenance

### To Update Prediction File Path

Edit line 18 in `visualize_transformer_predictions.m`:
```matlab
prediction_file = '../source/VEP/YOUR_NEW_FILE.mat';
```

### To Change Default Parameters

Edit lines 24-29 in `visualize_transformer_predictions.m`:
```matlab
num_samples_to_show = 10;   % Your value
face_alpha = 0.9;           % Your value
threshold = 0.2;            % Your value
view_angle = [90, 0];       % Your value
```

### To Add New Visualization Script

1. Create new `.m` file in `misc_scripts/`
2. Use `visualize_result.m` as the core function
3. Follow the pattern in existing scripts
4. Document in `README_VISUALIZATION.md`

## Testing

To verify everything works:

```matlab
cd misc_scripts
test_visualization
```

This will check:
- ✓ All required files exist
- ✓ Data can be loaded
- ✓ Visualization function works
- ✓ Region-to-vertex conversion works

## Support

### If You Have Issues

1. Run `test_visualization.m` to diagnose
2. Check `HOW_TO_VISUALIZE.md` troubleshooting section
3. Verify file paths and data formats
4. Check MATLAB console for error messages

### For More Examples

- See `VISUALIZATION_GUIDE.md` for comprehensive examples
- See `HOW_TO_VISUALIZE.md` for step-by-step tutorials
- Check example code in documentation files

## Summary

You now have a complete visualization system with:

- ✅ 5 MATLAB scripts for different visualization tasks
- ✅ 8 comprehensive documentation files
- ✅ Quick reference cards
- ✅ Step-by-step tutorials
- ✅ Troubleshooting guides
- ✅ Advanced examples
- ✅ Visual workflow diagrams

Everything is ready to use. Just open MATLAB and run:

```matlab
cd misc_scripts
test_visualization
visualize_transformer_predictions
```

Enjoy visualizing your EEG transformer predictions! 🧠✨

---

**Created:** 2025-12-31  
**Purpose:** EEG Source Localization Visualization  
**Project:** EEG_Inverse_Problem_with_Transformer_Model

