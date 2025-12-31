# Temporal Visualization Guide

This guide explains how to visualize all 500 timepoints from your EEG transformer predictions.

## Overview

If your predictions have shape `(num_samples, 500, 994)`, you have temporal data with 500 timepoints per sample. This guide shows you how to visualize the temporal evolution of brain activity.

## Three Approaches

### 1. Multiple Subplots (Recommended for Analysis)

Show selected timepoints in a single figure with subplots.

```matlab
% Show 16 evenly spaced timepoints
visualize_timepoints(1, 16)

% Show 20 timepoints
visualize_timepoints(1, 20)

% Show 25 timepoints (5x5 grid)
visualize_timepoints(1, 25)
```

**Pros:** Easy to compare different timepoints, good for publications  
**Cons:** Limited number of timepoints visible at once

### 2. Custom Selected Timepoints

Choose specific timepoints you want to visualize.

```matlab
% Show specific timepoints
visualize_selected_timepoints(1, [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

% Show every 25th timepoint
visualize_selected_timepoints(1, 1:25:500)

% Show early timepoints (first 200ms)
visualize_selected_timepoints(1, 1:10:100)

% With different view angle
visualize_selected_timepoints(1, 1:50:500, [86, 17])  % Right view
```

**Pros:** Full control over which timepoints to show  
**Cons:** Need to know which timepoints are interesting

### 3. Animation (Best for Presentations)

Create a video showing all 500 timepoints.

```matlab
% Create animation for sample 1
visualize_all_timepoints_animation(1)

% Save to specific location
visualize_all_timepoints_animation(1, '../results/my_animation.mp4')
```

**Pros:** Shows complete temporal evolution, great for presentations  
**Cons:** Takes time to generate, harder to analyze specific timepoints

## Quick Start

### If your data has temporal information:

```matlab
cd misc_scripts

% Option 1: Quick grid view (16 timepoints)
visualize_timepoints(1, 16)

% Option 2: Custom timepoints (every 50 timepoints)
visualize_selected_timepoints(1, 1:50:500)

% Option 3: Full animation (all 500 timepoints)
visualize_all_timepoints_animation(1)
```

### If your data is single timepoint:

Your predictions have shape `(num_samples, 994)` - use the regular scripts:

```matlab
visualize_single_prediction(1)
visualize_transformer_predictions
```

## Detailed Usage

### visualize_timepoints.m

Shows evenly spaced timepoints in a grid.

```matlab
function visualize_timepoints(sample_idx, num_timepoints)

% Parameters:
%   sample_idx      - Which sample to visualize (1, 2, 3, ...)
%   num_timepoints  - How many timepoints to show (default: 16)

% Examples:
visualize_timepoints(1, 16)   % Sample 1, 16 timepoints (4x4 grid)
visualize_timepoints(2, 20)   % Sample 2, 20 timepoints (4x5 grid)
visualize_timepoints(1, 9)    % Sample 1, 9 timepoints (3x3 grid)
```

**Best for:** Quick overview of temporal evolution

### visualize_selected_timepoints.m

Shows specific timepoints you choose.

```matlab
function visualize_selected_timepoints(sample_idx, timepoints, view_angle)

% Parameters:
%   sample_idx   - Which sample to visualize
%   timepoints   - Array of timepoint indices to show
%   view_angle   - Camera angle [azimuth, elevation] (default: [-86, 17])

% Examples:

% Show first, middle, and last
visualize_selected_timepoints(1, [1, 250, 500])

% Show every 50th
visualize_selected_timepoints(1, 1:50:500)

% Show early response (0-200ms, every 10 timepoints)
visualize_selected_timepoints(1, 1:10:100)

% Show peak activity periods (you determined these from analysis)
visualize_selected_timepoints(1, [50, 75, 100, 125, 150, 175, 200])

% Right hemisphere view
visualize_selected_timepoints(1, 1:50:500, [86, 17])

% Top view
visualize_selected_timepoints(1, 1:50:500, [0, 90])
```

**Best for:** Focused analysis of specific time periods

### visualize_all_timepoints_animation.m

Creates video with all timepoints.

```matlab
function visualize_all_timepoints_animation(sample_idx, output_file)

% Parameters:
%   sample_idx   - Which sample to animate
%   output_file  - Output video path (default: ../results/temporal_evolution_sample_N.mp4)

% Examples:

% Default output location
visualize_all_timepoints_animation(1)

% Custom output path
visualize_all_timepoints_animation(1, '../results/my_animation.mp4')

% Multiple samples
for i = 1:5
    visualize_all_timepoints_animation(i, sprintf('../results/sample_%d.mp4', i));
end
```

**Settings:**
- Frame rate: 30 fps (adjustable in script)
- Quality: 95% (adjustable in script)
- Duration: ~16.7 seconds for 500 frames at 30 fps

**Best for:** Presentations, videos, full temporal visualization

## Understanding Your Data

### Check Data Shape

```matlab
pred_data = load('../source/VEP/transformer_predictions_best_model.mat');
fprintf('Shape: %s\n', mat2str(size(pred_data.all_out)));
```

**Possible shapes:**

1. **`(num_samples, 994)`** - Single timepoint per sample
   - Use: `visualize_single_prediction`, `visualize_transformer_predictions`

2. **`(num_samples, 500, 994)`** - Full temporal data
   - Use: `visualize_timepoints`, `visualize_selected_timepoints`, `visualize_all_timepoints_animation`

## Examples by Use Case

### For Publication Figures

```matlab
% Show key timepoints in a clean grid
visualize_selected_timepoints(1, [1, 100, 200, 300, 400, 500])
saveas(gcf, 'figure_temporal_evolution.png')

% Multiple views of same timepoint
for t = [50, 100, 150, 200]
    visualize_selected_timepoints(1, t, [-86, 17]);
    saveas(gcf, sprintf('timepoint_%d_left.png', t));
    
    visualize_selected_timepoints(1, t, [86, 17]);
    saveas(gcf, sprintf('timepoint_%d_right.png', t));
end
```

### For Presentations

```matlab
% Create smooth animation
visualize_all_timepoints_animation(1, '../presentation/brain_activity.mp4')

% Or show key frames in PowerPoint
visualize_timepoints(1, 12)  % 3x4 grid fits nicely in slides
saveas(gcf, '../presentation/temporal_overview.png')
```

### For Analysis

```matlab
% Quick overview
visualize_timepoints(1, 16)

% Focus on early response (0-200ms)
visualize_selected_timepoints(1, 1:10:100)

% Focus on late response (200-1000ms)
visualize_selected_timepoints(1, 100:25:500)

% Compare different samples at same timepoints
for sample_idx = 1:5
    visualize_selected_timepoints(sample_idx, 1:50:500);
    title(sprintf('Sample %d', sample_idx));
    saveas(gcf, sprintf('sample_%d_temporal.png', sample_idx));
    close(gcf);
end
```

### For Video Presentations

```matlab
% Create high-quality animations for all samples
for sample_idx = 1:num_samples
    fprintf('Processing sample %d...\n', sample_idx);
    visualize_all_timepoints_animation(sample_idx, ...
        sprintf('../videos/sample_%03d.mp4', sample_idx));
end
```

## Customization

### Change Subplot Layout

Edit `visualize_timepoints.m`:

```matlab
% Current (automatic):
subplot_rows = floor(sqrt(num_timepoints));
subplot_cols = ceil(num_timepoints / subplot_rows);

% Force specific layout (e.g., always 4 rows):
subplot_rows = 4;
subplot_cols = ceil(num_timepoints / 4);
```

### Change Animation Speed

Edit `visualize_all_timepoints_animation.m`:

```matlab
% Current:
v.FrameRate = 30;  % 30 fps

% Slower (easier to see details):
v.FrameRate = 15;  % 15 fps

% Faster (shorter video):
v.FrameRate = 60;  % 60 fps
```

### Change Threshold

All scripts use `threshold = 0.1`. To change:

```matlab
% In any script, find and modify:
threshold = 0.1;  % Default

% To show more activation:
threshold = 0.05;

% To show only strong activation:
threshold = 0.3;
```

### Change View Angle

For `visualize_selected_timepoints`:

```matlab
% Left view (default)
visualize_selected_timepoints(1, 1:50:500, [-86, 17])

% Right view
visualize_selected_timepoints(1, 1:50:500, [86, 17])

% Top view
visualize_selected_timepoints(1, 1:50:500, [0, 90])

% Front view
visualize_selected_timepoints(1, 1:50:500, [0, 0])
```

For `visualize_timepoints` and animation, edit the script to change view angle.

## Troubleshooting

### Error: "Expected 3D data"

Your data has shape `(num_samples, 994)` instead of `(num_samples, 500, 994)`.

**Solution:** You only have single timepoint data. Use regular visualization scripts:
```matlab
visualize_single_prediction(1)
```

### Animation takes too long

**Solutions:**
1. Reduce frame rate: Change `v.FrameRate = 30` to `v.FrameRate = 15`
2. Skip frames: Modify script to process every 2nd or 5th frame
3. Reduce figure size in script

### Out of memory

**Solutions:**
1. Process fewer timepoints at once
2. Reduce subplot count in `visualize_timepoints`
3. Close figures between processing: `close(gcf)`

### Want to show all 500 in grid

500 subplots is too many! Instead:

```matlab
% Option 1: Show every 10th (50 subplots)
visualize_selected_timepoints(1, 1:10:500)

% Option 2: Show every 5th (100 subplots)
visualize_selected_timepoints(1, 1:5:500)

% Option 3: Create animation
visualize_all_timepoints_animation(1)
```

## Time Interpretation

Assuming 2ms sampling rate (500 Hz):
- Timepoint 1 = 0 ms
- Timepoint 50 = 98 ms (100 ms)
- Timepoint 100 = 198 ms (200 ms)
- Timepoint 250 = 498 ms (500 ms)
- Timepoint 500 = 998 ms (1000 ms)

Adjust labels in scripts if your sampling rate is different.

## Summary

```matlab
% Quick comparison
visualize_timepoints(1, 16)              % Grid of 16 timepoints
visualize_selected_timepoints(1, 1:50:500)  % Every 50th timepoint
visualize_all_timepoints_animation(1)    % Full animation

% For publications
visualize_selected_timepoints(1, [1, 100, 200, 300, 400, 500])

% For presentations
visualize_all_timepoints_animation(1, '../presentation/demo.mp4')

% For analysis
visualize_timepoints(1, 25)  % 5x5 grid
```

## Tips

- ✅ Start with `visualize_timepoints(1, 16)` for quick overview
- ✅ Use `visualize_selected_timepoints` for focused analysis
- ✅ Create animations for presentations and videos
- ✅ Save figures for publications
- ✅ Compare multiple samples at same timepoints
- ✅ Adjust threshold to focus on strong activations
- ✅ Use different view angles for complete understanding

Happy temporal visualization! 🧠⏱️✨

