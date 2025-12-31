# Visualization Scripts for EEG Transformer Predictions

This directory contains MATLAB scripts to visualize the predictions from your EEG source localization transformer model.

## Overview

After running `eval_real.py`, you get predictions in `.mat` format with shape `(num_samples, 994)` representing activity in 994 brain regions. These scripts help you visualize those predictions on a 3D cortex surface.

## Quick Start

```matlab
cd misc_scripts
visualize_transformer_predictions  % Visualize first 5 samples
```

Or for a single sample:

```matlab
visualize_single_prediction(1)  % Visualize sample 1
```

## Scripts Overview

### Main Visualization Scripts

1. **`visualize_transformer_predictions.m`** - Main script
   - Visualizes multiple samples in a subplot
   - Converts region-level (994) to vertex-level (20k) predictions
   - Configurable parameters (threshold, transparency, view angle)
   - Prints statistics

2. **`visualize_single_prediction.m`** - Quick single sample viewer
   - Function to quickly view one sample
   - Usage: `visualize_single_prediction(sample_idx)`
   - Good for interactive exploration

3. **`compare_predictions.m`** - Compare multiple models
   - Side-by-side comparison of different model outputs
   - Difference maps and correlation analysis
   - Useful for model evaluation

4. **`inspect_anatomy_data.m`** - Explore anatomy files
   - Shows what's in the anatomy folder
   - Displays cortex from multiple angles
   - Visualizes region mapping
   - Good for understanding the data structure

### Core Function

5. **`visualize_result.m`** - Core visualization function
   - Low-level function used by other scripts
   - Flexible parameters for customization
   - Handles cortex rendering, colormaps, lighting

## Data Flow

```
Python (eval_real.py)
  ↓
all_out: (num_samples, 994 regions)
  ↓
MATLAB (region_mapping)
  ↓
vertex_predictions: (num_samples, 20484 vertices)
  ↓
visualize_result.m
  ↓
3D Brain Visualization
```

## Required Anatomy Files

Located in `../anatomy/`:

- `fs_cortex_20k.mat` - Cortex geometry
  - `pos`: 20,484 × 3 (vertex positions)
  - `tri`: triangles (faces)

- `fs_cortex_20k_region_mapping.mat` - Region mapping
  - `region_mapping`: 20,484 × 1 (maps vertices to 994 regions)

## Configuration

### Prediction File

Edit in `visualize_transformer_predictions.m`:

```matlab
prediction_file = '../source/VEP/transformer_predictions_best_model.mat';
```

### Visualization Parameters

```matlab
num_samples_to_show = 5;   % Number of samples to display
subplot_rows = 1;          % Subplot rows
subplot_cols = 5;          % Subplot columns
face_alpha = 0.8;          % Transparency (0-1)
threshold = 0.1;           % Activation threshold (0-1)
view_angle = [-86, 17];    % Camera angle [azimuth, elevation]
```

### View Angles

| Angle | View |
|-------|------|
| `[-86, 17]` | Left lateral (default) |
| `[86, 17]` | Right lateral |
| `[0, 90]` | Top |
| `[0, 0]` | Front |
| `[180, 0]` | Back |
| `[0, -90]` | Bottom |

## Examples

### Example 1: Basic Visualization

```matlab
cd misc_scripts
visualize_transformer_predictions
```

### Example 2: Custom Parameters

```matlab
% Edit visualize_transformer_predictions.m
num_samples_to_show = 10;
subplot_rows = 2;
subplot_cols = 5;
threshold = 0.2;  % Show only strong activations
```

### Example 3: Multiple Views

```matlab
% View same sample from different angles
for angle_idx = 1:4
    angles = {[-86, 17], [86, 17], [0, 90], [0, 0]};
    view_names = {'Left', 'Right', 'Top', 'Front'};
    
    figure;
    visualize_single_prediction(1);
    view(angles{angle_idx});
    title(sprintf('Sample 1 - %s View', view_names{angle_idx}));
end
```

### Example 4: Compare Models

```matlab
% Edit compare_predictions.m
prediction_files = {
    '../source/VEP/transformer_predictions_best_model.mat';
    '../source/VEP/transformer_predictions_epoch50.mat';
};
sample_idx = 1;

% Run
compare_predictions
```

### Example 5: Batch Save All Samples

```matlab
% Load data
cortex_data = load('../anatomy/fs_cortex_20k.mat');
rm_data = load('../anatomy/fs_cortex_20k_region_mapping.mat');
pred_data = load('../source/VEP/transformer_predictions_best_model.mat');

% Create output directory
output_dir = '../results/visualizations/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Process each sample
for sample_idx = 1:size(pred_data.all_out, 1)
    fprintf('Processing sample %d/%d...\n', sample_idx, size(pred_data.all_out, 1));
    
    % Convert to vertex level
    vertex_values = zeros(size(cortex_data.pos, 1), 1);
    for v = 1:length(rm_data.region_mapping)
        r = rm_data.region_mapping(v);
        if r > 0
            vertex_values(v) = pred_data.all_out(sample_idx, r);
        end
    end
    
    % Visualize
    figure('Visible', 'off');  % Don't display
    visualize_result(cortex_data.pos, cortex_data.tri, vertex_values', ...
        'FaceAlpha', 0.8, 'thre', 0.1, 'view', [-86, 17]);
    
    % Save
    saveas(gcf, sprintf('%s/sample_%03d.png', output_dir, sample_idx));
    close(gcf);
end

fprintf('Done! Saved to %s\n', output_dir);
```

## Troubleshooting

### Issue: "Cannot find file"

**Solution:**
```matlab
% Check current directory
pwd

% Should be in misc_scripts
cd D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\misc_scripts

% Or add to path
addpath('misc_scripts')
```

### Issue: "Undefined function 'visualize_result'"

**Solution:**
```matlab
% Make sure you're in the correct directory
cd misc_scripts

% Or add to path
addpath('D:\fyp\EEG_Inverse_Problem_with_Transformer_Model\misc_scripts')
```

### Issue: Predictions look strange

**Diagnosis:**
```matlab
% Check statistics
pred_data = load('../source/VEP/transformer_predictions_best_model.mat');
fprintf('Min: %.6f\n', min(pred_data.all_out(:)));
fprintf('Max: %.6f\n', max(pred_data.all_out(:)));
fprintf('Mean: %.6f\n', mean(pred_data.all_out(:)));
fprintf('Std: %.6f\n', std(pred_data.all_out(:)));
```

**Solutions:**
- Try `threshold = 0` to see all values
- Try different normalization: `'normalize', 0` or `'normalize', 1`
- Check if predictions are in expected range

### Issue: Want to see negative values

**Solution:**
```matlab
visualize_result(pos, tri, values, ...
    'neg', 1, ...  % Enable bipolar colormap
    'normalize', 0);  % Don't normalize to [0,1]
```

## Advanced Usage

### Custom Colormap

```matlab
% Define custom colormap
my_cmap = parula(64);  % or jet, hot, cool, etc.
colormap(my_cmap);
```

### Overlay Ground Truth

If you have ground truth source locations:

```matlab
% Ground truth region indices
gt_regions = [100, 101, 102, 103];

% Visualize with overlay
visualize_result(pos, tri, predictions, ...
    'source', {gt_regions}, ...
    'SourceFaceAlpha', 0.5);
```

### Export High-Resolution Figures

```matlab
% Set figure size
figure('Position', [100, 100, 1920, 1080]);

% ... visualization code ...

% Save high-res
print(gcf, 'my_figure.png', '-dpng', '-r300');  % 300 DPI
```

### Create Animation

```matlab
% Create video writer
v = VideoWriter('predictions_animation.mp4', 'MPEG-4');
v.FrameRate = 2;  % 2 frames per second
open(v);

% For each sample
for sample_idx = 1:num_samples
    visualize_single_prediction(sample_idx);
    frame = getframe(gcf);
    writeVideo(v, frame);
    close(gcf);
end

close(v);
fprintf('Animation saved!\n');
```

## File Structure

```
misc_scripts/
├── visualize_transformer_predictions.m  # Main script
├── visualize_single_prediction.m        # Single sample viewer
├── compare_predictions.m                # Compare multiple models
├── inspect_anatomy_data.m               # Explore anatomy data
├── visualize_result.m                   # Core function
├── README_VISUALIZATION.md              # This file
└── VISUALIZATION_GUIDE.md               # Detailed guide
```

## Tips

1. **Start simple**: Use `visualize_single_prediction(1)` first
2. **Check data**: Run `inspect_anatomy_data` to understand the structure
3. **Adjust threshold**: Start with `threshold = 0`, then increase to focus on strong activations
4. **Try different views**: Rotate the figure interactively or set `view_angle`
5. **Compare models**: Use `compare_predictions.m` to evaluate different checkpoints
6. **Save figures**: Uncomment save sections or use `saveas(gcf, 'filename.png')`

## References

- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
- MATLAB patch documentation: https://www.mathworks.com/help/matlab/ref/patch.html
- MATLAB visualization: https://www.mathworks.com/help/matlab/graphics.html

## Support

For issues or questions:
1. Check the VISUALIZATION_GUIDE.md for detailed documentation
2. Run `inspect_anatomy_data` to verify data structure
3. Check file paths and ensure all anatomy files exist
4. Verify prediction file was generated correctly by eval_real.py

