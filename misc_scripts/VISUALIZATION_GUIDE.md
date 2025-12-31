# Visualization Guide for Transformer Predictions

This guide explains how to visualize the predictions from `eval_real.py` using MATLAB.

## Prerequisites

1. You have run `eval_real.py` and generated a `.mat` file with predictions (e.g., `transformer_predictions_best_model.mat`)
2. MATLAB is installed
3. The `anatomy` folder contains the required files:
   - `fs_cortex_20k.mat` - Cortex geometry (vertices and triangles)
   - `fs_cortex_20k_region_mapping.mat` - Mapping from 20k vertices to 994 regions

## Quick Start

### Option 1: Visualize Multiple Samples (Recommended)

```matlab
cd misc_scripts
visualize_transformer_predictions
```

This will:
- Load your predictions from `source/VEP/transformer_predictions_best_model.mat`
- Convert region-level predictions (994 regions) to vertex-level (20k vertices)
- Display the first 5 samples in a subplot
- Show statistics

### Option 2: Visualize a Single Sample

```matlab
cd misc_scripts
visualize_single_prediction(1)  % Visualize sample 1
visualize_single_prediction(3)  % Visualize sample 3
```

## Understanding the Data Flow

```
eval_real.py output:
  all_out: (num_samples, 994)  <- Region-level predictions

        ↓ (region_mapping conversion)

Vertex-level predictions:
  vertex_predictions: (num_samples, 20484)  <- For visualization

        ↓ (visualize_result.m)

3D Brain Visualization
```

## Customization

### Change the Prediction File

Edit `visualize_transformer_predictions.m` line 18:

```matlab
prediction_file = '../source/VEP/transformer_predictions_YOUR_FILE.mat';
```

### Adjust Visualization Parameters

In `visualize_transformer_predictions.m`, modify:

```matlab
num_samples_to_show = 10;  % Show more samples
subplot_rows = 2;          % 2 rows
subplot_cols = 5;          % 5 columns
face_alpha = 0.9;          % Less transparent
threshold = 0.2;           % Higher threshold (only show strong activations)
view_angle = [90, 0];      % Different view angle
```

### View Angles

Common view angles:
- `[-86, 17]` - Left lateral view (default)
- `[86, 17]` - Right lateral view
- `[0, 90]` - Top view
- `[0, 0]` - Front view
- `[180, 0]` - Back view

### Multiple Views of Same Sample

```matlab
% Load data once
cortex_data = load('../anatomy/fs_cortex_20k.mat');
rm_data = load('../anatomy/fs_cortex_20k_region_mapping.mat');
pred_data = load('../source/VEP/transformer_predictions_best_model.mat');

% Convert to vertex level (for sample 1)
vertex_values = zeros(size(cortex_data.pos, 1), 1);
for i = 1:length(rm_data.region_mapping)
    region_id = rm_data.region_mapping(i);
    if region_id > 0
        vertex_values(i) = pred_data.all_out(1, region_id);
    end
end

% Show multiple views
figure;
subplot(2,2,1);
visualize_result(cortex_data.pos, cortex_data.tri, vertex_values', ...
    'view', [-86, 17], 'new_fig', 0);
title('Left');

subplot(2,2,2);
visualize_result(cortex_data.pos, cortex_data.tri, vertex_values', ...
    'view', [86, 17], 'new_fig', 0);
title('Right');

subplot(2,2,3);
visualize_result(cortex_data.pos, cortex_data.tri, vertex_values', ...
    'view', [0, 90], 'new_fig', 0);
title('Top');

subplot(2,2,4);
visualize_result(cortex_data.pos, cortex_data.tri, vertex_values', ...
    'view', [0, 0], 'new_fig', 0);
title('Front');
```

## Saving Figures

Uncomment the save section at the end of `visualize_transformer_predictions.m`:

```matlab
save_dir = '../results/visualizations/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
saveas(gcf, [save_dir 'transformer_predictions.png']);
```

Or save manually in MATLAB:
```matlab
saveas(gcf, 'my_visualization.png');
saveas(gcf, 'my_visualization.fig');  % Save as MATLAB figure
```

## Troubleshooting

### Error: "Cannot find file"
- Make sure you're in the `misc_scripts` directory when running
- Check that the paths in the script match your file locations

### Error: "Undefined function 'visualize_result'"
- Make sure you're in the `misc_scripts` directory
- Or add it to path: `addpath('misc_scripts')`

### Predictions look strange
- Check the normalization: try setting `'normalize', 0` or `'normalize', 1`
- Adjust threshold: try `'thre', 0` to see all values
- Check your prediction statistics (printed by the script)

### Want to compare with ground truth
If you have ground truth source locations, you can overlay them:

```matlab
% Assuming you have ground truth region indices
gt_regions = [100, 101, 102];  % Example regions

visualize_result(pos, tri, vertex_values', ...
    'source', {gt_regions}, ...
    'SourceFaceAlpha', 0.5);
```

## Understanding the Region Mapping

The `region_mapping` array maps each of the 20,484 cortex vertices to one of 994 regions:

```matlab
region_mapping(vertex_idx) = region_id
```

For example:
- `region_mapping(1) = 5` means vertex 1 belongs to region 5
- `region_mapping(2) = 5` means vertex 2 also belongs to region 5
- `region_mapping(100) = 12` means vertex 100 belongs to region 12

Your transformer outputs predictions for 994 regions, and we use this mapping to assign those region values to all vertices in each region for visualization.

## Advanced: Batch Processing Multiple Files

```matlab
% Process all prediction files
pred_files = dir('../source/VEP/transformer_predictions_*.mat');

for i = 1:length(pred_files)
    fprintf('Processing %s...\n', pred_files(i).name);
    
    % Load and visualize
    pred_data = load(fullfile(pred_files(i).folder, pred_files(i).name));
    % ... conversion and visualization code ...
    
    % Save with unique name
    saveas(gcf, sprintf('visualization_%d.png', i));
    close(gcf);
end
```

## Files Overview

- `visualize_result.m` - Core visualization function (already existed)
- `visualize_transformer_predictions.m` - Main script to visualize multiple samples
- `visualize_single_prediction.m` - Quick function to visualize one sample
- `VISUALIZATION_GUIDE.md` - This guide

## Contact

If you encounter issues, check:
1. File paths are correct
2. All required .mat files exist in anatomy folder
3. Prediction file was generated successfully by eval_real.py

