% Script to visualize transformer predictions from eval_real.py
% This script loads the prediction results and visualizes them on the cortex
% 
% Usage:
%   1. Make sure you have run eval_real.py and generated the prediction .mat file
%   2. Update the prediction_file path below if needed
%   3. Run this script in MATLAB
%
% The script will:
%   - Load cortex geometry (pos, tri) from anatomy folder
%   - Load region mapping to convert from 994 regions to 20k vertices
%   - Load your transformer predictions
%   - Visualize the predictions on the cortex surface

clear; close all;

%% ========== Configuration ==========
% Path to your prediction file (update this!)
prediction_file = '../source/VEP/transformer_predictions_best_model.mat';

% Path to anatomy files
anatomy_dir = '../anatomy/';
cortex_file = [anatomy_dir 'fs_cortex_20k.mat'];
region_mapping_file = [anatomy_dir 'fs_cortex_20k_region_mapping.mat'];

% Visualization parameters
num_samples_to_show = 5;  % Number of samples to visualize
subplot_rows = 1;         % Subplot layout
subplot_cols = 5;
face_alpha = 0.8;         % Transparency (0=transparent, 1=opaque)
threshold = 0.1;          % Threshold for visualization (0-1)
view_angle = [-86, 17];   % Camera view angle

%% ========== Load Data ==========
fprintf('Loading cortex geometry...\n');
cortex_data = load(cortex_file);
pos = cortex_data.pos;  % Vertex positions (20484 x 3)
tri = cortex_data.tri;  % Triangle faces

fprintf('Cortex: %d vertices, %d triangles\n', size(pos,1), size(tri,1));

fprintf('Loading region mapping...\n');
rm_data = load(region_mapping_file);
region_mapping = rm_data.region_mapping;  % Maps 20k vertices to 994 regions

fprintf('Region mapping: %d vertices -> %d regions\n', length(region_mapping), length(unique(region_mapping)));

fprintf('Loading predictions...\n');
pred_data = load(prediction_file);
all_out = pred_data.all_out;  % Shape: (num_samples, 994)

fprintf('Predictions shape: %d samples x %d regions\n', size(all_out,1), size(all_out,2));

%% ========== Convert Region-Level Predictions to Vertex-Level ==========
fprintf('Converting region predictions to vertex level...\n');

num_samples = size(all_out, 1);
num_vertices = size(pos, 1);
vertex_predictions = zeros(num_samples, num_vertices);

% For each sample, map region values to vertices
for sample_idx = 1:num_samples
    region_values = all_out(sample_idx, :);  % 1 x 994
    
    % Map each region value to all vertices in that region
    for vertex_idx = 1:num_vertices
        region_id = region_mapping(vertex_idx);
        if region_id > 0 && region_id <= length(region_values)
            vertex_predictions(sample_idx, vertex_idx) = region_values(region_id);
        end
    end
end

fprintf('Converted to vertex predictions: %d samples x %d vertices\n', ...
        size(vertex_predictions,1), size(vertex_predictions,2));

%% ========== Visualize Predictions ==========
fprintf('Visualizing predictions...\n');

% Limit number of samples to visualize
num_to_show = min(num_samples_to_show, num_samples);

% Create titles for each subplot
titles = cell(1, num_to_show);
for i = 1:num_to_show
    titles{i} = sprintf('Sample %d', i);
end

% Visualize using the visualize_result function
visualize_result(pos, tri, vertex_predictions(1:num_to_show,:), ...
    'FaceAlpha', face_alpha, ...
    'thre', threshold, ...
    'view', view_angle, ...
    'row', subplot_rows, ...
    'col', subplot_cols, ...
    'titles', titles, ...
    'normalize', 1);

fprintf('Visualization complete!\n');

%% ========== Print Statistics ==========
fprintf('\n========== Prediction Statistics ==========\n');
fprintf('Total samples: %d\n', num_samples);
fprintf('Region-level predictions (994 regions):\n');
fprintf('  Min:  %.6f\n', min(all_out(:)));
fprintf('  Max:  %.6f\n', max(all_out(:)));
fprintf('  Mean: %.6f\n', mean(all_out(:)));
fprintf('  Std:  %.6f\n', std(all_out(:)));

fprintf('\nVertex-level predictions (20k vertices):\n');
fprintf('  Min:  %.6f\n', min(vertex_predictions(:)));
fprintf('  Max:  %.6f\n', max(vertex_predictions(:)));
fprintf('  Mean: %.6f\n', mean(vertex_predictions(:)));
fprintf('  Std:  %.6f\n', std(vertex_predictions(:)));

%% ========== Optional: Save Visualization ==========
% Uncomment the following lines to save the figure
% save_dir = '../results/visualizations/';
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
% saveas(gcf, [save_dir 'transformer_predictions.png']);
% fprintf('Figure saved to: %s\n', [save_dir 'transformer_predictions.png']);

