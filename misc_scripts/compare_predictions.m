% Script to compare predictions from different models/checkpoints
% This allows you to visualize and compare results from multiple runs
%
% Usage:
%   1. Edit the prediction_files cell array below
%   2. Run this script
%   3. It will show the same sample from different models side by side

clear; close all;

%% ========== Configuration ==========
% List of prediction files to compare
prediction_files = {
    '../source/VEP/transformer_predictions_best_model.mat';
    '../source/VEP/transformer_predictions_data1_uploaded.mat';
    '../source/VEP/transformer_predictions_data2_uploaded.mat';
    % Add more files here...
};

% Which sample to compare (same sample across all models)
sample_idx = 1;

% Visualization parameters
face_alpha = 0.8;
threshold = 0.1;
view_angle = [-86, 17];

%% ========== Load Anatomy Data ==========
fprintf('Loading anatomy data...\n');
anatomy_dir = '../anatomy/';
cortex_data = load([anatomy_dir 'fs_cortex_20k.mat']);
rm_data = load([anatomy_dir 'fs_cortex_20k_region_mapping.mat']);

pos = cortex_data.pos;
tri = cortex_data.tri;
region_mapping = rm_data.region_mapping;

%% ========== Load and Compare Predictions ==========
num_files = length(prediction_files);
fprintf('Comparing %d prediction files...\n', num_files);

% Determine subplot layout
if num_files <= 3
    subplot_rows = 1;
    subplot_cols = num_files;
elseif num_files <= 6
    subplot_rows = 2;
    subplot_cols = 3;
elseif num_files <= 9
    subplot_rows = 3;
    subplot_cols = 3;
else
    subplot_rows = ceil(sqrt(num_files));
    subplot_cols = ceil(num_files / subplot_rows);
end

figure('Name', sprintf('Comparison - Sample %d', sample_idx), ...
       'Position', [100, 100, 400*subplot_cols, 400*subplot_rows]);

all_vertex_predictions = cell(num_files, 1);
titles = cell(num_files, 1);

for file_idx = 1:num_files
    pred_file = prediction_files{file_idx};
    
    % Check if file exists
    if ~exist(pred_file, 'file')
        fprintf('Warning: File not found: %s\n', pred_file);
        continue;
    end
    
    fprintf('Loading: %s\n', pred_file);
    
    % Load predictions
    pred_data = load(pred_file);
    all_out = pred_data.all_out;
    
    if sample_idx > size(all_out, 1)
        fprintf('Warning: Sample %d not available in %s (only %d samples)\n', ...
                sample_idx, pred_file, size(all_out, 1));
        continue;
    end
    
    % Convert to vertex level
    region_values = all_out(sample_idx, :);
    vertex_values = zeros(size(pos, 1), 1);
    
    for vertex_idx = 1:length(region_mapping)
        region_id = region_mapping(vertex_idx);
        if region_id > 0 && region_id <= length(region_values)
            vertex_values(vertex_idx) = region_values(region_id);
        end
    end
    
    all_vertex_predictions{file_idx} = vertex_values;
    
    % Create title from filename
    [~, fname, ~] = fileparts(pred_file);
    fname = strrep(fname, 'transformer_predictions_', '');
    fname = strrep(fname, '_', ' ');
    titles{file_idx} = fname;
    
    % Print statistics
    fprintf('  Min: %.6f, Max: %.6f, Mean: %.6f\n', ...
            min(vertex_values), max(vertex_values), mean(vertex_values));
end

%% ========== Visualize All ==========
fprintf('\nVisualizing...\n');

for file_idx = 1:num_files
    if isempty(all_vertex_predictions{file_idx})
        continue;
    end
    
    subplot(subplot_rows, subplot_cols, file_idx);
    hold on; grid off; axis off;
    
    vertex_values = all_vertex_predictions{file_idx};
    
    % Normalize and threshold
    tmp_value = vertex_values;
    tmp_value(abs(tmp_value) < threshold * max(abs(tmp_value))) = 0;
    tmp_value = tmp_value / max(abs(tmp_value));
    
    % Plot
    hpatch = patch('vertices', pos, 'faces', tri, 'FaceVertexCData', tmp_value);
    set(hpatch, 'EdgeColor', 'none', 'FaceColor', 'interp', ...
        'FaceLighting', 'phong', 'DiffuseStrength', 1, 'FaceAlpha', face_alpha);
    
    % Colormap
    cmap1 = hot(38);
    cmap1 = cmap1(end:-1:1,:);
    mid_tran = gray(64);
    mid = mid_tran(57:58,:);
    cmap = [mid; cmap1(1:31,:)];
    colormap(cmap);
    caxis([0 1]);
    
    % Lighting and view
    light('position', [3,3,1]);
    light('position', [-3,-3,-1]);
    view(view_angle);
    
    title(titles{file_idx}, 'Interpreter', 'none');
end

fprintf('Done!\n');

%% ========== Compute Differences (Optional) ==========
if num_files == 2
    fprintf('\n========== Difference Analysis ==========\n');
    
    diff_values = all_vertex_predictions{1} - all_vertex_predictions{2};
    
    fprintf('Difference statistics:\n');
    fprintf('  Min:  %.6f\n', min(diff_values));
    fprintf('  Max:  %.6f\n', max(diff_values));
    fprintf('  Mean: %.6f\n', mean(diff_values));
    fprintf('  Std:  %.6f\n', std(diff_values));
    fprintf('  RMS:  %.6f\n', sqrt(mean(diff_values.^2)));
    
    % Visualize difference
    figure('Name', 'Difference Map');
    visualize_result(pos, tri, diff_values', ...
        'FaceAlpha', face_alpha, ...
        'thre', 0, ...
        'view', view_angle, ...
        'neg', 1, ...  % Use bipolar colormap
        'normalize', 0);
    title(sprintf('Difference: %s - %s', titles{1}, titles{2}));
    
    % Correlation
    corr_val = corr(all_vertex_predictions{1}, all_vertex_predictions{2});
    fprintf('  Correlation: %.4f\n', corr_val);
end

%% ========== Summary Statistics ==========
fprintf('\n========== Summary ==========\n');
fprintf('Compared %d models on sample %d\n', num_files, sample_idx);
fprintf('Files:\n');
for i = 1:num_files
    fprintf('  %d. %s\n', i, prediction_files{i});
end
fprintf('============================\n');

