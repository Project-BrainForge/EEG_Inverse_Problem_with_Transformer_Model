% Quick script to visualize a single prediction
% Usage: visualize_single_prediction(sample_idx)
%
% Example:
%   visualize_single_prediction(1)  % Visualize first sample
%   visualize_single_prediction(3)  % Visualize third sample

function visualize_single_prediction(sample_idx)
    if nargin < 1
        sample_idx = 1;  % Default to first sample
    end
    
    %% Load data
    fprintf('Loading data for sample %d...\n', sample_idx);
    
    % Load cortex
    cortex_data = load('../anatomy/fs_cortex_20k.mat');
    pos = cortex_data.pos;
    tri = cortex_data.tri;
    
    % Load region mapping
    rm_data = load('../anatomy/fs_cortex_20k_region_mapping.mat');
    region_mapping = rm_data.region_mapping;
    
    % Load predictions
    pred_data = load('../source/VEP/transformer_predictions_best_model.mat');
    all_out = pred_data.all_out;
    
    if sample_idx > size(all_out, 1)
        error('Sample index %d exceeds number of samples (%d)', sample_idx, size(all_out, 1));
    end
    
    %% Convert to vertex level
    region_values = all_out(sample_idx, :);
    vertex_values = zeros(size(pos, 1), 1);
    
    for vertex_idx = 1:length(region_mapping)
        region_id = region_mapping(vertex_idx);
        if region_id > 0 && region_id <= length(region_values)
            vertex_values(vertex_idx) = region_values(region_id);
        end
    end
    
    %% Visualize
    figure('Name', sprintf('Transformer Prediction - Sample %d', sample_idx));
    visualize_result(pos, tri, vertex_values', ...
        'FaceAlpha', 0.8, ...
        'thre', 0.1, ...
        'view', [-86, 17], ...
        'normalize', 1);
    
    title(sprintf('Sample %d - Transformer Prediction', sample_idx));
    
    fprintf('Done! Showing sample %d\n', sample_idx);
    fprintf('  Min:  %.6f\n', min(vertex_values));
    fprintf('  Max:  %.6f\n', max(vertex_values));
    fprintf('  Mean: %.6f\n', mean(vertex_values));
end

