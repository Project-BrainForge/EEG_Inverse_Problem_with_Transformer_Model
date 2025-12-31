% Script to visualize multiple timepoints from predictions
% Shows temporal evolution of brain activity
%
% Usage:
%   visualize_timepoints(sample_idx, num_timepoints)
%   visualize_timepoints(1, 16)  % Show 16 timepoints from sample 1

function visualize_timepoints(sample_idx, num_timepoints)
    if nargin < 1
        sample_idx = 1;  % Default to first sample
    end
    if nargin < 2
        num_timepoints = 16;  % Default to 16 timepoints
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
    
    fprintf('Prediction shape: %s\n', mat2str(size(all_out)));
    
    % Check if we have temporal data
    if ndims(all_out) == 3
        % Shape: (num_samples, 500, 994)
        fprintf('Temporal data detected!\n');
        num_samples = size(all_out, 1);
        total_timepoints = size(all_out, 2);
        num_regions = size(all_out, 3);
        
        if sample_idx > num_samples
            error('Sample index %d exceeds number of samples (%d)', sample_idx, num_samples);
        end
        
        % Select timepoints to show
        timepoint_indices = round(linspace(1, total_timepoints, num_timepoints));
        
    elseif ndims(all_out) == 2
        % Shape: (num_samples, 994) - single timepoint
        fprintf('Single timepoint data detected.\n');
        fprintf('Cannot show temporal evolution. Use visualize_single_prediction instead.\n');
        return;
    else
        error('Unexpected data shape: %s', mat2str(size(all_out)));
    end
    
    %% Setup subplot layout
    subplot_rows = floor(sqrt(num_timepoints));
    subplot_cols = ceil(num_timepoints / subplot_rows);
    
    fprintf('Creating %dx%d subplot grid for %d timepoints...\n', ...
            subplot_rows, subplot_cols, num_timepoints);
    
    %% Create figure
    figure('Name', sprintf('Temporal Evolution - Sample %d', sample_idx), ...
           'Position', [50, 50, 300*subplot_cols, 300*subplot_rows]);
    
    %% Process and visualize each timepoint
    for i = 1:num_timepoints
        t_idx = timepoint_indices(i);
        
        % Get region values at this timepoint
        region_values = squeeze(all_out(sample_idx, t_idx, :));  % Shape: (994,)
        
        % Convert to vertex level
        vertex_values = zeros(size(pos, 1), 1);
        for vertex_idx = 1:length(region_mapping)
            region_id = region_mapping(vertex_idx);
            if region_id > 0 && region_id <= length(region_values)
                vertex_values(vertex_idx) = region_values(region_id);
            end
        end
        
        % Visualize this timepoint
        subplot(subplot_rows, subplot_cols, i);
        hold on; grid off; axis off;
        
        % Normalize and threshold
        tmp_value = vertex_values;
        threshold = 0.1;
        tmp_value(abs(tmp_value) < threshold * max(abs(tmp_value))) = 0;
        if max(abs(tmp_value)) > 0
            tmp_value = tmp_value / max(abs(tmp_value));
        end
        
        % Plot
        hpatch = patch('vertices', pos, 'faces', tri, 'FaceVertexCData', tmp_value);
        set(hpatch, 'EdgeColor', 'none', 'FaceColor', 'interp', ...
            'FaceLighting', 'phong', 'DiffuseStrength', 1, 'FaceAlpha', 0.8);
        
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
        view([-86, 17]);
        
        title(sprintf('t=%d ms', t_idx), 'FontSize', 10);
    end
    
    % Add main title
    sgtitle(sprintf('Sample %d - Temporal Evolution (%d timepoints)', ...
                    sample_idx, num_timepoints), 'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('Done!\n');
    fprintf('Showing %d timepoints: %s\n', num_timepoints, mat2str(timepoint_indices));
end

