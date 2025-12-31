% Visualize specific timepoints in one figure
% Allows custom selection of which timepoints to show
%
% Usage:
%   visualize_selected_timepoints(sample_idx, timepoints)
%   visualize_selected_timepoints(1, [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
%   visualize_selected_timepoints(1, 1:50:500)  % Every 50th timepoint

function visualize_selected_timepoints(sample_idx, timepoints, view_angle)
    if nargin < 1
        sample_idx = 1;
    end
    if nargin < 2
        % Default: show 20 evenly spaced timepoints
        timepoints = round(linspace(1, 500, 20));
    end
    if nargin < 3
        view_angle = [-86, 17];  % Left lateral view
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
    
    % Check shape
    if ndims(all_out) == 2
        fprintf('WARNING: Data has shape (num_samples, 994) - only single timepoint.\n');
        fprintf('Cannot show temporal evolution.\n');
        fprintf('Use visualize_single_prediction instead.\n');
        return;
    elseif ndims(all_out) ~= 3
        error('Expected 3D data (num_samples, timepoints, 994), got: %s', mat2str(size(all_out)));
    end
    
    num_samples = size(all_out, 1);
    total_timepoints = size(all_out, 2);
    
    if sample_idx > num_samples
        error('Sample index %d exceeds number of samples (%d)', sample_idx, num_samples);
    end
    
    % Validate timepoints
    timepoints = timepoints(timepoints >= 1 & timepoints <= total_timepoints);
    num_show = length(timepoints);
    
    fprintf('Showing %d timepoints: %s\n', num_show, mat2str(timepoints));
    
    %% Setup subplot layout
    if num_show <= 4
        subplot_rows = 1;
        subplot_cols = num_show;
    elseif num_show <= 9
        subplot_rows = ceil(sqrt(num_show));
        subplot_cols = ceil(num_show / subplot_rows);
    elseif num_show <= 20
        subplot_rows = 4;
        subplot_cols = 5;
    else
        subplot_rows = 5;
        subplot_cols = ceil(num_show / subplot_rows);
    end
    
    %% Create figure
    figure('Name', sprintf('Selected Timepoints - Sample %d', sample_idx), ...
           'Position', [50, 50, 250*subplot_cols, 250*subplot_rows]);
    
    %% Visualize each timepoint
    for i = 1:num_show
        t_idx = timepoints(i);
        
        % Get region values at this timepoint
        region_values = squeeze(all_out(sample_idx, t_idx, :));
        
        % Convert to vertex level
        vertex_values = zeros(size(pos, 1), 1);
        for vertex_idx = 1:length(region_mapping)
            region_id = region_mapping(vertex_idx);
            if region_id > 0 && region_id <= length(region_values)
                vertex_values(vertex_idx) = region_values(region_id);
            end
        end
        
        % Create subplot
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
        view(view_angle);
        
        % Title
        title(sprintf('t=%d', t_idx), 'FontSize', 9);
    end
    
    % Add main title
    sgtitle(sprintf('Sample %d - Selected Timepoints', sample_idx), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('Done!\n');
    fprintf('Statistics:\n');
    fprintf('  Min activation: %.6f\n', min(all_out(sample_idx, :, :), [], 'all'));
    fprintf('  Max activation: %.6f\n', max(all_out(sample_idx, :, :), [], 'all'));
    fprintf('  Mean activation: %.6f\n', mean(all_out(sample_idx, :, :), 'all'));
end

