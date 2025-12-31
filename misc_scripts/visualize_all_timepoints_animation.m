% Create animation showing all 500 timepoints
% This creates a video file showing temporal evolution
%
% Usage:
%   visualize_all_timepoints_animation(sample_idx)
%   visualize_all_timepoints_animation(1)  % Animate sample 1

function visualize_all_timepoints_animation(sample_idx, output_file)
    if nargin < 1
        sample_idx = 1;
    end
    if nargin < 2
        output_file = sprintf('../results/temporal_evolution_sample_%d.mp4', sample_idx);
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
    
    % Check shape
    if ndims(all_out) ~= 3
        error('Expected 3D data (num_samples, 500, 994), got shape: %s', mat2str(size(all_out)));
    end
    
    num_samples = size(all_out, 1);
    num_timepoints = size(all_out, 2);
    num_regions = size(all_out, 3);
    
    if sample_idx > num_samples
        error('Sample index %d exceeds number of samples (%d)', sample_idx, num_samples);
    end
    
    fprintf('Creating animation for %d timepoints...\n', num_timepoints);
    
    %% Setup video writer
    [output_dir, ~, ~] = fileparts(output_file);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    v = VideoWriter(output_file, 'MPEG-4');
    v.FrameRate = 30;  % 30 frames per second
    v.Quality = 95;
    open(v);
    
    %% Create figure
    fig = figure('Position', [100, 100, 800, 600], 'Color', 'white');
    
    % Find global min/max for consistent colormap
    sample_data = squeeze(all_out(sample_idx, :, :));  % (500, 994)
    global_max = 0;
    
    fprintf('Computing global maximum for consistent scaling...\n');
    for t = 1:num_timepoints
        region_values = sample_data(t, :);
        vertex_values = zeros(size(pos, 1), 1);
        for vertex_idx = 1:length(region_mapping)
            region_id = region_mapping(vertex_idx);
            if region_id > 0 && region_id <= length(region_values)
                vertex_values(vertex_idx) = region_values(region_id);
            end
        end
        global_max = max(global_max, max(abs(vertex_values)));
    end
    
    %% Generate frames
    fprintf('Generating frames...\n');
    for t = 1:num_timepoints
        if mod(t, 50) == 0
            fprintf('  Frame %d/%d\n', t, num_timepoints);
        end
        
        % Get region values at this timepoint
        region_values = sample_data(t, :);
        
        % Convert to vertex level
        vertex_values = zeros(size(pos, 1), 1);
        for vertex_idx = 1:length(region_mapping)
            region_id = region_mapping(vertex_idx);
            if region_id > 0 && region_id <= length(region_values)
                vertex_values(vertex_idx) = region_values(region_id);
            end
        end
        
        % Clear and redraw
        clf(fig);
        hold on; grid off; axis off;
        
        % Normalize
        if global_max > 0
            tmp_value = vertex_values / global_max;
        else
            tmp_value = vertex_values;
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
        
        % Title with timepoint
        title(sprintf('Sample %d - Timepoint %d/%d (%.1f ms)', ...
                      sample_idx, t, num_timepoints, t*2), ...  % Assuming 2ms sampling
              'FontSize', 14, 'FontWeight', 'bold');
        
        % Capture frame
        frame = getframe(fig);
        writeVideo(v, frame);
    end
    
    %% Close video
    close(v);
    close(fig);
    
    fprintf('Animation complete!\n');
    fprintf('Saved to: %s\n', output_file);
    fprintf('Duration: %.1f seconds at %d fps\n', num_timepoints/v.FrameRate, v.FrameRate);
end

