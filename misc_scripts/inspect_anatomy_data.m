% Script to inspect anatomy data structure
% This helps you understand what data is available in the anatomy folder

clear; close all;

fprintf('========== Inspecting Anatomy Data ==========\n\n');

anatomy_dir = '../anatomy/';

%% 1. Cortex Geometry
fprintf('1. CORTEX GEOMETRY (fs_cortex_20k.mat)\n');
fprintf('   Loading...\n');
cortex_data = load([anatomy_dir 'fs_cortex_20k.mat']);
fprintf('   Variables in file:\n');
disp(fieldnames(cortex_data));

if isfield(cortex_data, 'pos')
    fprintf('   pos: %d vertices x %d dimensions (x,y,z coordinates)\n', ...
            size(cortex_data.pos, 1), size(cortex_data.pos, 2));
    fprintf('   Range: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]\n', ...
            min(cortex_data.pos(:,1)), max(cortex_data.pos(:,1)), ...
            min(cortex_data.pos(:,2)), max(cortex_data.pos(:,2)), ...
            min(cortex_data.pos(:,3)), max(cortex_data.pos(:,3)));
end

if isfield(cortex_data, 'tri')
    fprintf('   tri: %d triangles x 3 (vertex indices for each triangle)\n', ...
            size(cortex_data.tri, 1));
end

%% 2. Region Mapping
fprintf('\n2. REGION MAPPING (fs_cortex_20k_region_mapping.mat)\n');
fprintf('   Loading...\n');
rm_data = load([anatomy_dir 'fs_cortex_20k_region_mapping.mat']);
fprintf('   Variables in file:\n');
disp(fieldnames(rm_data));

if isfield(rm_data, 'region_mapping')
    fprintf('   region_mapping: %d vertices\n', length(rm_data.region_mapping));
    fprintf('   Maps to %d unique regions\n', length(unique(rm_data.region_mapping)));
    fprintf('   Region IDs range: [%d, %d]\n', ...
            min(rm_data.region_mapping), max(rm_data.region_mapping));
    
    % Count vertices per region
    region_counts = histcounts(rm_data.region_mapping, ...
                               max(rm_data.region_mapping));
    fprintf('   Vertices per region: min=%d, max=%d, mean=%.1f\n', ...
            min(region_counts), max(region_counts), mean(region_counts));
end

%% 3. Leadfield Matrix
fprintf('\n3. LEADFIELD MATRIX (leadfield_75_20k.mat)\n');
if exist([anatomy_dir 'leadfield_75_20k.mat'], 'file')
    fprintf('   Loading...\n');
    lf_data = load([anatomy_dir 'leadfield_75_20k.mat']);
    fprintf('   Variables in file:\n');
    disp(fieldnames(lf_data));
    
    if isfield(lf_data, 'L')
        fprintf('   L: %d x %d (EEG channels x sources)\n', ...
                size(lf_data.L, 1), size(lf_data.L, 2));
    end
    if isfield(lf_data, 'lf_cortex_20k')
        fprintf('   lf_cortex_20k: %d x %d\n', ...
                size(lf_data.lf_cortex_20k, 1), size(lf_data.lf_cortex_20k, 2));
    end
else
    fprintf('   File not found.\n');
end

%% 4. Electrode Positions
fprintf('\n4. ELECTRODE POSITIONS (electrode_75.mat)\n');
if exist([anatomy_dir 'electrode_75.mat'], 'file')
    fprintf('   Loading...\n');
    elec_data = load([anatomy_dir 'electrode_75.mat']);
    fprintf('   Variables in file:\n');
    disp(fieldnames(elec_data));
    
    if isfield(elec_data, 'pos')
        fprintf('   pos: %d electrodes x %d dimensions\n', ...
                size(elec_data.pos, 1), size(elec_data.pos, 2));
    end
    if isfield(elec_data, 'label')
        fprintf('   label: %d electrode names\n', length(elec_data.label));
        fprintf('   First 5 labels: ');
        for i = 1:min(5, length(elec_data.label))
            fprintf('%s ', elec_data.label{i});
        end
        fprintf('\n');
    end
else
    fprintf('   File not found.\n');
end

%% 5. Visualize Cortex
fprintf('\n5. VISUALIZING CORTEX\n');
fprintf('   Creating figure...\n');

figure('Name', 'Cortex Anatomy Overview');

% Plot 1: Full cortex
subplot(2,2,1);
trisurf(cortex_data.tri, cortex_data.pos(:,1), cortex_data.pos(:,2), cortex_data.pos(:,3), ...
        'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal; axis off;
light('position', [3,3,1]);
light('position', [-3,-3,-1]);
view([-86, 17]);
title('Left View');

subplot(2,2,2);
trisurf(cortex_data.tri, cortex_data.pos(:,1), cortex_data.pos(:,2), cortex_data.pos(:,3), ...
        'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal; axis off;
light('position', [3,3,1]);
light('position', [-3,-3,-1]);
view([86, 17]);
title('Right View');

subplot(2,2,3);
trisurf(cortex_data.tri, cortex_data.pos(:,1), cortex_data.pos(:,2), cortex_data.pos(:,3), ...
        'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal; axis off;
light('position', [3,3,1]);
light('position', [-3,-3,-1]);
view([0, 90]);
title('Top View');

% Plot 4: Region mapping visualization
subplot(2,2,4);
trisurf(cortex_data.tri, cortex_data.pos(:,1), cortex_data.pos(:,2), cortex_data.pos(:,3), ...
        rm_data.region_mapping, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal; axis off;
colormap(jet);
colorbar;
light('position', [3,3,1]);
light('position', [-3,-3,-1]);
view([-86, 17]);
title('Region Mapping (994 regions)');

fprintf('   Done!\n');

%% Summary
fprintf('\n========== SUMMARY ==========\n');
fprintf('Your transformer model:\n');
fprintf('  - Takes input: 75 EEG channels\n');
fprintf('  - Produces output: 994 region activations\n');
fprintf('\nFor visualization:\n');
fprintf('  - Use region_mapping to convert 994 regions -> 20,484 vertices\n');
fprintf('  - Use pos and tri to render the 3D cortex surface\n');
fprintf('  - Each vertex gets the value of its corresponding region\n');
fprintf('\nNext steps:\n');
fprintf('  1. Run: visualize_transformer_predictions\n');
fprintf('  2. Or: visualize_single_prediction(1)\n');
fprintf('=============================\n');

