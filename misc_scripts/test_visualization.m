% Test script to verify visualization setup
% Run this to check if all required files are present and working

clear; close all;

fprintf('========================================\n');
fprintf('Testing Visualization Setup\n');
fprintf('========================================\n\n');

test_passed = true;

%% Test 1: Check anatomy files
fprintf('Test 1: Checking anatomy files...\n');

required_files = {
    '../anatomy/fs_cortex_20k.mat';
    '../anatomy/fs_cortex_20k_region_mapping.mat';
};

for i = 1:length(required_files)
    if exist(required_files{i}, 'file')
        fprintf('  ✓ Found: %s\n', required_files{i});
    else
        fprintf('  ✗ MISSING: %s\n', required_files{i});
        test_passed = false;
    end
end

%% Test 2: Load anatomy data
fprintf('\nTest 2: Loading anatomy data...\n');

try
    cortex_data = load('../anatomy/fs_cortex_20k.mat');
    fprintf('  ✓ Loaded cortex data\n');
    
    if isfield(cortex_data, 'pos') && isfield(cortex_data, 'tri')
        fprintf('    - pos: %d vertices\n', size(cortex_data.pos, 1));
        fprintf('    - tri: %d triangles\n', size(cortex_data.tri, 1));
    else
        fprintf('  ✗ Missing pos or tri fields\n');
        test_passed = false;
    end
catch ME
    fprintf('  ✗ Error loading cortex: %s\n', ME.message);
    test_passed = false;
end

try
    rm_data = load('../anatomy/fs_cortex_20k_region_mapping.mat');
    fprintf('  ✓ Loaded region mapping\n');
    
    if isfield(rm_data, 'region_mapping')
        fprintf('    - %d vertices -> %d regions\n', ...
                length(rm_data.region_mapping), ...
                length(unique(rm_data.region_mapping)));
    else
        fprintf('  ✗ Missing region_mapping field\n');
        test_passed = false;
    end
catch ME
    fprintf('  ✗ Error loading region mapping: %s\n', ME.message);
    test_passed = false;
end

%% Test 3: Check prediction files
fprintf('\nTest 3: Checking prediction files...\n');

pred_dir = '../source/VEP/';
pred_files = dir([pred_dir 'transformer_predictions_*.mat']);

if isempty(pred_files)
    fprintf('  ✗ No prediction files found in %s\n', pred_dir);
    fprintf('    Run eval_real.py first to generate predictions!\n');
    test_passed = false;
else
    fprintf('  ✓ Found %d prediction file(s):\n', length(pred_files));
    for i = 1:length(pred_files)
        fprintf('    - %s\n', pred_files(i).name);
    end
    
    % Try loading the first one
    try
        pred_data = load(fullfile(pred_files(1).folder, pred_files(1).name));
        if isfield(pred_data, 'all_out')
            fprintf('  ✓ Loaded predictions: %d samples x %d regions\n', ...
                    size(pred_data.all_out, 1), size(pred_data.all_out, 2));
            
            % Check dimensions
            if size(pred_data.all_out, 2) == 994
                fprintf('  ✓ Correct number of regions (994)\n');
            else
                fprintf('  ✗ Unexpected number of regions: %d (expected 994)\n', ...
                        size(pred_data.all_out, 2));
                test_passed = false;
            end
        else
            fprintf('  ✗ Missing all_out field in prediction file\n');
            test_passed = false;
        end
    catch ME
        fprintf('  ✗ Error loading predictions: %s\n', ME.message);
        test_passed = false;
    end
end

%% Test 4: Check visualize_result function
fprintf('\nTest 4: Checking visualize_result function...\n');

if exist('visualize_result.m', 'file')
    fprintf('  ✓ Found visualize_result.m\n');
else
    fprintf('  ✗ Cannot find visualize_result.m\n');
    fprintf('    Make sure you are in the misc_scripts directory\n');
    test_passed = false;
end

%% Test 5: Try simple visualization
fprintf('\nTest 5: Testing simple visualization...\n');

if test_passed
    try
        % Create simple test data
        test_values = zeros(1, size(cortex_data.pos, 1));
        test_values(1:100) = 1;  % Activate first 100 vertices
        
        figure('Visible', 'off');  % Don't show window
        visualize_result(cortex_data.pos, cortex_data.tri, test_values, ...
            'FaceAlpha', 0.8, 'thre', 0);
        close(gcf);
        
        fprintf('  ✓ Visualization test passed\n');
    catch ME
        fprintf('  ✗ Visualization test failed: %s\n', ME.message);
        test_passed = false;
    end
else
    fprintf('  ⊘ Skipped (previous tests failed)\n');
end

%% Test 6: Test region mapping conversion
fprintf('\nTest 6: Testing region mapping conversion...\n');

if test_passed && exist('pred_data', 'var')
    try
        % Convert first sample from regions to vertices
        region_values = pred_data.all_out(1, :);
        vertex_values = zeros(size(cortex_data.pos, 1), 1);
        
        for vertex_idx = 1:length(rm_data.region_mapping)
            region_id = rm_data.region_mapping(vertex_idx);
            if region_id > 0 && region_id <= length(region_values)
                vertex_values(vertex_idx) = region_values(region_id);
            end
        end
        
        fprintf('  ✓ Region-to-vertex conversion successful\n');
        fprintf('    - Input: 994 regions\n');
        fprintf('    - Output: %d vertices\n', length(vertex_values));
        fprintf('    - Non-zero vertices: %d\n', sum(vertex_values ~= 0));
    catch ME
        fprintf('  ✗ Conversion test failed: %s\n', ME.message);
        test_passed = false;
    end
else
    fprintf('  ⊘ Skipped (previous tests failed)\n');
end

%% Summary
fprintf('\n========================================\n');
if test_passed
    fprintf('✓ ALL TESTS PASSED!\n');
    fprintf('========================================\n\n');
    fprintf('You are ready to visualize your predictions!\n\n');
    fprintf('Next steps:\n');
    fprintf('  1. Run: visualize_transformer_predictions\n');
    fprintf('  2. Or:  visualize_single_prediction(1)\n');
    fprintf('  3. Or:  compare_predictions\n\n');
else
    fprintf('✗ SOME TESTS FAILED\n');
    fprintf('========================================\n\n');
    fprintf('Please fix the issues above before proceeding.\n\n');
    fprintf('Common solutions:\n');
    fprintf('  - Make sure you are in the misc_scripts directory\n');
    fprintf('  - Check that anatomy files exist in ../anatomy/\n');
    fprintf('  - Run eval_real.py to generate prediction files\n');
    fprintf('  - Verify file paths are correct\n\n');
end

fprintf('For more help, see:\n');
fprintf('  - VISUALIZATION_GUIDE.md\n');
fprintf('  - README_VISUALIZATION.md\n');
fprintf('  - VISUALIZATION_QUICKSTART.txt\n\n');

