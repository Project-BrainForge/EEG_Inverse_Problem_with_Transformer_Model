# 📊 Model Evaluation Guide

## Overview

This guide explains how to evaluate your trained transformer model on both **simulated** and **real** EEG data, generating `.mat` output files with predictions (`all_out` values).

## Available Evaluation Scripts

### 1. `eval_sim.py` - Evaluate on Simulated Test Data
Evaluates the model on test data with ground truth, calculates metrics, and saves predictions.

### 2. `eval_real.py` - Evaluate on Real EEG Data
Evaluates the model on real EEG recordings (no ground truth), outputs source localization predictions.

---

## 🚀 Quick Start

### Evaluate on Simulated Test Data

```bash
python eval_sim.py \
    --checkpoint checkpoints/best_model.pt \
    --test_metadata source/test_sample_source1.mat \
    --batch_size 32
```

### Evaluate on Real EEG Data

```bash
python eval_real.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir source \
    --subjects VEP \
    --device cpu
```

---

## 📖 Detailed Usage

### eval_sim.py - Simulated Data Evaluation

#### Basic Usage

```bash
python eval_sim.py --checkpoint checkpoints/best_model.pt
```

#### Full Command Options

```bash
python eval_sim.py \
    --checkpoint checkpoints/best_model.pt \
    --test_metadata source/test_sample_source1.mat \
    --nmm_spikes_dir source/nmm_spikes \
    --fwd_matrix anatomy/leadfield_75_20k.mat \
    --batch_size 32 \
    --workers 4 \
    --device cuda:0 \
    --dataset_len 1000 \
    --output my_eval_results.mat \
    --save_full \
    --save_eeg \
    --save_regions
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | `checkpoints/best_model.pt` | Path to trained model checkpoint |
| `--test_metadata` | `source/test_sample_source1.mat` | Test metadata file |
| `--nmm_spikes_dir` | `source/nmm_spikes` | NMM spikes directory |
| `--fwd_matrix` | `anatomy/leadfield_75_20k.mat` | Forward matrix file |
| `--batch_size` | `32` | Batch size for evaluation |
| `--workers` | `4` | Number of data loading workers |
| `--device` | `cuda:0` | Device (cuda:0, cpu, etc.) |
| `--dataset_len` | `None` | Number of samples (None = all) |
| `--output` | `auto` | Output filename |
| `--save_full` | `False` | Save full temporal predictions |
| `--save_eeg` | `False` | Save input EEG data |
| `--save_regions` | `False` | Identify active regions |

#### Output

Creates a `.mat` file in the checkpoint directory with:

```matlab
all_out: [num_samples × regions]         % Model predictions (temporal mean)
all_nmm: [num_samples × regions]         % Ground truth
all_regions: [num_samples]               % Identified active regions (if --save_regions)
avg_loss: scalar                         % Average MSE loss
```

If `--save_full` is used:
```matlab
all_out: [num_samples × time × regions]  % Full temporal predictions
all_nmm: [num_samples × time × regions]  % Full ground truth
```

#### Example Output

```bash
$ python eval_sim.py --checkpoint checkpoints/best_model.pt

Using device: cuda:0

Loading forward matrix from anatomy/leadfield_75_20k.mat...
Found forward matrix with key 'fwd', shape: (75, 994)

Loading test data from source/test_sample_source1.mat...
Dataset length: 50000
Test dataset size: 50000
Number of batches: 1563

=> Loading checkpoint from checkpoints/best_model.pt
=> Loaded checkpoint from epoch 89
   Train loss: 0.002341
   Val loss: 0.002567

Number of parameters: 15,234,567

======================================================================
Starting Evaluation
======================================================================
Processed 10/1563 batches
Processed 20/1563 batches
...
Processed 1563/1563 batches

Average MSE Loss: 0.002598

Saving results to checkpoints/eval_results_epoch_89.mat...
Evaluation complete!

Total evaluation time: 342.15s
```

---

### eval_real.py - Real Data Evaluation

#### Basic Usage

```bash
python eval_real.py \
    --checkpoint checkpoints/best_model.pt \
    --subjects VEP
```

#### Full Command Options

```bash
python eval_real.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir source \
    --subjects VEP AEP Patient1 Patient2 \
    --file_pattern "data*.mat" \
    --batch_size 32 \
    --device cpu \
    --normalize
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | `checkpoints/best_model.pt` | Path to trained model |
| `--data_dir` | `source` | Directory containing subject folders |
| `--subjects` | `['VEP']` | List of subject folders to process |
| `--file_pattern` | `data*.mat` | Pattern for data files |
| `--batch_size` | `32` | Batch size for inference |
| `--device` | `cpu` | Device (cuda:0, cpu, etc.) |
| `--normalize` | `True` | Normalize input data |

#### Data Structure Expected

```
source/
├── VEP/
│   ├── data1.mat
│   ├── data2.mat
│   └── data3.mat
├── AEP/
│   ├── data1.mat
│   └── data2.mat
└── Patient1/
    └── data1.mat
```

Each `.mat` file should contain:
- `data` or `eeg_data`: EEG recordings with shape `(500, 75)` (time × channels)

#### Output

Creates a `.mat` file in each subject folder:

```matlab
all_out: [num_files × time × regions]    % Source predictions
file_names: [num_files]                  % Original file names
checkpoint: string                       % Checkpoint path used
inference_time: scalar                   % Time taken (seconds)
num_samples: scalar                      % Number of samples processed
```

Filename format: `transformer_predictions_<checkpoint_name>.mat`

#### Example Output

```bash
$ python eval_real.py --checkpoint checkpoints/best_model.pt --subjects VEP

Using device: cpu

=> Loading checkpoint from checkpoints/best_model.pt
=> Loaded checkpoint from epoch 89
   Train loss: 0.002341
   Val loss: 0.002567

Number of parameters: 15,234,567
Preparation time: 2.34s

======================================================================
Processing subject: VEP
======================================================================
Found 25 files
Loaded 10/25 files
Loaded 20/25 files
Successfully loaded 25 files
Data tensor shape: torch.Size([25, 500, 75])
Running inference...
Inference complete: 3.21s
Output shape: (25, 500, 994)
Saved predictions to: source/VEP/transformer_predictions_best_model.mat

Prediction statistics:
  Min: -0.012345
  Max: 0.987654
  Mean: 0.123456
  Std: 0.234567

======================================================================
Total processing time: 8.67s
======================================================================
```

---

## 📁 Output File Structure

### Simulated Data Output (`eval_sim.py`)

```matlab
% Load results
results = load('checkpoints/eval_results_epoch_89.mat');

% Access predictions
predictions = results.all_out;      % (num_samples, regions) or (num_samples, time, regions)
ground_truth = results.all_nmm;     % Same shape as predictions
avg_loss = results.avg_loss;        % Scalar

% If --save_regions was used
active_regions = results.all_regions;  % Cell array of identified regions
```

### Real Data Output (`eval_real.py`)

```matlab
% Load results
results = load('source/VEP/transformer_predictions_best_model.mat');

% Access predictions
predictions = results.all_out;      % (num_files, time, regions)
file_names = results.file_names;    % Cell array of file names
checkpoint = results.checkpoint;    % String
```

---

## 🔧 Common Use Cases

### Use Case 1: Evaluate Best Model on Test Set

```bash
# After training completes
python eval_sim.py \
    --checkpoint checkpoints/best_model.pt \
    --save_full \
    --save_regions
```

### Use Case 2: Evaluate Multiple Checkpoints

```bash
# Evaluate checkpoint from epoch 50
python eval_sim.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pt \
    --output eval_epoch_50.mat

# Evaluate checkpoint from epoch 100
python eval_sim.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pt \
    --output eval_epoch_100.mat

# Compare results in MATLAB
```

### Use Case 3: Process Real Patient Data

```bash
# Single patient
python eval_real.py \
    --checkpoint checkpoints/best_model.pt \
    --subjects Patient1 \
    --device cuda:0

# Multiple patients
python eval_real.py \
    --checkpoint checkpoints/best_model.pt \
    --subjects Patient1 Patient2 Patient3 VEP AEP \
    --device cuda:0
```

### Use Case 4: Quick Test on Subset

```bash
# Test on first 100 samples
python eval_sim.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset_len 100 \
    --batch_size 10
```

### Use Case 5: CPU-Only Evaluation

```bash
# For systems without GPU
python eval_sim.py \
    --checkpoint checkpoints/best_model.pt \
    --device cpu \
    --batch_size 8 \
    --workers 0
```

---

## 📊 Analyzing Results in MATLAB

### Load and Visualize Predictions

```matlab
% Load results
results = load('checkpoints/eval_results_epoch_89.mat');

% Get predictions and ground truth
pred = results.all_out;  % (samples, time, regions) or (samples, regions)
gt = results.all_nmm;

% Visualize sample 1
sample_idx = 1;

if ndims(pred) == 3
    % Full temporal data
    figure;
    subplot(2,1,1);
    imagesc(pred(sample_idx, :, :)');
    title('Predictions');
    xlabel('Time'); ylabel('Regions');
    colorbar;
    
    subplot(2,1,2);
    imagesc(gt(sample_idx, :, :)');
    title('Ground Truth');
    xlabel('Time'); ylabel('Regions');
    colorbar;
else
    % Temporal mean
    figure;
    subplot(2,1,1);
    plot(pred(sample_idx, :));
    title('Predictions (Temporal Mean)');
    xlabel('Regions'); ylabel('Activity');
    
    subplot(2,1,2);
    plot(gt(sample_idx, :));
    title('Ground Truth (Temporal Mean)');
    xlabel('Regions'); ylabel('Activity');
end
```

### Calculate Metrics

```matlab
% Load results
results = load('checkpoints/eval_results_epoch_89.mat');

pred = results.all_out;
gt = results.all_nmm;

% If temporal data, take mean
if ndims(pred) == 3
    pred_mean = squeeze(mean(pred, 2));
    gt_mean = squeeze(mean(gt, 2));
else
    pred_mean = pred;
    gt_mean = gt;
end

% Calculate correlation
correlations = zeros(size(pred_mean, 1), 1);
for i = 1:size(pred_mean, 1)
    correlations(i) = corr(pred_mean(i, :)', gt_mean(i, :)');
end

fprintf('Mean correlation: %.4f\n', mean(correlations));
fprintf('Median correlation: %.4f\n', median(correlations));

% Calculate MSE
mse = mean((pred_mean - gt_mean).^2, 'all');
fprintf('MSE: %.6f\n', mse);
```

---

## ⚡ Performance Tips

### For Faster Evaluation

```bash
# Use GPU
--device cuda:0

# Larger batch size (if memory allows)
--batch_size 64

# More workers
--workers 8

# Don't save full temporal data
# (omit --save_full flag)
```

### For Memory-Constrained Systems

```bash
# Smaller batch size
--batch_size 8

# Fewer workers
--workers 2

# Use CPU
--device cpu

# Don't save EEG data
# (omit --save_eeg flag)

# Process subset
--dataset_len 1000
```

---

## 🔍 Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch_size 4
```

**Solution 2**: Use CPU
```bash
--device cpu
```

**Solution 3**: Don't save full temporal data
```bash
# Omit --save_full flag
```

### Issue: Checkpoint Not Found

**Solution**: Verify checkpoint path
```bash
ls checkpoints/
python eval_sim.py --checkpoint checkpoints/best_model.pt
```

### Issue: Data Files Not Found

**For eval_sim.py**:
```bash
# Check metadata file exists
ls source/test_sample_source1.mat
```

**For eval_real.py**:
```bash
# Check subject folder and files
ls source/VEP/
ls source/VEP/data*.mat
```

### Issue: Wrong Data Shape

**For real data**: Ensure each `.mat` file contains `data` field with shape `(500, 75)`

```matlab
% In MATLAB, verify data shape
data = load('source/VEP/data1.mat');
size(data.data)  % Should be [500, 75]
```

### Issue: Slow Evaluation

**Solutions**:
- Use GPU: `--device cuda:0`
- Increase batch size: `--batch_size 64`
- Increase workers: `--workers 8`
- Reduce dataset: `--dataset_len 1000`

---

## 📈 Comparing with DeepSIF Output

### Output Format Compatibility

Both evaluation scripts produce `.mat` files with `all_out` field, compatible with DeepSIF output format:

```matlab
% Load transformer results
transformer_results = load('checkpoints/eval_results_epoch_89.mat');
transformer_out = transformer_results.all_out;

% Load DeepSIF results (if available)
deepsif_results = load('model_result/64_the_model/preds_test_sample_source2.mat');
deepsif_out = deepsif_results.all_out;

% Compare
% ... your comparison code ...
```

---

## 📚 Next Steps

After evaluation:

1. **Analyze Results**: Use MATLAB to visualize and analyze predictions
2. **Calculate Metrics**: Compute correlation, MSE, localization error, etc.
3. **Compare Models**: Evaluate multiple checkpoints and compare
4. **Visualize on Brain**: Project source activity onto brain surface
5. **Refine Training**: Use insights to improve model training

---

## 🎯 Summary

| Task | Command |
|------|---------|
| Evaluate on test data | `python eval_sim.py --checkpoint checkpoints/best_model.pt` |
| Evaluate on real data | `python eval_real.py --checkpoint checkpoints/best_model.pt --subjects VEP` |
| Save full predictions | Add `--save_full` flag |
| Identify regions | Add `--save_regions` flag |
| Use GPU | Add `--device cuda:0` |
| Quick test | Add `--dataset_len 100` |

Both scripts output `.mat` files with `all_out` values ready for analysis! 🎉

