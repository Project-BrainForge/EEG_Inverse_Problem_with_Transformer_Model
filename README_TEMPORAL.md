# Temporal Visualization - Quick Guide

## 🎯 Problem: How to Visualize All 500 Timepoints?

If your predictions have shape `(num_samples, 500, 994)`, you have **temporal data** with 500 timepoints per sample. Here's how to visualize them!

## ⚡ Three Solutions

### 1. Grid View (Recommended)

Show multiple timepoints in a grid:

```matlab
cd misc_scripts
visualize_timepoints(1, 16)  % 4x4 grid
```

**When to use:** Quick overview, comparing timepoints, publications

### 2. Custom Selection

Choose specific timepoints:

```matlab
visualize_selected_timepoints(1, 1:50:500)  % Every 50th
visualize_selected_timepoints(1, [1, 100, 200, 300, 400, 500])  % Specific times
```

**When to use:** Focused analysis, key moments

### 3. Animation

Create video with all timepoints:

```matlab
visualize_all_timepoints_animation(1)
```

**When to use:** Presentations, full temporal evolution

## 📚 Documentation

- **Quick Start:** `TEMPORAL_VISUALIZATION_SUMMARY.txt`
- **Detailed Guide:** `misc_scripts/TEMPORAL_VISUALIZATION_GUIDE.md`
- **Quick Reference:** `misc_scripts/TEMPORAL_QUICKSTART.txt`

## 🆕 New Scripts Created

1. `visualize_timepoints.m` - Grid view
2. `visualize_selected_timepoints.m` - Custom selection  
3. `visualize_all_timepoints_animation.m` - Animation

## 📊 Your Code Explained

```matlab
region_values = all_out(sample_idx, :);
vertex_values = zeros(size(pos, 1), 1);

for vertex_idx = 1:length(region_mapping)
    region_id = region_mapping(vertex_idx);
    if region_id > 0 && region_id <= length(region_values)
        vertex_values(vertex_idx) = region_values(region_id);
    end
end
```

This code converts **994 regions** → **20,484 vertices** for ONE timepoint.

For 500 timepoints, you need to repeat this for each timepoint. The new scripts do this automatically!

## 🚀 Complete Example

```matlab
cd misc_scripts

% Check your data shape
pred_data = load('../source/VEP/transformer_predictions_best_model.mat');
size(pred_data.all_out)  % Should be (N, 500, 994)

% Visualize!
visualize_timepoints(1, 16)              % Grid overview
visualize_selected_timepoints(1, 1:50:500)  % Every 50th
visualize_all_timepoints_animation(1)    % Full animation
```

## 💡 Tips

- Start with `visualize_timepoints(1, 16)` for quick overview
- Use custom selection for focused analysis  
- Create animations for presentations
- Adjust threshold in scripts to focus on strong activations

## 📖 More Information

See `TEMPORAL_VISUALIZATION_SUMMARY.txt` for complete guide!

---

Happy temporal visualization! 🧠⏱️✨

