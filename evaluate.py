"""
Evaluation and inference script for EEG Source Localization Transformer
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from scipy.io import savemat

from models.transformer_model import EEGSourceTransformerV2
from utils.dataset import create_dataloaders, EEGSourceDataset
from configs.config import Config


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint['config']
    
    # Create model
    model = EEGSourceTransformerV2(
        eeg_channels=model_config['EEG_CHANNELS'],
        source_regions=model_config['SOURCE_REGIONS'],
        d_model=model_config['D_MODEL'],
        nhead=model_config['NHEAD'],
        num_layers=model_config['NUM_LAYERS'],
        dim_feedforward=model_config['DIM_FEEDFORWARD'],
        dropout=model_config['DROPOUT']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Train Loss: {checkpoint['train_loss']:.6f}, Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for eeg_data, source_data in tqdm(data_loader):
            # Move data to device
            eeg_data = eeg_data.to(device)
            source_data = source_data.to(device)
            
            # Forward pass
            predictions = model(eeg_data)
            
            # Calculate metrics
            mse = criterion_mse(predictions, source_data)
            mae = criterion_mae(predictions, source_data)
            
            total_mse += mse.item() * eeg_data.size(0)
            total_mae += mae.item() * eeg_data.size(0)
            total_samples += eeg_data.size(0)
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(source_data.cpu().numpy())
    
    # Calculate average metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate correlation
    correlations = []
    for i in range(all_predictions.shape[0]):
        pred_flat = all_predictions[i].flatten()
        target_flat = all_targets[i].flatten()
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
        correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"MSE: {avg_mse:.6f}")
    print(f"RMSE: {avg_rmse:.6f}")
    print(f"MAE: {avg_mae:.6f}")
    print(f"Average Correlation: {avg_correlation:.6f}")
    print("="*50)
    
    return {
        'mse': avg_mse,
        'rmse': avg_rmse,
        'mae': avg_mae,
        'correlation': avg_correlation,
        'predictions': all_predictions,
        'targets': all_targets
    }


def predict_single_sample(model, eeg_data, device, stats=None):
    """
    Predict source data for a single EEG sample
    
    Args:
        model: Trained model
        eeg_data: EEG data of shape (seq_len, n_channels) or (1, seq_len, n_channels)
        device: Device to run inference on
        stats: Normalization statistics (optional)
    
    Returns:
        predicted_source: Predicted source data of shape (seq_len, n_sources)
    """
    model.eval()
    
    # Ensure correct shape
    if eeg_data.dim() == 2:
        eeg_data = eeg_data.unsqueeze(0)  # Add batch dimension
    
    # Normalize if stats provided
    if stats is not None:
        eeg_mean = torch.from_numpy(stats['eeg_mean']).float()
        eeg_std = torch.from_numpy(stats['eeg_std']).float()
        eeg_data = (eeg_data - eeg_mean) / eeg_std
    
    # Move to device
    eeg_data = eeg_data.to(device)
    
    # Predict
    with torch.no_grad():
        predicted_source = model(eeg_data)
    
    # Denormalize if stats provided
    if stats is not None:
        source_mean = torch.from_numpy(stats['source_mean']).float().to(device)
        source_std = torch.from_numpy(stats['source_std']).float().to(device)
        predicted_source = predicted_source * source_std + source_mean
    
    # Remove batch dimension
    predicted_source = predicted_source.squeeze(0)
    
    return predicted_source.cpu().numpy()


def visualize_predictions(predictions, targets, num_samples=3, save_dir='visualizations'):
    """Visualize predictions vs targets"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_samples, predictions.shape[0])):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot time series for a few regions
        regions_to_plot = [0, 100, 500, 900]
        
        for idx, region in enumerate(regions_to_plot):
            ax = axes[idx // 2, idx % 2]
            ax.plot(targets[i, :, region], label='Ground Truth', alpha=0.7)
            ax.plot(predictions[i, :, region], label='Prediction', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Activity')
            ax.set_title(f'Sample {i+1} - Region {region}')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_sample_{i+1}.png'), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {save_dir}/")


def save_predictions(predictions, targets, save_path='predictions.mat'):
    """Save predictions to .mat file"""
    savemat(save_path, {
        'predictions': predictions,
        'targets': targets
    })
    print(f"Predictions saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG Source Localization Transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset_with_label', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load normalization statistics
    stats_path = os.path.join(os.path.dirname(args.checkpoint), 'normalization_stats.pt')
    stats = None
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        print("Normalization statistics loaded")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=4,
        normalize=True
    )
    
    # Evaluate on test set
    results = evaluate_model(model, test_loader, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_predictions(
            results['predictions'], 
            results['targets'],
            num_samples=5,
            save_dir=os.path.join(args.output_dir, 'visualizations')
        )
    
    # Save predictions
    if args.save_predictions:
        print("\nSaving predictions...")
        save_predictions(
            results['predictions'],
            results['targets'],
            save_path=os.path.join(args.output_dir, 'predictions.mat')
        )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n")
        f.write(f"Average Correlation: {results['correlation']:.6f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

