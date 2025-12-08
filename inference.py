"""
Simple inference script for single sample prediction
"""
import os
import torch
import numpy as np
from scipy.io import loadmat, savemat
import argparse

from models.transformer_model import EEGSourceTransformerV2


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
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
    return model


def predict(model, eeg_data, device, stats=None):
    """
    Predict source data from EEG data
    
    Args:
        model: Trained model
        eeg_data: EEG data of shape (500, 75)
        device: Device to run inference on
        stats: Normalization statistics (optional)
    
    Returns:
        predicted_source: Predicted source data of shape (500, 994)
    """
    # Convert to tensor
    if isinstance(eeg_data, np.ndarray):
        eeg_tensor = torch.from_numpy(eeg_data).float()
    else:
        eeg_tensor = eeg_data
    
    # Normalize if stats provided
    if stats is not None:
        eeg_mean = torch.from_numpy(stats['eeg_mean']).float()
        eeg_std = torch.from_numpy(stats['eeg_std']).float()
        eeg_tensor = (eeg_tensor - eeg_mean) / eeg_std
    
    # Add batch dimension
    eeg_tensor = eeg_tensor.unsqueeze(0)  # (1, 500, 75)
    
    # Move to device
    eeg_tensor = eeg_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predicted_source = model(eeg_tensor)
    
    # Remove batch dimension
    predicted_source = predicted_source.squeeze(0)  # (500, 994)
    
    # Denormalize if stats provided
    if stats is not None:
        source_mean = torch.from_numpy(stats['source_mean']).float().to(device)
        source_std = torch.from_numpy(stats['source_std']).float().to(device)
        predicted_source = predicted_source * source_std + source_mean
    
    return predicted_source.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Inference for EEG Source Localization')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input .mat file containing eeg_data')
    parser.add_argument('--output', type=str, default='predicted_source.mat', 
                       help='Path to save predicted source data')
    parser.add_argument('--stats', type=str, default=None,
                       help='Path to normalization statistics (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)
    
    # Load normalization statistics
    stats = None
    if args.stats is not None:
        stats_path = args.stats
    else:
        # Try to find stats in checkpoint directory
        stats_path = os.path.join(os.path.dirname(args.checkpoint), 
                                  'normalization_stats.pt')
    
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, weights_only=False)
        print(f"Loaded normalization statistics from {stats_path}")
    else:
        print("Warning: No normalization statistics found. Using raw data.")
    
    # Load input EEG data
    print(f"\nLoading input from {args.input}...")
    input_data = loadmat(args.input)
    
    if 'eeg_data' not in input_data:
        raise ValueError("Input .mat file must contain 'eeg_data' variable")
    
    eeg_data = input_data['eeg_data'].astype(np.float32)
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Check shape
    if eeg_data.shape != (500, 75):
        raise ValueError(f"Expected EEG shape (500, 75), got {eeg_data.shape}")
    
    # Predict
    print("\nRunning inference...")
    predicted_source = predict(model, eeg_data, device, stats)
    print(f"Predicted source shape: {predicted_source.shape}")
    
    # Save output
    print(f"\nSaving prediction to {args.output}...")
    output_data = {
        'predicted_source': predicted_source,
        'eeg_data': eeg_data
    }
    
    # If ground truth exists, include it
    if 'source_data' in input_data:
        output_data['ground_truth_source'] = input_data['source_data']
        
        # Calculate error metrics
        ground_truth = input_data['source_data'].astype(np.float32)
        mse = np.mean((predicted_source - ground_truth) ** 2)
        mae = np.mean(np.abs(predicted_source - ground_truth))
        corr = np.corrcoef(predicted_source.flatten(), ground_truth.flatten())[0, 1]
        
        print("\nComparison with ground truth:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {corr:.6f}")
        
        output_data['metrics'] = {
            'mse': mse,
            'mae': mae,
            'correlation': corr
        }
    
    savemat(args.output, output_data)
    print(f"✓ Prediction saved successfully!")


if __name__ == "__main__":
    main()

