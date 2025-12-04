"""
Inference script for EEG source localization transformer
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import numpy as np

import torch
from scipy.io import loadmat, savemat

sys.path.append(str(Path(__file__).parent))

from models.transformer import create_model
from utils.helpers import load_checkpoint


class Inferencer:
    """Inference class for EEG source localization"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        print("\nCreating model...")
        self.model = create_model(
            model_type=config['model']['type'],
            input_channels=config['model']['input_channels'],
            output_channels=config['model']['output_channels'],
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model'].get('num_layers', 6),
            num_encoder_layers=config['model'].get('num_encoder_layers', 6),
            num_decoder_layers=config['model'].get('num_decoder_layers', 6),
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        )
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.model)
        
        # Load normalization statistics from checkpoint config if available
        if 'normalization_stats' in checkpoint:
            self.eeg_mean = checkpoint['normalization_stats']['eeg_mean']
            self.eeg_std = checkpoint['normalization_stats']['eeg_std']
            self.source_mean = checkpoint['normalization_stats']['source_mean']
            self.source_std = checkpoint['normalization_stats']['source_std']
            self.normalize = True
            print("Loaded normalization statistics from checkpoint")
        else:
            self.normalize = False
            print("Warning: No normalization statistics found. Using raw data.")
        
        self.model.eval()
        print("Model ready for inference!")
    
    def predict(self, eeg_data):
        """
        Predict source activity from EEG data
        
        Args:
            eeg_data: EEG data of shape (time_points, channels) or (batch, time_points, channels)
                      e.g., (500, 75) or (batch, 500, 75)
        
        Returns:
            predictions: Predicted source activity of shape (time_points, regions) or (batch, time_points, regions)
                        e.g., (500, 994) or (batch, 500, 994)
        """
        # Convert to numpy if needed
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.numpy()
        
        # Add batch dimension if needed
        squeeze_output = False
        if eeg_data.ndim == 2:
            eeg_data = eeg_data[np.newaxis, ...]  # (1, time, channels)
            squeeze_output = True
        
        # Normalize
        if self.normalize:
            eeg_data = (eeg_data - self.eeg_mean) / self.eeg_std
        
        # Convert to tensor
        eeg_tensor = torch.from_numpy(eeg_data.astype(np.float32)).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(eeg_tensor)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        
        # Denormalize
        if self.normalize:
            predictions = predictions * self.source_std + self.source_mean
        
        # Remove batch dimension if needed
        if squeeze_output:
            predictions = predictions[0]
        
        return predictions
    
    def predict_from_file(self, input_path, output_path=None):
        """
        Predict from a .mat file
        
        Args:
            input_path: Path to input .mat file containing 'eeg_data'
            output_path: Path to save output .mat file (optional)
        
        Returns:
            predictions: Predicted source activity
        """
        print(f"\nLoading EEG data from {input_path}")
        
        # Load .mat file
        data = loadmat(input_path)
        
        if 'eeg_data' not in data:
            raise ValueError("Input file must contain 'eeg_data' field")
        
        eeg_data = data['eeg_data'].astype(np.float32)
        print(f"EEG data shape: {eeg_data.shape}")
        
        # Predict
        print("Running inference...")
        predictions = self.predict(eeg_data)
        print(f"Predictions shape: {predictions.shape}")
        
        # Save output if specified
        if output_path:
            output_data = {
                'source_data_predicted': predictions,
                'eeg_data': eeg_data
            }
            
            # Include ground truth if available
            if 'source_data' in data:
                output_data['source_data_true'] = data['source_data']
            
            savemat(output_path, output_data)
            print(f"Predictions saved to {output_path}")
        
        return predictions
    
    def predict_batch_files(self, input_dir, output_dir):
        """
        Predict for all .mat files in a directory
        
        Args:
            input_dir: Directory containing input .mat files
            output_dir: Directory to save output files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all .mat files
        mat_files = sorted(input_dir.glob("*.mat"))
        
        if len(mat_files) == 0:
            print(f"No .mat files found in {input_dir}")
            return
        
        print(f"\nProcessing {len(mat_files)} files...")
        
        for i, mat_file in enumerate(mat_files):
            print(f"\n[{i+1}/{len(mat_files)}] Processing {mat_file.name}")
            
            output_path = output_dir / f"predicted_{mat_file.name}"
            
            try:
                self.predict_from_file(mat_file, output_path)
            except Exception as e:
                print(f"Error processing {mat_file.name}: {e}")
                continue
        
        print(f"\nBatch processing completed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference for EEG source localization transformer')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .mat file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file or directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process all files in input directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inferencer
    inferencer = Inferencer(config, args.checkpoint)
    
    # Run inference
    if args.batch:
        # Batch processing
        input_dir = args.input
        output_dir = args.output if args.output else 'results/predictions'
        inferencer.predict_batch_files(input_dir, output_dir)
    else:
        # Single file processing
        input_file = args.input
        output_file = args.output if args.output else 'results/prediction.mat'
        inferencer.predict_from_file(input_file, output_file)


if __name__ == "__main__":
    main()

