"""
Evaluation script for EEG source localization transformer
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))

from models.transformer import create_model
from data.dataset import create_dataloaders
from utils.helpers import load_checkpoint, compute_metrics


class Evaluator:
    """Evaluator class for EEG source localization"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader, self.val_loader, self.train_dataset, self.val_dataset = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            train_ratio=config['train_ratio'],
            seed=config['seed']
        )
        
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
        load_checkpoint(checkpoint_path, self.model)
        
        self.criterion = nn.MSELoss()
    
    def evaluate(self, split='val', save_results=True):
        """
        Evaluate the model
        
        Args:
            split: 'train' or 'val'
            save_results: Whether to save results to file
        """
        self.model.eval()
        
        dataloader = self.train_loader if split == 'train' else self.val_loader
        dataset = self.train_dataset if split == 'train' else self.val_dataset
        
        print(f"\n{'='*50}")
        print(f"Evaluating on {split} set ({len(dataloader.dataset)} samples)")
        print(f"{'='*50}\n")
        
        all_predictions = []
        all_targets = []
        all_eeg = []
        losses = []
        
        with torch.no_grad():
            for batch_idx, (eeg_data, source_data) in enumerate(dataloader):
                eeg_data = eeg_data.to(self.device)
                source_data = source_data.to(self.device)
                
                # Forward pass
                predictions = self.model(eeg_data)
                
                # Compute loss
                loss = self.criterion(predictions, source_data)
                losses.append(loss.item())
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(source_data.cpu())
                all_eeg.append(eeg_data.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_eeg = torch.cat(all_eeg, dim=0)
        
        # Denormalize if needed
        if dataset.normalize:
            all_predictions = dataset.denormalize_source(all_predictions)
            all_targets = dataset.denormalize_source(all_targets)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results on {split} set:")
        print(f"{'='*50}")
        print(f"  MSE:             {metrics['mse']:.6f}")
        print(f"  MAE:             {metrics['mae']:.6f}")
        print(f"  RMSE:            {metrics['rmse']:.6f}")
        print(f"  Correlation:     {metrics['correlation']:.4f}")
        print(f"  R²:              {metrics['r2']:.4f}")
        print(f"  Relative Error:  {metrics['relative_error']:.4f}")
        print(f"{'='*50}\n")
        
        # Save results
        if save_results:
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            results = {
                'predictions': all_predictions.numpy(),
                'targets': all_targets.numpy(),
                'eeg': all_eeg.numpy(),
                'metrics': metrics
            }
            
            results_path = results_dir / f'evaluation_{split}.npz'
            np.savez(results_path, **results)
            print(f"Results saved to {results_path}")
            
            # Save metrics to text file
            metrics_path = results_dir / f'metrics_{split}.txt'
            with open(metrics_path, 'w') as f:
                f.write(f"Evaluation Results on {split} set\n")
                f.write("="*50 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"Metrics saved to {metrics_path}")
        
        return metrics, all_predictions, all_targets, all_eeg
    
    def visualize_predictions(self, sample_idx=0, split='val', save_fig=True):
        """
        Visualize predictions for a single sample
        
        Args:
            sample_idx: Index of sample to visualize
            split: 'train' or 'val'
            save_fig: Whether to save figure
        """
        dataset = self.train_dataset if split == 'train' else self.val_dataset
        
        # Get sample
        eeg_data, source_data = dataset[sample_idx]
        
        # Add batch dimension and move to device
        eeg_data = eeg_data.unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(eeg_data)
        
        # Remove batch dimension
        prediction = prediction.squeeze(0).cpu()
        source_data = source_data.cpu()
        eeg_data = eeg_data.squeeze(0).cpu()
        
        # Denormalize
        if dataset.normalize:
            prediction = dataset.denormalize_source(prediction)
            source_data = dataset.denormalize_source(source_data)
        
        # Convert to numpy
        prediction = prediction.numpy()
        source_data = source_data.numpy()
        eeg_data = eeg_data.numpy()
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot EEG data (first 5 channels)
        axes[0].plot(eeg_data[:, :5])
        axes[0].set_title('EEG Input (first 5 channels)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        axes[0].legend([f'Ch {i+1}' for i in range(5)])
        
        # Plot target source activity (first 5 regions)
        axes[1].plot(source_data[:, :5])
        axes[1].set_title('Ground Truth Source Activity (first 5 regions)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)
        axes[1].legend([f'Region {i+1}' for i in range(5)])
        
        # Plot predicted source activity (first 5 regions)
        axes[2].plot(prediction[:, :5])
        axes[2].set_title('Predicted Source Activity (first 5 regions)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True)
        axes[2].legend([f'Region {i+1}' for i in range(5)])
        
        # Plot error
        error = np.abs(prediction - source_data)
        axes[3].plot(error[:, :5])
        axes[3].set_title('Absolute Error (first 5 regions)')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Error')
        axes[3].grid(True)
        axes[3].legend([f'Region {i+1}' for i in range(5)])
        
        plt.tight_layout()
        
        if save_fig:
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            fig_path = results_dir / f'visualization_sample_{sample_idx}_{split}.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {fig_path}")
        
        plt.close()
        
        # Compute metrics for this sample
        sample_metrics = compute_metrics(prediction, source_data)
        print(f"\nMetrics for sample {sample_idx}:")
        for key, value in sample_metrics.items():
            print(f"  {key}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG source localization transformer')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index for visualization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Evaluate
    evaluator.evaluate(split=args.split, save_results=True)
    
    # Visualize if requested
    if args.visualize:
        evaluator.visualize_predictions(
            sample_idx=args.sample_idx,
            split=args.split,
            save_fig=True
        )


if __name__ == "__main__":
    main()

