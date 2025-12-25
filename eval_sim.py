"""
Evaluation script for simulated EEG data
Evaluates the trained transformer model and saves predictions
"""
import argparse
import os
import time
import collections
import numpy as np
from scipy.io import loadmat, savemat
import logging
import datetime

import torch
from torch.utils.data import DataLoader

from models.transformer_model import EEGSourceTransformerV2
from utils.loader import SpikeEEGMetadataDataset, load_mat_file
from configs.config import Config


def get_otsu_regions(predictions, labels, threshold_method='otsu'):
    """
    Identify active source regions using Otsu's method or percentile thresholding
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (batch_size, time, regions)
    labels : np.ndarray
        Ground truth labels (batch_size, num_sources, max_patch_size)
    threshold_method : str
        'otsu' or 'percentile'
    
    Returns
    -------
    dict
        Dictionary containing identified regions and predictions
    """
    from sklearn.preprocessing import threshold
    try:
        from skimage.filters import threshold_otsu
    except ImportError:
        print("Warning: skimage not installed. Using simple thresholding.")
        threshold_otsu = None
    
    batch_size = predictions.shape[0]
    results = {
        'all_regions': [],
        'all_out': []
    }
    
    for i in range(batch_size):
        # Get temporal mean of prediction
        pred_mean = np.mean(predictions[i], axis=0)  # (regions,)
        
        # Apply Otsu's threshold
        if threshold_otsu is not None and threshold_method == 'otsu':
            try:
                thresh = threshold_otsu(pred_mean)
            except:
                thresh = np.percentile(pred_mean, 95)
        else:
            # Use 95th percentile as threshold
            thresh = np.percentile(pred_mean, 95)
        
        # Identify regions above threshold
        active_regions = np.where(pred_mean > thresh)[0]
        
        results['all_regions'].append(active_regions)
        results['all_out'].append(pred_mean)
    
    return results


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"=> Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = EEGSourceTransformerV2(
            eeg_channels=config['EEG_CHANNELS'],
            source_regions=config['SOURCE_REGIONS'],
            d_model=config['D_MODEL'],
            nhead=config['NHEAD'],
            num_layers=config['NUM_LAYERS'],
            dim_feedforward=config['DIM_FEEDFORWARD'],
            dropout=config['DROPOUT']
        ).to(device)
    else:
        # Use default config
        model = EEGSourceTransformerV2(
            eeg_channels=75,
            source_regions=994,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"=> Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Train loss: {checkpoint['train_loss']:.6f}")
    print(f"   Val loss: {checkpoint['val_loss']:.6f}")
    
    return model, checkpoint


def evaluate_simulated(args):
    """Evaluate model on simulated test data"""
    start_time = time.time()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Load forward matrix
    print(f"\nLoading forward matrix from {args.fwd_matrix}...")
    fwd_data = load_mat_file(args.fwd_matrix)
    
    fwd = None
    for key in ['fwd', 'forward', 'leadfield', 'L']:
        if key in fwd_data:
            fwd = fwd_data[key]
            print(f"Found forward matrix with key '{key}', shape: {fwd.shape}")
            break
    
    if fwd is None:
        raise ValueError("Could not find forward matrix in file")
    
    # Transpose if needed
    if fwd.shape[0] == 994 and fwd.shape[1] == 75:
        fwd = fwd.T
    
    # Load test dataset
    print(f"\nLoading test data from {args.test_metadata}...")
    test_dataset = SpikeEEGMetadataDataset(
        metadata_path=args.test_metadata,
        fwd=fwd,
        nmm_spikes_dir=args.nmm_spikes_dir,
        dataset_len=args.dataset_len,
        normalize=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(f"Preparation time: {time.time() - start_time:.2f}s")
    
    # Setup logging
    result_dir = os.path.dirname(args.checkpoint)
    log_file = os.path.join(result_dir, 'evaluation_log.txt')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    logger.info(f"=================== Evaluation: {datetime.datetime.now()} ==================")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test data: {args.test_metadata}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Evaluation
    print("\n" + "="*70)
    print("Starting Evaluation")
    print("="*70)
    
    model.eval()
    
    eval_dict = collections.defaultdict(list)
    eval_dict['all_out'] = []        # Model predictions
    eval_dict['all_nmm'] = []        # Ground truth source activity
    eval_dict['all_regions'] = []    # Identified source regions
    eval_dict['all_eeg'] = []        # Input EEG data (optional)
    
    criterion = torch.nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (eeg_data, source_data) in enumerate(test_loader):
            # Move to device
            eeg_data = eeg_data.to(device)
            source_data = source_data.to(device)
            
            # Forward pass
            predictions = model(eeg_data)
            
            # Calculate loss
            loss = criterion(predictions, source_data)
            total_loss += loss.item()
            
            # Convert to numpy
            pred_np = predictions.cpu().numpy()
            source_np = source_data.cpu().numpy()
            eeg_np = eeg_data.cpu().numpy()
            
            # Identify active regions
            if args.save_regions:
                # Create dummy labels for region identification
                # In real scenario, you'd load actual labels from metadata
                dummy_labels = np.zeros((pred_np.shape[0], 2, 70))
                eval_results = get_otsu_regions(pred_np, dummy_labels)
                eval_dict['all_regions'].extend(eval_results['all_regions'])
            
            # Save predictions and ground truth
            if args.save_full:
                eval_dict['all_out'].append(pred_np)
                eval_dict['all_nmm'].append(source_np)
                if args.save_eeg:
                    eval_dict['all_eeg'].append(eeg_np)
            else:
                # Save only temporal mean to reduce file size
                eval_dict['all_out'].append(np.mean(pred_np, axis=1))  # (batch, regions)
                eval_dict['all_nmm'].append(np.mean(source_np, axis=1))
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate average loss
    avg_loss = total_loss / len(test_loader)
    print(f"\nAverage MSE Loss: {avg_loss:.6f}")
    logger.info(f"Average MSE Loss: {avg_loss:.6f}")
    
    # Concatenate results
    if args.save_full:
        eval_dict['all_out'] = np.concatenate(eval_dict['all_out'], axis=0)
        eval_dict['all_nmm'] = np.concatenate(eval_dict['all_nmm'], axis=0)
        if args.save_eeg:
            eval_dict['all_eeg'] = np.concatenate(eval_dict['all_eeg'], axis=0)
    else:
        eval_dict['all_out'] = np.concatenate(eval_dict['all_out'], axis=0)
        eval_dict['all_nmm'] = np.concatenate(eval_dict['all_nmm'], axis=0)
    
    eval_dict['avg_loss'] = avg_loss
    
    # Save results
    output_name = args.output if args.output else f"eval_results_epoch_{checkpoint['epoch']}.mat"
    output_path = os.path.join(result_dir, output_name)
    
    print(f"\nSaving results to {output_path}...")
    savemat(output_path, eval_dict)
    print("Evaluation complete!")
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Output shape: {eval_dict['all_out'].shape}")
    logger.info(f"Total evaluation time: {time.time() - start_time:.2f}s")
    
    print(f"\nTotal evaluation time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer Model on Simulated Data')
    
    # Data parameters
    parser.add_argument('--test_metadata', type=str, default='source/test_sample_source1.mat',
                       help='Path to test metadata file')
    parser.add_argument('--nmm_spikes_dir', type=str, default='source/nmm_spikes',
                       help='Path to NMM spikes directory')
    parser.add_argument('--fwd_matrix', type=str, default='anatomy/leadfield_75_20k.mat',
                       help='Path to forward matrix file')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--dataset_len', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='',
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--save_full', action='store_true',
                       help='Save full temporal predictions (default: temporal mean only)')
    parser.add_argument('--save_eeg', action='store_true',
                       help='Save input EEG data as well')
    parser.add_argument('--save_regions', action='store_true',
                       help='Identify and save active regions using thresholding')
    
    args = parser.parse_args()
    
    evaluate_simulated(args)


if __name__ == '__main__':
    main()

