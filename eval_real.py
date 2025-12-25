"""
Evaluation script for real EEG data
Evaluates the trained transformer model on real EEG recordings
"""
import argparse
import os
import time
import numpy as np
from scipy.io import loadmat, savemat
import glob

import torch

from models.transformer_model import EEGSourceTransformerV2


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


def preprocess_eeg(data, normalize=True):
    """
    Preprocess real EEG data
    
    Parameters
    ----------
    data : np.ndarray
        Raw EEG data (time, channels)
    normalize : bool
        Whether to normalize the data
    
    Returns
    -------
    np.ndarray
        Preprocessed EEG data
    """
    # Remove mean (time-wise and channel-wise)
    data = data - np.mean(data, axis=0, keepdims=True)
    data = data - np.mean(data, axis=1, keepdims=True)
    
    # Normalize
    if normalize:
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
    
    return data


def evaluate_real_data(args):
    """Evaluate model on real EEG data"""
    start_time = time.time()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(f"Preparation time: {time.time() - start_time:.2f}s")
    
    # Process each subject/dataset
    for subject in args.subjects:
        print(f"\n{'='*70}")
        print(f"Processing subject: {subject}")
        print(f"{'='*70}")
        
        folder_name = os.path.join(args.data_dir, subject)
        
        if not os.path.exists(folder_name):
            print(f"WARNING: Folder not found: {folder_name}")
            continue
        
        # Find all data files
        file_pattern = os.path.join(folder_name, args.file_pattern)
        flist = glob.glob(file_pattern)
        
        if len(flist) == 0:
            print(f"WARNING: No files found matching: {file_pattern}")
            continue
        
        # Sort files naturally
        flist = sorted(flist, key=lambda name: int(os.path.basename(name).replace('data', '').replace('.mat', '')))
        
        print(f"Found {len(flist)} files")
        
        # Load and preprocess data
        test_data = []
        file_names = []
        
        for i, file_path in enumerate(flist):
            try:
                # Load data
                mat_data = loadmat(file_path)
                
                # Try different field names
                data = None
                for key in ['data', 'eeg_data', 'eeg', 'EEG']:
                    if key in mat_data:
                        data = mat_data[key]
                        break
                
                if data is None:
                    print(f"Warning: Could not find data in {file_path}")
                    continue
                
                # Ensure correct shape (time, channels)
                if data.shape[1] == 75 and data.shape[0] != 75:
                    pass  # Already correct
                elif data.shape[0] == 75 and data.shape[1] != 75:
                    data = data.T  # Transpose
                else:
                    print(f"Warning: Unexpected data shape {data.shape} in {file_path}")
                    continue
                
                # Ensure 500 time points
                if data.shape[0] != 500:
                    print(f"Warning: Expected 500 time points, got {data.shape[0]} in {file_path}")
                    # Simple resampling or padding/truncation
                    if data.shape[0] > 500:
                        data = data[:500, :]
                    else:
                        # Pad with zeros
                        padded = np.zeros((500, data.shape[1]))
                        padded[:data.shape[0], :] = data
                        data = padded
                
                # Preprocess
                data = preprocess_eeg(data, normalize=args.normalize)
                
                test_data.append(data)
                file_names.append(os.path.basename(file_path))
                
                if (i + 1) % 10 == 0:
                    print(f"Loaded {i + 1}/{len(flist)} files")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if len(test_data) == 0:
            print(f"ERROR: No valid data loaded for {subject}")
            continue
        
        print(f"Successfully loaded {len(test_data)} files")
        
        # Convert to tensor
        data_tensor = torch.from_numpy(np.array(test_data)).to(device, torch.float)
        print(f"Data tensor shape: {data_tensor.shape}")
        
        # Run inference
        print("Running inference...")
        inference_start = time.time()
        
        with torch.no_grad():
            # Process in batches if needed
            if len(test_data) > args.batch_size:
                all_predictions = []
                for i in range(0, len(test_data), args.batch_size):
                    batch = data_tensor[i:i+args.batch_size]
                    predictions = model(batch)
                    all_predictions.append(predictions.cpu().numpy())
                all_out = np.concatenate(all_predictions, axis=0)
            else:
                predictions = model(data_tensor)
                all_out = predictions.cpu().numpy()
        
        inference_time = time.time() - inference_start
        print(f"Inference complete: {inference_time:.2f}s")
        print(f"Output shape: {all_out.shape}")
        
        # Save results
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        output_filename = f"transformer_predictions_{checkpoint_name}.mat"
        output_path = os.path.join(folder_name, output_filename)
        
        # Prepare output data
        output_data = {
            'all_out': all_out,
            'file_names': file_names,
            'checkpoint': args.checkpoint,
            'inference_time': inference_time,
            'num_samples': len(test_data)
        }
        
        # Save
        savemat(output_path, output_data)
        print(f"Saved predictions to: {output_path}")
        
        # Print statistics
        print(f"\nPrediction statistics:")
        print(f"  Min: {np.min(all_out):.6f}")
        print(f"  Max: {np.max(all_out):.6f}")
        print(f"  Mean: {np.mean(all_out):.6f}")
        print(f"  Std: {np.std(all_out):.6f}")
    
    print(f"\n{'='*70}")
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer Model on Real EEG Data')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='source',
                       help='Directory containing subject folders')
    parser.add_argument('--subjects', type=str, nargs='+', default=['VEP'],
                       help='List of subject names/folders to process')
    parser.add_argument('--file_pattern', type=str, default='data*.mat',
                       help='Pattern for data files (e.g., data*.mat)')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize input data')
    
    args = parser.parse_args()
    
    evaluate_real_data(args)


if __name__ == '__main__':
    main()

