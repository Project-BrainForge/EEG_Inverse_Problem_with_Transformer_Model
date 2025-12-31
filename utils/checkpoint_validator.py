"""
Checkpoint validation and inspection utility
"""
import torch
import os
import sys


def validate_checkpoint(checkpoint_path, verbose=True):
    """
    Validate a checkpoint file for NaN/Inf values
    
    Args:
        checkpoint_path: Path to checkpoint file
        verbose: Whether to print detailed information
        
    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    if not os.path.exists(checkpoint_path):
        if verbose:
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating: {checkpoint_path}")
            print('='*60)
        
        # Check epoch and losses
        epoch = ckpt.get('epoch', 'N/A')
        train_loss = ckpt.get('train_loss', float('nan'))
        val_loss = ckpt.get('val_loss', float('nan'))
        
        if verbose:
            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
        
        is_valid = True
        
        # Check losses for NaN/Inf
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
            if verbose:
                print(f"ERROR: Train loss is {'NaN' if torch.isnan(torch.tensor(train_loss)) else 'Inf'}")
            is_valid = False
        
        if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            if verbose:
                print(f"ERROR: Val loss is {'NaN' if torch.isnan(torch.tensor(val_loss)) else 'Inf'}")
            is_valid = False
        
        # Check model weights
        model_state = ckpt.get('model_state_dict', {})
        if verbose:
            print(f"\nChecking {len(model_state)} model parameters...")
        
        nan_params = []
        inf_params = []
        
        for name, param in model_state.items():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
        
        if nan_params:
            if verbose:
                print(f"ERROR: {len(nan_params)} parameters contain NaN:")
                for name in nan_params[:5]:  # Show first 5
                    print(f"  - {name}")
                if len(nan_params) > 5:
                    print(f"  ... and {len(nan_params) - 5} more")
            is_valid = False
        
        if inf_params:
            if verbose:
                print(f"ERROR: {len(inf_params)} parameters contain Inf:")
                for name in inf_params[:5]:  # Show first 5
                    print(f"  - {name}")
                if len(inf_params) > 5:
                    print(f"  ... and {len(inf_params) - 5} more")
            is_valid = False
        
        if is_valid and verbose:
            print("\n✓ Checkpoint is VALID")
            print(f"  All {len(model_state)} parameters are finite")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Val loss: {val_loss:.6f}")
            
            # Show sample weight statistics
            print("\nSample parameter statistics:")
            for i, (name, param) in enumerate(list(model_state.items())[:3]):
                print(f"  {name}:")
                print(f"    Shape: {param.shape}")
                print(f"    Range: [{param.min():.6f}, {param.max():.6f}]")
                print(f"    Mean: {param.mean():.6f}, Std: {param.std():.6f}")
        elif not is_valid and verbose:
            print("\n✗ Checkpoint is INVALID")
        
        return is_valid
        
    except Exception as e:
        if verbose:
            print(f"ERROR: Failed to load checkpoint: {e}")
        return False


def inspect_all_checkpoints(checkpoint_dir="checkpoints"):
    """
    Inspect all checkpoints in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}")
    print('='*60)
    
    valid_count = 0
    invalid_count = 0
    
    for ckpt_file in sorted(checkpoint_files):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        is_valid = validate_checkpoint(ckpt_path, verbose=False)
        
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{status:12} - {ckpt_file}")
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {valid_count} valid, {invalid_count} invalid")
    print('='*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate checkpoint files')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    parser.add_argument('--dir', type=str, default='checkpoints', help='Directory containing checkpoints')
    parser.add_argument('--all', action='store_true', help='Inspect all checkpoints in directory')
    
    args = parser.parse_args()
    
    if args.all:
        inspect_all_checkpoints(args.dir)
    elif args.checkpoint:
        validate_checkpoint(args.checkpoint, verbose=True)
    else:
        # Default: inspect all checkpoints
        inspect_all_checkpoints(args.dir)

