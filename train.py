"""
Training script for EEG Source Localization Transformer
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from models.transformer_model import EEGSourceTransformerV2
from utils.dataset import create_dataloaders
from configs.config import Config


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=15, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop


def get_loss_function(loss_fn_name: str):
    """Get loss function by name"""
    if loss_fn_name == "mse":
        return nn.MSELoss()
    elif loss_fn_name == "mae":
        return nn.L1Loss()
    elif loss_fn_name == "huber":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


def check_for_nan_inf(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        return f"{name} contains NaN"
    if torch.isinf(tensor).any():
        return f"{name} contains Inf"
    return None


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (eeg_data, source_data) in enumerate(pbar):
        # Move data to device
        eeg_data = eeg_data.to(device)
        source_data = source_data.to(device)
        
        # Check input data for NaN/Inf
        error = check_for_nan_inf(eeg_data, "EEG data")
        if error:
            print(f"\nERROR: {error} in batch {batch_idx}")
            return float('nan')
        
        error = check_for_nan_inf(source_data, "Source data")
        if error:
            print(f"\nERROR: {error} in batch {batch_idx}")
            return float('nan')
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.USE_AMP:
            with autocast():
                predictions = model(eeg_data)
                
                # Check predictions for NaN/Inf
                error = check_for_nan_inf(predictions, "Predictions")
                if error:
                    print(f"\nERROR: {error} in batch {batch_idx}")
                    print(f"  Input stats - mean: {eeg_data.mean():.6f}, std: {eeg_data.std():.6f}")
                    print(f"  Input range: [{eeg_data.min():.6f}, {eeg_data.max():.6f}]")
                    return float('nan')
                
                loss = criterion(predictions, source_data)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is {'NaN' if torch.isnan(loss) else 'Inf'} in batch {batch_idx}")
                return float('nan')
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Check gradients for NaN/Inf before clipping
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    error = check_for_nan_inf(param.grad, f"Gradient of {name}")
                    if error:
                        print(f"\nERROR: {error}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                return float('nan')
            
            # Gradient clipping
            if config.CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                
                # Check if gradient norm is too large
                if grad_norm > 100 * config.CLIP_GRAD_NORM:
                    print(f"\nWARNING: Very large gradient norm: {grad_norm:.2f} (threshold: {config.CLIP_GRAD_NORM})")
            
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(eeg_data)
            
            # Check predictions for NaN/Inf
            error = check_for_nan_inf(predictions, "Predictions")
            if error:
                print(f"\nERROR: {error} in batch {batch_idx}")
                print(f"  Input stats - mean: {eeg_data.mean():.6f}, std: {eeg_data.std():.6f}")
                print(f"  Input range: [{eeg_data.min():.6f}, {eeg_data.max():.6f}]")
                return float('nan')
            
            loss = criterion(predictions, source_data)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is {'NaN' if torch.isnan(loss) else 'Inf'} in batch {batch_idx}")
                return float('nan')
            
            loss.backward()
            
            # Check gradients for NaN/Inf before clipping
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    error = check_for_nan_inf(param.grad, f"Gradient of {name}")
                    if error:
                        print(f"\nERROR: {error}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                return float('nan')
            
            # Gradient clipping
            if config.CLIP_GRAD_NORM > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                
                # Check if gradient norm is too large
                if grad_norm > 100 * config.CLIP_GRAD_NORM:
                    print(f"\nWARNING: Very large gradient norm: {grad_norm:.2f} (threshold: {config.CLIP_GRAD_NORM})")
            
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, config, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
    
    with torch.no_grad():
        for eeg_data, source_data in pbar:
            # Move data to device
            eeg_data = eeg_data.to(device)
            source_data = source_data.to(device)
            
            # Forward pass
            if config.USE_AMP:
                with autocast():
                    predictions = model(eeg_data)
                    loss = criterion(predictions, source_data)
            else:
                predictions = model(eeg_data)
                loss = criterion(predictions, source_data)
            
            # Calculate MAE for additional metric
            mae = torch.mean(torch.abs(predictions - source_data))
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae.item()
            
            # Update progress bar
            avg_loss = total_loss / (pbar.n + 1)
            avg_mae = total_mae / (pbar.n + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'mae': f'{avg_mae:.6f}'})
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
    """Save model checkpoint with validation"""
    # Check if losses contain NaN or Inf
    if np.isnan(train_loss) or np.isinf(train_loss):
        print(f"WARNING: Not saving checkpoint - train_loss is {'NaN' if np.isnan(train_loss) else 'Inf'}")
        return False
    
    if np.isnan(val_loss) or np.isinf(val_loss):
        print(f"WARNING: Not saving checkpoint - val_loss is {'NaN' if np.isnan(val_loss) else 'Inf'}")
        return False
    
    # Check if model weights contain NaN or Inf
    has_nan_or_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: Parameter '{name}' contains NaN - not saving checkpoint")
            has_nan_or_inf = True
            break
        if torch.isinf(param).any():
            print(f"WARNING: Parameter '{name}' contains Inf - not saving checkpoint")
            has_nan_or_inf = True
            break
    
    if has_nan_or_inf:
        return False
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'D_MODEL': config.D_MODEL,
            'NHEAD': config.NHEAD,
            'NUM_LAYERS': config.NUM_LAYERS,
            'DIM_FEEDFORWARD': config.DIM_FEEDFORWARD,
            'DROPOUT': config.DROPOUT,
            'EEG_CHANNELS': config.EEG_CHANNELS,
            'SOURCE_REGIONS': config.SOURCE_REGIONS
        }
    }
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    return True


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch+1}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
    
    return epoch + 1


def train(config):
    """Main training function"""
    
    # Display configuration
    config.display()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, stats = create_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        num_workers=config.NUM_WORKERS,
        normalize=config.NORMALIZE
    )
    
    # Save normalization statistics
    if stats is not None:
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(stats, os.path.join(config.CHECKPOINT_DIR, 'normalization_stats.pt'))
        print("Normalization statistics saved")
    
    # Create model
    print("\nInitializing model...")
    model = EEGSourceTransformerV2(
        eeg_channels=config.EEG_CHANNELS,
        source_regions=config.SOURCE_REGIONS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    
    # Loss function and optimizer
    criterion = get_loss_function(config.LOSS_FN)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / 
                                  (config.NUM_EPOCHS - config.WARMUP_EPOCHS)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # TensorBoard writer
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.RESUME_CHECKPOINT is not None and os.path.exists(config.RESUME_CHECKPOINT):
        start_epoch = load_checkpoint(model, optimizer, config.RESUME_CHECKPOINT)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                config.DEVICE, config, epoch)
        
        # Check if training produced NaN
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"\n{'='*60}")
            print(f"CRITICAL ERROR: Training produced {'NaN' if np.isnan(train_loss) else 'Inf'} loss!")
            print(f"Training stopped at epoch {epoch+1}")
            print(f"{'='*60}")
            break
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, config.DEVICE, config, epoch)
        
        # Check if validation produced NaN
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"\n{'='*60}")
            print(f"CRITICAL ERROR: Validation produced {'NaN' if np.isnan(val_loss) else 'Inf'} loss!")
            print(f"Training stopped at epoch {epoch+1}")
            print(f"{'='*60}")
            break
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Val MAE: {val_mae:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model (only if not NaN/Inf)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved = save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                                  config, 'best_model.pt')
            if saved:
                print(f"New best model saved! Val Loss: {val_loss:.6f}")
        
        # Save periodic checkpoint (only if not NaN/Inf)
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          config, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                   config, 'final_model.pt')
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae = validate(model, test_loader, criterion, config.DEVICE, config, epoch)
    print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
    
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('MAE/test', test_mae, 0)
    
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EEG Source Localization Transformer')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.data_dir is not None:
        Config.DATA_DIR = args.data_dir
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.resume is not None:
        Config.RESUME_CHECKPOINT = args.resume
    
    # Start training
    train(Config)

