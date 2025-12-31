"""
Training script using Enhanced Transformer V3 Model
This is the same as train_optimized.py but uses the V3 model architecture
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

# Import V3 model instead of V2
from models.transformer_model_v3 import EEGSourceTransformerV3, EEGSourceTransformerV3Large, EEGSourceTransformerV3Small
from utils.dataset import create_dataloaders
from configs.config_gpu_optimized import ConfigGPUOptimized, ConfigBalanced
from configs.config_conservative import ConfigConservative


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (eeg_data, source_data) in enumerate(pbar):
        eeg_data = eeg_data.to(device, non_blocking=True)
        source_data = source_data.to(device, non_blocking=True)
        
        error = check_for_nan_inf(eeg_data, "EEG data")
        if error:
            print(f"\nERROR: {error} in batch {batch_idx}")
            return float('nan')
        
        error = check_for_nan_inf(source_data, "Source data")
        if error:
            print(f"\nERROR: {error} in batch {batch_idx}")
            return float('nan')
        
        optimizer.zero_grad(set_to_none=True)
        
        if config.USE_AMP:
            with autocast():
                predictions = model(eeg_data)
                
                error = check_for_nan_inf(predictions, "Predictions")
                if error:
                    print(f"\nERROR: {error} in batch {batch_idx}")
                    return float('nan')
                
                loss = criterion(predictions, source_data)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is {'NaN' if torch.isnan(loss) else 'Inf'} in batch {batch_idx}")
                return float('nan')
            
            scaler.scale(loss).backward()
            
            if config.CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                
                if grad_norm > 100 * config.CLIP_GRAD_NORM:
                    print(f"\nWARNING: Very large gradient norm: {grad_norm:.2f}")
            
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(eeg_data)
            
            error = check_for_nan_inf(predictions, "Predictions")
            if error:
                print(f"\nERROR: {error} in batch {batch_idx}")
                return float('nan')
            
            loss = criterion(predictions, source_data)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nERROR: Loss is {'NaN' if torch.isnan(loss) else 'Inf'}")
                return float('nan')
            
            loss.backward()
            
            if config.CLIP_GRAD_NORM > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            
            optimizer.step()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
    
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
            eeg_data = eeg_data.to(device, non_blocking=True)
            source_data = source_data.to(device, non_blocking=True)
            
            if config.USE_AMP:
                with autocast():
                    predictions = model(eeg_data)
                    loss = criterion(predictions, source_data)
            else:
                predictions = model(eeg_data)
                loss = criterion(predictions, source_data)
            
            mae = torch.mean(torch.abs(predictions - source_data))
            
            total_loss += loss.item()
            total_mae += mae.item()
            
            avg_loss = total_loss / (pbar.n + 1)
            avg_mae = total_mae / (pbar.n + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'mae': f'{avg_mae:.6f}'})
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, filename, ema=None):
    """Save model checkpoint with validation"""
    if np.isnan(train_loss) or np.isinf(train_loss):
        print(f"WARNING: Not saving checkpoint - train_loss is {'NaN' if np.isnan(train_loss) else 'Inf'}")
        return False
    
    if np.isnan(val_loss) or np.isinf(val_loss):
        print(f"WARNING: Not saving checkpoint - val_loss is {'NaN' if np.isnan(val_loss) else 'Inf'}")
        return False
    
    has_nan_or_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: Parameter '{name}' contains {'NaN' if torch.isnan(param).any() else 'Inf'} - not saving checkpoint")
            has_nan_or_inf = True
            break
    
    if has_nan_or_inf:
        return False
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
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
    
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")
    return True


def train(config, model_size='default'):
    """Main training function with V3 model"""
    
    # Set up GPU optimizations
    if config.DEVICE.type == 'cuda':
        if hasattr(config, 'CUDNN_BENCHMARK') and config.CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
        
        if hasattr(config, 'TF32_ALLOW') and config.TF32_ALLOW:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Display configuration and GPU info
    if hasattr(config, 'verify_gpu'):
        config.verify_gpu()
    
    print("\n" + "=" * 70)
    print(f"Using Enhanced Transformer V3 Model ({model_size.upper()})")
    print("=" * 70)
    
    config.display()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if config.DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    # Create dataloaders
    print("\nLoading datasets...")
    dataloader_kwargs = {
        'data_dir': config.DATA_DIR,
        'batch_size': config.BATCH_SIZE,
        'train_split': config.TRAIN_SPLIT,
        'val_split': config.VAL_SPLIT,
        'num_workers': config.NUM_WORKERS if config.DEVICE.type == 'cuda' else 0,
        'normalize': config.NORMALIZE
    }
    
    train_loader, val_loader, test_loader, stats = create_dataloaders(**dataloader_kwargs)
    
    # Save normalization statistics
    if stats is not None:
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(stats, os.path.join(config.CHECKPOINT_DIR, 'normalization_stats.pt'))
        print("✓ Normalization statistics saved")
    
    # Create V3 model based on size
    print("\nInitializing Enhanced V3 model...")
    if model_size == 'small':
        model = EEGSourceTransformerV3Small(
            eeg_channels=config.EEG_CHANNELS,
            source_regions=config.SOURCE_REGIONS
        ).to(config.DEVICE)
    elif model_size == 'large':
        model = EEGSourceTransformerV3Large(
            eeg_channels=config.EEG_CHANNELS,
            source_regions=config.SOURCE_REGIONS
        ).to(config.DEVICE)
    else:  # default
        model = EEGSourceTransformerV3(
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
    print(f"✓ Total parameters: {num_params:,}")
    print(f"✓ Trainable parameters: {num_trainable_params:,}")
    print(f"✓ Model features: Pre-LayerNorm, GELU, Skip Connections")
    
    # Initialize EMA
    ema = None
    if hasattr(config, 'USE_EMA') and config.USE_EMA:
        ema = ModelEMA(model, decay=config.EMA_DECAY)
        print(f"✓ Model EMA enabled (decay={config.EMA_DECAY})")
    
    # Loss function and optimizer
    criterion = get_loss_function(config.LOSS_FN)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    if hasattr(config, 'USE_COSINE_ANNEALING') and config.USE_COSINE_ANNEALING:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_MULT,
            eta_min=config.ETA_MIN
        )
        print(f"✓ Using CosineAnnealingWarmRestarts scheduler")
    else:
        def lr_lambda(epoch):
            if epoch < config.WARMUP_EPOCHS:
                return (epoch + 1) / config.WARMUP_EPOCHS
            return 0.5 * (1 + np.cos(np.pi * (epoch - config.WARMUP_EPOCHS) / 
                                      (config.NUM_EPOCHS - config.WARMUP_EPOCHS)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"✓ Using Lambda scheduler with warmup")
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP and config.DEVICE.type == 'cuda' else None
    if scaler:
        print(f"✓ Mixed precision training enabled")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # TensorBoard writer
    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting V3 Model Training...")
    print("=" * 70)
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                config.DEVICE, config, epoch, ema)
        
        # Check for NaN
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"\n{'='*70}")
            print(f"CRITICAL ERROR: Training produced {'NaN' if np.isnan(train_loss) else 'Inf'} loss!")
            print(f"Training stopped at epoch {epoch+1}")
            print(f"{'='*70}")
            break
        
        # Validate with EMA if available
        if ema is not None:
            ema.apply_shadow()
            val_loss, val_mae = validate(model, val_loader, criterion, config.DEVICE, config, epoch)
            ema.restore()
        else:
            val_loss, val_mae = validate(model, val_loader, criterion, config.DEVICE, config, epoch)
        
        # Check for NaN
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"\n{'='*70}")
            print(f"CRITICAL ERROR: Validation produced {'NaN' if np.isnan(val_loss) else 'Inf'} loss!")
            print(f"Training stopped at epoch {epoch+1}")
            print(f"{'='*70}")
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
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}")
        print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
        print(f"{'='*70}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved = save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                                  config, 'best_model.pt', ema)
            if saved:
                print(f"★ New best model! Val Loss: {val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                          config, f'checkpoint_epoch_{epoch+1}.pt', ema)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n{'='*70}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"{'='*70}")
            break
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   config, 'final_model.pt', ema)
    
    # Test on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    if ema is not None:
        ema.apply_shadow()
    
    test_loss, test_mae = validate(model, test_loader, criterion, config.DEVICE, config, epoch)
    print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
    
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('MAE/test', test_mae, 0)
    
    writer.close()
    print("\n" + "=" * 70)
    print("V3 Model Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with Enhanced Transformer V3 Model')
    parser.add_argument('--config', type=str, default='optimized', 
                       choices=['optimized', 'balanced', 'conservative'],
                       help='Configuration to use')
    parser.add_argument('--model_size', type=str, default='default',
                       choices=['small', 'default', 'large'],
                       help='V3 model size: small (3.5M), default (18M), large (65M)')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'optimized':
        Config = ConfigGPUOptimized
    elif args.config == 'balanced':
        Config = ConfigBalanced
    else:  # conservative
        Config = ConfigConservative
    
    # Update config with command line arguments
    if args.data_dir is not None:
        Config.DATA_DIR = args.data_dir
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    
    # Start training
    train(Config, model_size=args.model_size)

