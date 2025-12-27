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
import logging
import datetime
import traceback

from models import create_model
from utils.loader import create_dataloaders_from_metadata, create_dataloaders_from_spikes
from configs.config import Config


def setup_logger(log_dir, log_name='training'):
    """
    Setup logger for training
    
    Parameters
    ----------
    log_dir : str
        Directory to save log files
    log_name : str
        Name prefix for log file
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('EEG_Training')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for all logs (detailed)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    all_log_file = os.path.join(log_dir, f'{log_name}_{timestamp}_all.log')
    file_handler_all = logging.FileHandler(all_log_file, mode='w')
    file_handler_all.setLevel(logging.DEBUG)
    file_handler_all.setFormatter(detailed_formatter)
    logger.addHandler(file_handler_all)
    
    # File handler for errors only
    error_log_file = os.path.join(log_dir, f'{log_name}_{timestamp}_errors.log')
    file_handler_error = logging.FileHandler(error_log_file, mode='w')
    file_handler_error.setLevel(logging.ERROR)
    file_handler_error.setFormatter(detailed_formatter)
    logger.addHandler(file_handler_error)
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Logs saved to: {log_dir}")
    logger.info(f"  All logs: {all_log_file}")
    logger.info(f"  Errors only: {error_log_file}")
    
    return logger


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


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config, epoch, logger=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    num_errors = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (eeg_data, source_data) in enumerate(pbar):
        try:
            # Move data to device
            eeg_data = eeg_data.to(device)
            source_data = source_data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if config.USE_AMP:
                with autocast():
                    predictions = model(eeg_data)
                    loss = criterion(predictions, source_data)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(eeg_data)
                loss = criterion(predictions, source_data)
                loss.backward()
                
                # Gradient clipping
                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                
                optimizer.step()
            
            # Check for NaN loss
            if torch.isnan(loss):
                error_msg = f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}"
                if logger:
                    logger.error(error_msg)
                    logger.error(f"EEG data stats: min={eeg_data.min()}, max={eeg_data.max()}, mean={eeg_data.mean()}")
                    logger.error(f"Source data stats: min={source_data.min()}, max={source_data.max()}, mean={source_data.mean()}")
                print(f"\nERROR: {error_msg}")
                num_errors += 1
                continue
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1 - num_errors)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'errors': num_errors})
            
            # Log periodically
            if logger and (batch_idx + 1) % 100 == 0:
                logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.6f}")
        
        except Exception as e:
            error_msg = f"Error in batch {batch_idx} at epoch {epoch+1}: {str(e)}"
            if logger:
                logger.error(error_msg)
                logger.error(traceback.format_exc())
            print(f"\nERROR: {error_msg}")
            num_errors += 1
            continue
    
    if logger and num_errors > 0:
        logger.warning(f"Epoch {epoch+1}: {num_errors} batches failed out of {num_batches}")
    
    return total_loss / (num_batches - num_errors) if num_batches > num_errors else float('inf')


def validate(model, val_loader, criterion, device, config, epoch, logger=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = len(val_loader)
    num_errors = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
    
    with torch.no_grad():
        for batch_idx, (eeg_data, source_data) in enumerate(pbar):
            try:
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
                
                # Check for NaN loss
                if torch.isnan(loss):
                    error_msg = f"NaN loss in validation at epoch {epoch+1}, batch {batch_idx}"
                    if logger:
                        logger.error(error_msg)
                    print(f"\nERROR: {error_msg}")
                    num_errors += 1
                    continue
                
                # Calculate MAE for additional metric
                mae = torch.mean(torch.abs(predictions - source_data))
                
                # Update metrics
                total_loss += loss.item()
                total_mae += mae.item()
                
                # Update progress bar
                avg_loss = total_loss / (pbar.n + 1 - num_errors)
                avg_mae = total_mae / (pbar.n + 1 - num_errors)
                pbar.set_postfix({'loss': f'{avg_loss:.6f}', 'mae': f'{avg_mae:.6f}', 'errors': num_errors})
            
            except Exception as e:
                error_msg = f"Error in validation batch {batch_idx} at epoch {epoch+1}: {str(e)}"
                if logger:
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                print(f"\nERROR: {error_msg}")
                num_errors += 1
                continue
    
    if logger and num_errors > 0:
        logger.warning(f"Validation epoch {epoch+1}: {num_errors} batches failed out of {num_batches}")
    
    valid_batches = num_batches - num_errors
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    avg_mae = total_mae / valid_batches if valid_batches > 0 else float('inf')
    
    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
    """Save model checkpoint"""
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
    
    # Setup logger first
    logger = setup_logger(config.LOG_DIR, 'training')
    
    try:
        # Display configuration
        config.display()
        logger.info("="*70)
        logger.info("Training Configuration")
        logger.info("="*70)
        for key, value in config.__dict__.items():
            if not key.startswith('_') and not callable(value):
                logger.info(f"{key}: {value}")
        logger.info("="*70)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        logger.info("Random seeds set to 42")
        
        # Check for Windows multiprocessing issue
        import platform
        if platform.system() == 'Windows' and config.NUM_WORKERS > 0:
            logger.warning(f"Detected Windows OS with NUM_WORKERS={config.NUM_WORKERS}. "
                          "Setting NUM_WORKERS=0 to avoid multiprocessing errors.")
            print(f"WARNING: On Windows, setting NUM_WORKERS=0 to avoid multiprocessing errors.")
            config.NUM_WORKERS = 0
        
        # Create dataloaders
        print("\nLoading datasets...")
        logger.info("Loading datasets...")
        
        # Choose between metadata-based loader or dynamic generation
        use_metadata = getattr(config, 'USE_METADATA_LOADER', False)
        
        if use_metadata:
            print("Using metadata-based loader (train_sample_source1.mat)")
            logger.info("Using metadata-based loader")
            train_loader, val_loader, test_loader, stats = create_dataloaders_from_metadata(
                train_metadata_path=config.TRAIN_METADATA_PATH,
                test_metadata_path=config.TEST_METADATA_PATH,
                fwd_matrix_path=config.FWD_MATRIX_PATH,
                batch_size=config.BATCH_SIZE,
                val_split=config.VAL_SPLIT,
                num_workers=config.NUM_WORKERS,
                nmm_spikes_dir=getattr(config, 'NMM_SPIKES_DIR', None),
                train_dataset_len=getattr(config, 'TRAIN_DATASET_LEN', None),
                test_dataset_len=getattr(config, 'TEST_DATASET_LEN', None),
            )
        else:
            print("Using dynamic generation loader")
            logger.info("Using dynamic generation loader")
            train_loader, val_loader, test_loader, stats = create_dataloaders_from_spikes(
                data_root=config.DATA_DIR,
                fwd_matrix_path=config.FWD_MATRIX_PATH,
                batch_size=config.BATCH_SIZE,
                train_split=config.TRAIN_SPLIT,
                val_split=config.VAL_SPLIT,
                num_workers=config.NUM_WORKERS,
                dataset_len=getattr(config, 'DATASET_LEN', 1000),
                num_sources=getattr(config, 'NUM_SOURCES', 2),
                patch_size=getattr(config, 'PATCH_SIZE', 20),
                snr_range=getattr(config, 'SNR_RANGE', (0, 30))
            )
        
        logger.info(f"Datasets loaded: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
        
        # Save normalization statistics
        if stats is not None:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(stats, os.path.join(config.CHECKPOINT_DIR, 'normalization_stats.pt'))
            print("Normalization statistics saved")
            logger.info("Normalization statistics saved")
        
        # Create model via factory (supports 'transformer' or 'hybrid')
        print("\nInitializing model...")
        logger.info("Initializing model...")
        model = create_model(getattr(config, 'MODEL_TYPE', 'transformer'), config).to(config.DEVICE)
        logger.info(f"Model type: {getattr(config, 'MODEL_TYPE', 'transformer')}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable_params:,}")
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {num_trainable_params:,}")
        logger.info(f"Device: {config.DEVICE}")
        
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
            logger.info(f"Resuming from checkpoint: {config.RESUME_CHECKPOINT}")
            start_epoch = load_checkpoint(model, optimizer, config.RESUME_CHECKPOINT)
        
        # Training loop
        print("\nStarting training...")
        logger.info("Starting training loop...")
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            try:
                epoch_start_time = time.time()
                logger.info(f"Starting epoch {epoch+1}/{config.NUM_EPOCHS}")
                
                # Train
                train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                        config.DEVICE, config, epoch, logger)
                
                # Validate
                val_loss, val_mae = validate(model, val_loader, criterion, config.DEVICE, config, epoch, logger)
                
                # Update learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('MAE/val', val_mae, epoch)
                writer.add_scalar('Learning_rate', current_lr, epoch)
                
                epoch_time = time.time() - epoch_start_time
                
                log_msg = (f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                          f"Val MAE: {val_mae:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
                print(f"\n{log_msg}")
                logger.info(log_msg)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                                  config, 'best_model.pt')
                    logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
                
                # Save periodic checkpoint
                if (epoch + 1) % config.SAVE_EVERY == 0:
                    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                                  config, f'checkpoint_epoch_{epoch+1}.pt')
                    logger.info(f"Periodic checkpoint saved at epoch {epoch+1}")
                
                # Early stopping
                if early_stopping(val_loss):
                    stop_msg = f"Early stopping triggered after {epoch+1} epochs"
                    print(f"\n{stop_msg}")
                    logger.info(stop_msg)
                    break
            
            except Exception as e:
                error_msg = f"Error in epoch {epoch+1}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                print(f"\nERROR: {error_msg}")
                
                # Save emergency checkpoint
                try:
                    emergency_path = f'emergency_checkpoint_epoch_{epoch+1}.pt'
                    save_checkpoint(model, optimizer, epoch, train_loss if 'train_loss' in locals() else 0, 
                                  val_loss if 'val_loss' in locals() else 0, config, emergency_path)
                    logger.info(f"Emergency checkpoint saved: {emergency_path}")
                except:
                    logger.error("Failed to save emergency checkpoint")
                
                # Continue to next epoch or stop based on error severity
                continue
        
        # Save final model
        try:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                           config, 'final_model.pt')
            logger.info("Final model saved")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Test on test set
        try:
            print("\nEvaluating on test set...")
            logger.info("Evaluating on test set...")
            test_loss, test_mae = validate(model, test_loader, criterion, config.DEVICE, config, epoch, logger)
            test_msg = f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}"
            print(test_msg)
            logger.info(test_msg)
            
            writer.add_scalar('Loss/test', test_loss, 0)
            writer.add_scalar('MAE/test', test_mae, 0)
        except Exception as e:
            logger.error(f"Error during test evaluation: {e}")
            logger.error(traceback.format_exc())
        
        writer.close()
        print("\nTraining completed!")
        logger.info("Training completed successfully!")
    
    except Exception as e:
        error_msg = f"Fatal error during training: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(f"\nFATAL ERROR: {error_msg}")
        raise


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

