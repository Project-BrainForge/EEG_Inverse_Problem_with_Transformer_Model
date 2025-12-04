"""
Training script for EEG source localization transformer
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.transformer import create_model
from data.dataset import create_dataloaders
from utils.helpers import save_checkpoint, load_checkpoint, EarlyStopping, AverageMeter


class Trainer:
    """Trainer class for EEG source localization"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
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
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        if config['optimizer']['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['optimizer']['lr'],
                weight_decay=config['optimizer']['weight_decay']
            )
        elif config['optimizer']['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['optimizer']['lr'],
                weight_decay=config['optimizer']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']['type']}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            mode='min'
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Load checkpoint if specified
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        
        end = time.time()
        
        for batch_idx, (eeg_data, source_data) in enumerate(self.train_loader):
            # Move to device
            eeg_data = eeg_data.to(self.device)  # (batch, 500, 75)
            source_data = source_data.to(self.device)  # (batch, 500, 994)
            
            # Forward pass
            predictions = self.model(eeg_data)  # (batch, 500, 994)
            
            # Compute loss
            loss = self.criterion(predictions, source_data)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), eeg_data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Logging
            if batch_idx % self.config['log_interval'] == 0:
                print(f"Epoch [{epoch}/{self.config['num_epochs']}] "
                      f"Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {losses.avg:.6f} "
                      f"Time: {batch_time.avg:.3f}s")
                
                self.writer.add_scalar('train/loss', losses.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return losses.avg
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for eeg_data, source_data in self.val_loader:
                eeg_data = eeg_data.to(self.device)
                source_data = source_data.to(self.device)
                
                # Forward pass
                predictions = self.model(eeg_data)
                
                # Compute loss
                loss = self.criterion(predictions, source_data)
                losses.update(loss.item(), eeg_data.size(0))
                
                # Store for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(source_data.cpu())
        
        # Compute additional metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Correlation coefficient
        pred_flat = all_predictions.reshape(-1)
        target_flat = all_targets.reshape(-1)
        correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
        
        # Mean absolute error
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        print(f"\nValidation - Epoch [{epoch}/{self.config['num_epochs']}]")
        print(f"  Loss: {losses.avg:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/mae', mae, epoch)
        self.writer.add_scalar('val/correlation', correlation, epoch)
        
        return losses.avg, mae, correlation
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mae, val_corr = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                },
                is_best=is_best,
                checkpoint_dir=str(self.checkpoint_dir),
                filename=f'checkpoint_epoch_{epoch}.pth'
            )
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*50 + "\n")
        
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {self.start_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train EEG source localization transformer')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.resume:
        config['resume_checkpoint'] = args.resume
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

