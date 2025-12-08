"""
Helper utilities for training and evaluation
"""

import os
import torch
import numpy as np
from pathlib import Path


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, mode='min', delta=0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max' - whether lower or higher is better
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.delta
        else:
            return score > self.best_score + self.delta


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state and other info
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_filepath)
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(checkpoint_path, model=None, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
    
    Returns:
        checkpoint: Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    return checkpoint


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics
    
    Args:
        predictions: Predicted values (batch, time, channels)
        targets: Target values (batch, time, channels)
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Correlation coefficient
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Relative error
    relative_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'r2': r2,
        'relative_error': relative_error
    }


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

