"""
Dataset class for EEG source localization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from pathlib import Path
import glob


class EEGSourceDataset(Dataset):
    """
    Dataset for EEG to source localization
    
    Loads .mat files containing:
    - eeg_data: (500, 75) - EEG signals
    - source_data: (500, 994) - Brain source activity
    """
    
    def __init__(self, data_dir, normalize=True, split='train', train_ratio=0.8, seed=42):
        """
        Args:
            data_dir: Directory containing .mat files
            normalize: Whether to normalize the data
            split: 'train', 'val', or 'all'
            train_ratio: Ratio of training data (rest is validation)
            seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.split = split
        
        # Load all sample files
        self.sample_files = sorted(glob.glob(str(self.data_dir / "sample_*.mat")))
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No .mat files found in {data_dir}")
        
        print(f"Found {len(self.sample_files)} samples in {data_dir}")
        
        # Split into train and validation
        if split != 'all':
            np.random.seed(seed)
            indices = np.arange(len(self.sample_files))
            np.random.shuffle(indices)
            
            n_train = int(len(self.sample_files) * train_ratio)
            
            if split == 'train':
                indices = indices[:n_train]
            elif split == 'val':
                indices = indices[n_train:]
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.sample_files = [self.sample_files[i] for i in indices]
        
        print(f"Using {len(self.sample_files)} samples for {split}")
        
        # Compute normalization statistics from training data
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        print("Computing normalization statistics...")
        
        eeg_data_list = []
        source_data_list = []
        
        # Sample a subset for computing stats (to save memory)
        sample_indices = np.linspace(0, len(self.sample_files) - 1, 
                                     min(len(self.sample_files), 100), 
                                     dtype=int)
        
        for idx in sample_indices:
            data = loadmat(self.sample_files[idx])
            eeg_data_list.append(data['eeg_data'])
            source_data_list.append(data['source_data'])
        
        # Stack and compute statistics
        eeg_data_array = np.stack(eeg_data_list, axis=0)  # (N, 500, 75)
        source_data_array = np.stack(source_data_list, axis=0)  # (N, 500, 994)
        
        # Compute channel-wise statistics
        self.eeg_mean = eeg_data_array.mean(axis=(0, 1))  # (75,)
        self.eeg_std = eeg_data_array.std(axis=(0, 1)) + 1e-8  # (75,)
        
        self.source_mean = source_data_array.mean(axis=(0, 1))  # (994,)
        self.source_std = source_data_array.std(axis=(0, 1)) + 1e-8  # (994,)
        
        print(f"EEG mean: {self.eeg_mean[:5]}... | std: {self.eeg_std[:5]}...")
        print(f"Source mean: {self.source_mean[:5]}... | std: {self.source_std[:5]}...")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            eeg_data: Tensor of shape (500, 75)
            source_data: Tensor of shape (500, 994)
        """
        # Load .mat file
        data = loadmat(self.sample_files[idx])
        
        eeg_data = data['eeg_data'].astype(np.float32)  # (500, 75)
        source_data = data['source_data'].astype(np.float32)  # (500, 994)
        
        # Normalize
        if self.normalize:
            eeg_data = (eeg_data - self.eeg_mean) / self.eeg_std
            source_data = (source_data - self.source_mean) / self.source_std
        
        # Convert to tensors
        eeg_data = torch.from_numpy(eeg_data)
        source_data = torch.from_numpy(source_data)
        
        return eeg_data, source_data
    
    def denormalize_source(self, source_data):
        """
        Denormalize source data back to original scale
        
        Args:
            source_data: Tensor of shape (..., 994)
        
        Returns:
            Denormalized tensor
        """
        if not self.normalize:
            return source_data
        
        if isinstance(source_data, torch.Tensor):
            mean = torch.tensor(self.source_mean, device=source_data.device)
            std = torch.tensor(self.source_std, device=source_data.device)
        else:
            mean = self.source_mean
            std = self.source_std
        
        return source_data * std + mean


def create_dataloaders(data_dir, batch_size=8, num_workers=4, train_ratio=0.8, seed=42):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Directory containing .mat files
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_ratio: Ratio of training data
        seed: Random seed
    
    Returns:
        train_loader, val_loader, dataset objects
    """
    # Create datasets
    train_dataset = EEGSourceDataset(
        data_dir=data_dir,
        normalize=True,
        split='train',
        train_ratio=train_ratio,
        seed=seed
    )
    
    val_dataset = EEGSourceDataset(
        data_dir=data_dir,
        normalize=True,
        split='val',
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Share normalization statistics
    val_dataset.eeg_mean = train_dataset.eeg_mean
    val_dataset.eeg_std = train_dataset.eeg_std
    val_dataset.source_mean = train_dataset.source_mean
    val_dataset.source_std = train_dataset.source_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    data_dir = "../../dataset_with_label"
    
    # Create dataset
    dataset = EEGSourceDataset(data_dir, split='all')
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    eeg, source = dataset[0]
    print(f"\nSample shapes:")
    print(f"  EEG: {eeg.shape}")
    print(f"  Source: {source.shape}")
    print(f"\nSample statistics:")
    print(f"  EEG - min: {eeg.min():.4f}, max: {eeg.max():.4f}, mean: {eeg.mean():.4f}")
    print(f"  Source - min: {source.min():.4f}, max: {source.max():.4f}, mean: {source.mean():.4f}")
    
    # Test dataloader
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        data_dir, batch_size=4, num_workers=0
    )
    
    print(f"\nDataloader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test batch loading
    for eeg_batch, source_batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  EEG: {eeg_batch.shape}")
        print(f"  Source: {source_batch.shape}")
        break

