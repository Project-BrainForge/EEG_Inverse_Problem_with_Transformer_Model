"""
Dataset utilities for loading EEG and source data from .mat files
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import Tuple, List
import glob


class EEGSourceDataset(Dataset):
    """
    Dataset for EEG to Source localization
    
    Args:
        data_dir: Directory containing .mat files
        transform: Optional transform to be applied on a sample
        normalize: Whether to normalize the data
    """
    
    def __init__(self, data_dir: str, transform=None, normalize: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        
        # Get all .mat files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .mat files found in {data_dir}")
        
        print(f"Found {len(self.file_paths)} samples in {data_dir}")
        
        # Load one sample to get dimensions
        sample = loadmat(self.file_paths[0])
        self.eeg_shape = sample['eeg_data'].shape  # (500, 75)
        self.source_shape = sample['source_data'].shape  # (500, 994)
        
        print(f"EEG shape: {self.eeg_shape}, Source shape: {self.source_shape}")
        
        # Compute statistics for normalization if needed
        if self.normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std for normalization"""
        print("Computing dataset statistics for normalization...")
        
        eeg_list = []
        source_list = []
        
        # Sample a subset for statistics (to save memory)
        sample_indices = np.linspace(0, len(self.file_paths)-1, 
                                    min(len(self.file_paths), 100), 
                                    dtype=int)
        
        for idx in sample_indices:
            data = loadmat(self.file_paths[idx])
            eeg_list.append(data['eeg_data'])
            source_list.append(data['source_data'])
        
        eeg_data = np.concatenate(eeg_list, axis=0)
        source_data = np.concatenate(source_list, axis=0)
        
        self.eeg_mean = np.mean(eeg_data, axis=0, keepdims=True)
        self.eeg_std = np.std(eeg_data, axis=0, keepdims=True) + 1e-8
        
        self.source_mean = np.mean(source_data, axis=0, keepdims=True)
        self.source_std = np.std(source_data, axis=0, keepdims=True) + 1e-8
        
        print("Statistics computed successfully")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            eeg_data: Tensor of shape (seq_len, n_channels) = (500, 75)
            source_data: Tensor of shape (seq_len, n_sources) = (500, 994)
        """
        # Load .mat file
        data = loadmat(self.file_paths[idx])
        
        eeg_data = data['eeg_data'].astype(np.float32)  # (500, 75)
        source_data = data['source_data'].astype(np.float32)  # (500, 994)
        
        # Normalize
        if self.normalize:
            eeg_data = (eeg_data - self.eeg_mean) / self.eeg_std
            source_data = (source_data - self.source_mean) / self.source_std
        
        # Convert to tensors
        eeg_tensor = torch.from_numpy(eeg_data)
        source_tensor = torch.from_numpy(source_data)
        
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)
        
        return eeg_tensor, source_tensor
    
    def get_statistics(self):
        """Return normalization statistics"""
        if self.normalize:
            return {
                'eeg_mean': self.eeg_mean,
                'eeg_std': self.eeg_std,
                'source_mean': self.source_mean,
                'source_std': self.source_std
            }
        return None


def create_dataloaders(data_dir: str, 
                       batch_size: int = 8,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       num_workers: int = 4,
                       normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing .mat files
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of workers for data loading
        normalize: Whether to normalize the data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = EEGSourceDataset(data_dir, normalize=normalize)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, full_dataset.get_statistics()


if __name__ == "__main__":
    # Test the dataset
    data_dir = "../dataset_with_label"
    dataset = EEGSourceDataset(data_dir)
    
    print(f"Dataset size: {len(dataset)}")
    eeg, source = dataset[0]
    print(f"EEG shape: {eeg.shape}, Source shape: {source.shape}")

