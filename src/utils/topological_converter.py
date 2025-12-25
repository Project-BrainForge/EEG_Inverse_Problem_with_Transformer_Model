"""
EEG Topological Converter
Converts raw EEG channel data to 2D spatial topological maps for CNN processing.
Uses MNE library for electrode positioning and scipy for interpolation.
"""

import numpy as np
import scipy.io
import torch
from scipy.interpolate import griddata
import mne
from mne.channels import make_dig_montage
from mne.channels.layout import _find_topomap_coords


class EEGTopologicalConverter:
    """
    Fast converter from EEG data to spatial topological tensors for CNN input.
    
    Uses direct interpolation without rendering images for maximum speed.
    
    Parameters
    ----------
    electrode_file : str
        Path to the electrode configuration file
    image_size : tuple
        Output tensor size as (height, width) (default: (64, 64))
    sphere : float or str
        Sphere radius for projection (default: 'auto')
    normalize : bool
        Whether to normalize output to [0, 1] (default: True)
    """
    
    def __init__(self, 
                 electrode_file='anatomy/electrode_75.mat',
                 image_size=(64, 64),
                 sphere='auto',
                 normalize=True):
        
        self.image_size = image_size
        self.sphere = sphere
        self.normalize = normalize
        self._load_electrodes(electrode_file)
        self._setup_interpolation_grid()
        
    def _load_electrodes(self, electrode_file):
        """Load electrode positions and create MNE Info object."""
        data = scipy.io.loadmat(electrode_file)
        eloc = data['eloc75'][0]
        
        self.num_channels = len(eloc)
        positions = {}
        self.channel_names = []
        
        for electrode in eloc:
            ch_name = electrode['labels'][0]
            self.channel_names.append(ch_name)
            
            x = float(electrode['X'][0][0])
            y = float(electrode['Y'][0][0])
            z = float(electrode['Z'][0][0])
            positions[ch_name] = np.array([x, y, z])
        
        self.montage = make_dig_montage(ch_pos=positions, coord_frame='head')
        
        self.info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=1000,
            ch_types='eeg'
        )
        self.info.set_montage(self.montage)
        
        # Get 2D positions for interpolation
        self.pos = _find_topomap_coords(self.info, picks=None, sphere=self.sphere)
        
    def _setup_interpolation_grid(self):
        """Setup the interpolation grid for fast processing."""
        # Create meshgrid for interpolation
        xi = np.linspace(-1, 1, self.image_size[0])
        yi = np.linspace(-1, 1, self.image_size[1])
        self.Xi, self.Yi = np.meshgrid(xi, yi)
        
        # Create mask for head shape (circular)
        self.mask = (self.Xi**2 + self.Yi**2) <= 1.0
        
    def convert_single_timepoint(self, eeg_data, vmin=None, vmax=None):
        """
        Convert EEG data at a single time point to a 2D tensor.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data of shape (num_channels,)
        vmin : float, optional
            Minimum value for normalization
        vmax : float, optional
            Maximum value for normalization
            
        Returns
        -------
        tensor : np.ndarray
            2D tensor of shape (height, width)
        """
        assert len(eeg_data) == self.num_channels, \
            f"Expected {self.num_channels} channels, got {len(eeg_data)}"
        
        # Interpolate values to grid
        values = griddata(
            self.pos, 
            eeg_data, 
            (self.Xi, self.Yi), 
            method='cubic', 
            fill_value=0
        )
        
        # Apply head mask
        values = np.where(self.mask, values, 0)
        
        # Normalize if requested
        if self.normalize:
            if vmin is None or vmax is None:
                absmax = np.abs(eeg_data).max()
                vmin = -absmax if vmin is None else vmin
                vmax = absmax if vmax is None else vmax
            
            # Scale to [0, 1]
            values = (values - vmin) / (vmax - vmin + 1e-10)
            values = np.clip(values, 0, 1)
        
        return values.astype(np.float32)
    
    def convert_timeseries(self, eeg_data):
        """
        Convert EEG time series to a 3D tensor.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data of shape (num_timepoints, num_channels) or (num_channels, num_timepoints)
            
        Returns
        -------
        tensor : np.ndarray
            3D tensor of shape (num_timepoints, height, width)
        """
        # Handle both (time, channels) and (channels, time) formats
        if eeg_data.shape[0] == self.num_channels:
            # Input is (channels, time) - transpose to (time, channels)
            eeg_data = eeg_data.T
        
        num_timepoints, num_channels = eeg_data.shape
        assert num_channels == self.num_channels, \
            f"Expected {self.num_channels} channels, got {num_channels}"
        
        # Determine global vmin and vmax for consistent scaling
        vmin = -np.abs(eeg_data).max()
        vmax = np.abs(eeg_data).max()
        
        # Pre-allocate output tensor
        output = np.zeros((num_timepoints, self.image_size[0], self.image_size[1]), 
                         dtype=np.float32)
        
        # Convert each time point
        for t in range(num_timepoints):
            output[t] = self.convert_single_timepoint(eeg_data[t], vmin, vmax)
        
        return output
    
    def convert_batch(self, eeg_data, verbose=False):
        """
        Convert a batch of EEG time series to a 4D tensor.
        
        Parameters
        ----------
        eeg_data : np.ndarray or torch.Tensor
            EEG data of shape (batch_size, num_timepoints, num_channels)
        verbose : bool
            Whether to print progress (default: False)
        
        Returns
        -------
        tensor : np.ndarray
            4D tensor of shape (batch_size, num_timepoints, height, width)
        """
        # Convert torch tensor to numpy if needed
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.detach().cpu().numpy()
        
        batch_size, num_timepoints, num_channels = eeg_data.shape
        assert num_channels == self.num_channels, \
            f"Expected {self.num_channels} channels, got {num_channels}"
        
        # Pre-allocate output tensor
        output = np.zeros((batch_size, num_timepoints, 
                          self.image_size[0], self.image_size[1]), 
                         dtype=np.float32)
        
        # Convert each sample in the batch
        if verbose:
            print(f"Converting batch of {batch_size} samples...")
        
        for b in range(batch_size):
            if verbose and (b % 10 == 0 or b == batch_size - 1):
                print(f"  Processing sample {b+1}/{batch_size}")
            output[b] = self.convert_timeseries(eeg_data[b])
        
        if verbose:
            print("Batch conversion complete!")
        
        return output
    
    def to_torch(self, eeg_data, device='cpu', verbose=False):
        """
        Convert EEG data to PyTorch tensor ready for CNN.
        
        Parameters
        ----------
        eeg_data : np.ndarray or torch.Tensor
            EEG data of shape (batch_size, num_timepoints, num_channels)
        device : str or torch.device
            Device to place tensor on (default: 'cpu')
        verbose : bool
            Whether to print progress (default: False)
            
        Returns
        -------
        tensor : torch.Tensor
            PyTorch tensor of shape (batch_size, num_timepoints, height, width)
        """
        # Convert to numpy tensor
        numpy_tensor = self.convert_batch(eeg_data, verbose=verbose)
        
        # Convert to PyTorch
        torch_tensor = torch.from_numpy(numpy_tensor)
        
        # Move to device
        torch_tensor = torch_tensor.to(device)
        
        return torch_tensor


def test_converter():
    """Test the topological converter with sample data."""
    print("="*60)
    print("Testing EEG Topological Converter")
    print("="*60)
    
    # Create converter
    converter = EEGTopologicalConverter(
        electrode_file='anatomy/electrode_75.mat',
        image_size=(64, 64),
        sphere='auto',
        normalize=True
    )
    
    print(f"\nConverter initialized:")
    print(f"  Channels: {converter.num_channels}")
    print(f"  Image size: {converter.image_size}")
    print(f"  Normalize: {converter.normalize}")
    
    # Test single sample
    print("\n" + "-"*60)
    print("Test 1: Single Sample Conversion")
    print("-"*60)
    
    eeg_single = np.random.randn(500, 75).astype(np.float32)
    print(f"Input shape: {eeg_single.shape}")
    
    tensor_single = converter.convert_timeseries(eeg_single)
    print(f"Output shape: {tensor_single.shape}")
    print(f"Output dtype: {tensor_single.dtype}")
    print(f"Value range: [{tensor_single.min():.3f}, {tensor_single.max():.3f}]")
    
    # Test batch
    print("\n" + "-"*60)
    print("Test 2: Batch Conversion")
    print("-"*60)
    
    batch_size = 8
    eeg_batch = np.random.randn(batch_size, 500, 75).astype(np.float32)
    print(f"Input batch shape: {eeg_batch.shape}")
    
    tensor_batch = converter.to_torch(eeg_batch, device='cpu', verbose=True)
    print(f"Output batch shape: {tensor_batch.shape}")
    print(f"Output dtype: {tensor_batch.dtype}")
    print(f"Memory usage: {tensor_batch.element_size() * tensor_batch.nelement() / 1024**2:.2f} MB")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_converter()
