"""
Data loader for EEG source localization using nmm_spikes data

Supports two modes:
1. Metadata-based: Uses train_sample_source1.mat with selected_region, nmm_idx, etc.
2. Dynamic generation: Randomly generates samples from region folders
"""
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
from scipy import interpolate
import random
import os
import glob
import torch


def load_octave_text_file(file_path):
    """Load Octave text format file
    
    Parameters
    ----------
    file_path : str
        Path to Octave text file
    
    Returns
    -------
    dict
        Dictionary containing the data
    """
    data = {}
    current_var = None
    current_type = None
    current_dims = None
    current_data = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith("#"):
                if "name:" in line:
                    # Save previous variable if exists
                    if current_var is not None and current_data:
                        if current_type == "matrix":
                            arr = np.array(current_data)
                            if current_dims is not None and len(current_dims) > 1:
                                arr = arr.reshape(
                                    current_dims, order="F"
                                )  # Fortran order for MATLAB compatibility
                            data[current_var] = arr
                        elif current_type == "scalar":
                            data[current_var] = current_data[0] if current_data else 0
                    
                    # Start new variable
                    current_var = line.split("name:")[1].strip()
                    current_data = []
                    current_dims = None
                elif "type:" in line:
                    current_type = line.split("type:")[1].strip()
                elif "ndims:" in line or "rows:" in line or "columns:" in line:
                    continue
                continue
            
            # Parse dimensions
            if current_dims is None and current_type == "matrix":
                try:
                    dims = [int(x) for x in line.split()]
                    if len(dims) >= 1:
                        current_dims = dims
                        continue
                except ValueError:
                    pass
            
            # Parse data values
            if line and not line.startswith("#"):
                try:
                    values = [float(x) for x in line.split()]
                    current_data.extend(values)
                except ValueError:
                    # Try as single value
                    try:
                        current_data.append(float(line))
                    except ValueError:
                        pass
        
        # Save last variable
        if current_var is not None and current_data:
            if current_type == "matrix":
                arr = np.array(current_data)
                if current_dims is not None and len(current_dims) > 1:
                    arr = arr.reshape(
                        current_dims, order="F"
                    )  # Fortran order for MATLAB compatibility
                data[current_var] = arr
            elif current_type == "scalar":
                data[current_var] = current_data[0] if current_data else 0
    
    return data


def load_mat_file(file_path):
    """Load MAT file, handling v7, v7.3 (HDF5), and Octave text formats
    
    Parameters
    ----------
    file_path : str
        Path to MAT file
    
    Returns
    -------
    dict
        Dictionary containing the MAT file data
    """
    try:
        # Try loading with scipy first (for MAT files v7 and earlier)
        return loadmat(file_path)
    except (ValueError, NotImplementedError, OSError):
        pass
    
    # Try Octave text format
    try:
        return load_octave_text_file(file_path)
    except Exception as e:
        raise ValueError(
            f"Could not load file {file_path}. Tried MATLAB v7, v7.3 (HDF5), and Octave text formats. Error: {e}"
        )


def ispadding(arr):
    """Check which elements are padding values (marked as >= 999)"""
    return arr >= 999


def add_white_noise(signal, snr_db):
    """Add white Gaussian noise based on SNR in dB"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


class SpikeEEGMetadataDataset(Dataset):
    """
    Dataset that loads from metadata file (train_sample_source1.mat)
    
    This is the direct replacement for the old SpikeEEGBuild class.
    It reads metadata about source configurations and loads NMM data on-the-fly.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata file (e.g., 'source/train_sample_source1.mat')
    fwd : np.array
        Forward matrix: (num_electrodes, num_regions), e.g., (75, 994)
    nmm_spikes_dir : str
        Path to nmm_spikes directory containing a0/, a1/, etc. folders
    dataset_len : int, optional
        Number of samples to use (default: use all from metadata)
    normalize : bool
        Whether to normalize the output (default: True)
    """
    
    def __init__(
        self,
        metadata_path,
        fwd,
        nmm_spikes_dir=None,
        dataset_len=None,
        normalize=True,
    ):
        self.metadata_path = metadata_path
        self.fwd = fwd
        self.normalize = normalize
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        self.dataset_meta = load_mat_file(metadata_path)
        
        # Determine dataset length
        if dataset_len is None:
            self.dataset_len = self.dataset_meta["selected_region"].shape[0]
        else:
            self.dataset_len = min(dataset_len, self.dataset_meta["selected_region"].shape[0])
        
        print(f"Dataset length: {self.dataset_len}")
        
        # Determine nmm_spikes directory
        if nmm_spikes_dir is None:
            # Try to find it relative to metadata path
            metadata_dir = os.path.dirname(metadata_path)
            if os.path.exists(os.path.join(metadata_dir, "nmm_spikes")):
                self.nmm_spikes_dir = os.path.join(metadata_dir, "nmm_spikes")
            elif os.path.exists("source/nmm_spikes"):
                self.nmm_spikes_dir = "source/nmm_spikes"
            else:
                raise ValueError("Could not find nmm_spikes directory")
        else:
            self.nmm_spikes_dir = nmm_spikes_dir
        
        print(f"Using nmm_spikes directory: {self.nmm_spikes_dir}")
        
        # Index all available mat files in each region folder
        self.region_files = {}
        self.num_regions = 994
        
        print("Indexing region files...")
        for region_idx in range(self.num_regions):
            region_dir = os.path.join(self.nmm_spikes_dir, f"a{region_idx}")
            if os.path.exists(region_dir):
                mat_files = sorted(glob.glob(os.path.join(region_dir, "*.mat")))
                if len(mat_files) > 0:
                    self.region_files[region_idx] = mat_files
        
        print(f"Found {len(self.region_files)} regions with data")
        
        # Determine num_scale_ratio
        if "scale_ratio" in self.dataset_meta:
            scale_ratio_shape = self.dataset_meta["scale_ratio"].shape
            if len(scale_ratio_shape) >= 3:
                self.num_scale_ratio = scale_ratio_shape[2]
            elif len(scale_ratio_shape) >= 2:
                self.num_scale_ratio = scale_ratio_shape[1]
            else:
                self.num_scale_ratio = 1
        else:
            self.num_scale_ratio = 1
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        """
        Generate one sample based on metadata
        
        Returns
        -------
        tuple
            (eeg_data, source_data) both as torch tensors
            eeg_data: (500, 75) - EEG sensor data
            source_data: (500, 994) - Source space activity
        """
        # Get labels with padding
        raw_lb = self.dataset_meta["selected_region"][index].astype(np.int64)
        
        # Handle different array dimensions
        if raw_lb.ndim == 1:
            raw_lb = raw_lb.reshape(1, -1)
        elif raw_lb.ndim == 3:
            raw_lb = raw_lb.squeeze(1)  # Remove middle dimension if present
        
        # Get labels without padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]
        
        # Initialize source space
        raw_nmm = np.zeros((500, self.fwd.shape[1]))
        
        # Iterate through sources
        num_sources = raw_lb.shape[0]
        for kk in range(num_sources):
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            
            # Check if curr_lb is empty
            if len(curr_lb) == 0:
                continue
            
            # Get NMM index
            if self.dataset_meta["nmm_idx"].ndim == 1:
                nmm_idx = self.dataset_meta["nmm_idx"][index]
            else:
                nmm_idx = self.dataset_meta["nmm_idx"][index][kk]
            
            # Load NMM data
            current_nmm = self._load_nmm_data(int(nmm_idx))
            
            if current_nmm is None:
                continue
            
            # Get center region activity
            ssig = current_nmm[:, [curr_lb[0]]]
            
            # Set source space SNR
            if np.max(ssig) > 0:
                # Get scale ratio
                if self.dataset_meta["scale_ratio"].ndim == 2:
                    scale_ratio_val = self.dataset_meta["scale_ratio"][index][
                        random.randint(0, self.num_scale_ratio - 1)
                    ]
                else:
                    scale_ratio_val = self.dataset_meta["scale_ratio"][index][kk][
                        random.randint(0, self.num_scale_ratio - 1)
                    ]
                
                # Handle NaN
                if np.isnan(scale_ratio_val):
                    scale_ratio_val = 30.0
                
                ssig = ssig / np.max(ssig) * scale_ratio_val
            else:
                continue
            
            # Set weight decay inside patch
            if self.dataset_meta["mag_change"].ndim == 2:
                weight_decay = self.dataset_meta["mag_change"][index]
            else:
                weight_decay = self.dataset_meta["mag_change"][index][kk]
            
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay
            
            raw_nmm = raw_nmm + current_nmm
        
        # Project to sensor space
        eeg = np.matmul(self.fwd, raw_nmm.transpose()).transpose()
        
        # Add noise
        csnr = self.dataset_meta["current_snr"][index]
        if np.isscalar(csnr):
            csnr_val = float(csnr)
        else:
            csnr_val = float(csnr[0]) if len(csnr) > 0 else 10.0
        
        noisy_eeg = add_white_noise(eeg.transpose(), csnr_val).transpose()
        
        # Normalize EEG
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)
        
        if self.normalize:
            if np.max(np.abs(noisy_eeg)) > 0:
                noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))
        
        # Prepare source output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        
        if self.normalize:
            if np.max(empty_nmm) > 0:
                empty_nmm = empty_nmm / np.max(empty_nmm)
        
        # Convert to tensors
        eeg_tensor = torch.from_numpy(noisy_eeg.astype(np.float32))
        source_tensor = torch.from_numpy(empty_nmm.astype(np.float32))
        
        return eeg_tensor, source_tensor
    
    def _load_nmm_data(self, nmm_idx):
        """Load NMM data from region files based on index"""
        # Available files in nmm_spikes:
        # a0: nmm_1, nmm_2, nmm_3
        # a1: nmm_1 to nmm_24
        # Total: 27 files approximately
        
        available_files = []
        # a0 files
        for i in [1, 2, 3]:
            available_files.append(("a0", i))
        # a1 files (adjust based on your actual files)
        for i in range(1, 25):  # Assuming up to nmm_24
            available_files.append(("a1", i))
        
        # Cycle through available files
        file_idx = nmm_idx % len(available_files)
        a_dir, file_num = available_files[file_idx]
        
        file_path = os.path.join(self.nmm_spikes_dir, a_dir, f"nmm_{file_num}.mat")
        
        try:
            data = load_mat_file(file_path)
            
            # Extract data field
            if 'data' in data:
                nmm_data = data['data']
            else:
                for key in ['nmm_data', 'source_data', 'signals']:
                    if key in data:
                        nmm_data = data[key]
                        break
                else:
                    return None
            
            # Ensure correct shape
            if nmm_data.shape[1] > 994:
                nmm_data = nmm_data[:, :994]
            elif nmm_data.shape[1] < 994:
                padded = np.zeros((nmm_data.shape[0], 994))
                padded[:, :nmm_data.shape[1]] = nmm_data
                nmm_data = padded
            
            # Resample to 500 time points
            if nmm_data.shape[0] != 500:
                nmm_data = self._resample_data(nmm_data, 500)
            
            return nmm_data
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return np.zeros((500, 994))
    
    def _resample_data(self, data, target_length):
        """Resample data to target length"""
        original_time = np.linspace(0, 1, data.shape[0])
        new_time = np.linspace(0, 1, target_length)
        resampled = np.zeros((target_length, data.shape[1]))
        
        for i in range(data.shape[1]):
            f = interpolate.interp1d(original_time, data[:, i], kind='linear')
            resampled[:, i] = f(new_time)
        
        return resampled


class SpikeEEGDataset(Dataset):
    """
    Dataset for loading EEG source data from nmm_spikes directory structure
    
    Directory structure:
        source/nmm_spikes/a0/*.mat
        source/nmm_spikes/a1/*.mat
        ...
        source/nmm_spikes/a993/*.mat
    
    Each mat file contains 'data' with shape (500, 994)
    
    Parameters
    ----------
    data_root : str
        Root directory for data (should contain 'source/nmm_spikes' or point to 'nmm_spikes')
    fwd : np.array
        Forward matrix: (num_electrodes, num_regions), e.g., (75, 994)
    num_sources : int
        Number of active sources per sample (default: 2)
    patch_size : int
        Size of each source patch (default: 20)
    dataset_len : int
        Total number of samples to generate (default: 1000)
    snr_range : tuple
        Range of SNR values in dB (default: (0, 30))
    normalize : bool
        Whether to normalize the output (default: True)
    """
    
    def __init__(
        self,
        data_root,
        fwd,
        num_sources=2,
        patch_size=20,
        dataset_len=1000,
        snr_range=(0, 30),
        normalize=True,
    ):
        self.data_root = data_root
        self.fwd = fwd
        self.num_sources = num_sources
        self.patch_size = patch_size
        self.dataset_len = dataset_len
        self.snr_range = snr_range
        self.normalize = normalize
        
        # Find nmm_spikes directory
        if os.path.exists(os.path.join(data_root, "source", "nmm_spikes")):
            self.nmm_spikes_dir = os.path.join(data_root, "source", "nmm_spikes")
        elif os.path.exists(os.path.join(data_root, "nmm_spikes")):
            self.nmm_spikes_dir = os.path.join(data_root, "nmm_spikes")
        else:
            raise ValueError(f"Could not find nmm_spikes directory in {data_root}")
        
        # Index all available mat files in each region folder
        self.region_files = {}  # region_idx -> list of file paths
        self.num_regions = 994
        
        print(f"Indexing mat files in {self.nmm_spikes_dir}...")
        for region_idx in range(self.num_regions):
            region_dir = os.path.join(self.nmm_spikes_dir, f"a{region_idx}")
            if os.path.exists(region_dir):
                mat_files = sorted(glob.glob(os.path.join(region_dir, "*.mat")))
                if len(mat_files) > 0:
                    self.region_files[region_idx] = mat_files
        
        if len(self.region_files) == 0:
            raise ValueError(f"No mat files found in {self.nmm_spikes_dir}")
        
        print(f"Found {len(self.region_files)} regions with data")
        print(f"Total files indexed: {sum(len(files) for files in self.region_files.values())}")
        
        # Available regions for sampling
        self.available_regions = list(self.region_files.keys())
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        """
        Generate one sample by:
        1. Randomly selecting source regions
        2. Creating source patches around each region
        3. Loading NMM data for each patch
        4. Projecting to sensor space
        5. Adding noise
        
        Returns
        -------
        tuple
            (eeg_data, source_data) both as torch tensors
            eeg_data: (500, 75) - EEG sensor data
            source_data: (500, 994) - Source space activity
        """
        # Set seed for reproducibility (optional)
        np.random.seed(index)
        random.seed(index)
        
        # Initialize source space data
        source_data = np.zeros((500, self.num_regions))  # (time, regions)
        
        # Select random source regions (center of patches)
        selected_regions = random.sample(self.available_regions, 
                                        min(self.num_sources, len(self.available_regions)))
        
        for source_idx, center_region in enumerate(selected_regions):
            # Create a source patch around the center region
            # For simplicity, we select nearby regions (you can make this more sophisticated)
            patch_regions = self._get_patch_regions(center_region, self.patch_size)
            
            # Load NMM data for the center region
            nmm_data = self._load_region_data(center_region)
            
            if nmm_data is None or np.max(np.abs(nmm_data)) == 0:
                continue
            
            # Extract center region activity
            center_activity = nmm_data[:, center_region:center_region+1]  # (500, 1)
            
            # Normalize and scale
            if np.max(np.abs(center_activity)) > 0:
                # Scale to random amplitude
                scale = np.random.uniform(10, 50)
                center_activity = center_activity / np.max(np.abs(center_activity)) * scale
            
            # Assign activity to patch with decay
            for i, region_idx in enumerate(patch_regions):
                if region_idx >= self.num_regions:
                    continue
                # Simple distance-based decay
                decay = 1.0 / (1.0 + i * 0.1)  # Center has decay=1, others decay
                source_data[:, region_idx] += center_activity.flatten() * decay
        
        # Project to sensor space using forward matrix
        # fwd: (75, 994), source_data: (500, 994)
        eeg_data = np.matmul(self.fwd, source_data.T).T  # (75, 500).T = (500, 75)
        
        # Add white Gaussian noise
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        eeg_data = self._add_white_noise(eeg_data, snr_db)
        
        # Normalize EEG data
        eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)  # channel-wise
        eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)  # time-wise
        
        if self.normalize:
            if np.max(np.abs(eeg_data)) > 0:
                eeg_data = eeg_data / np.max(np.abs(eeg_data))
            
            if np.max(np.abs(source_data)) > 0:
                source_data = source_data / np.max(np.abs(source_data))
        
        # Convert to float32 and torch tensors
        eeg_tensor = torch.from_numpy(eeg_data.astype(np.float32))
        source_tensor = torch.from_numpy(source_data.astype(np.float32))
        
        return eeg_tensor, source_tensor
    
    def _get_patch_regions(self, center_region, patch_size):
        """Get nearby regions for a patch (simple adjacent regions)"""
        patch = [center_region]
        
        # Add nearby regions (simple linear neighborhood)
        for offset in range(1, patch_size):
            if center_region - offset >= 0 and center_region - offset in self.available_regions:
                patch.append(center_region - offset)
            if center_region + offset < self.num_regions and center_region + offset in self.available_regions:
                patch.append(center_region + offset)
            
            if len(patch) >= patch_size:
                break
        
        return patch[:patch_size]
    
    def _load_region_data(self, region_idx):
        """Load data from a random file in the region's folder"""
        if region_idx not in self.region_files:
            return None
        
        # Randomly select a file from this region
        file_path = random.choice(self.region_files[region_idx])
        
        try:
            data = load_mat_file(file_path)
            
            # Extract 'data' field
            if 'data' in data:
                nmm_data = data['data']
            else:
                # Try other common field names
                for key in ['nmm_data', 'source_data', 'signals']:
                    if key in data:
                        nmm_data = data[key]
                        break
                else:
                    print(f"Warning: Could not find data field in {file_path}")
                    return None
            
            # Ensure correct shape (500, 994)
            if nmm_data.shape[1] > 994:
                nmm_data = nmm_data[:, :994]
            elif nmm_data.shape[1] < 994:
                # Pad with zeros
                padded = np.zeros((nmm_data.shape[0], 994))
                padded[:, :nmm_data.shape[1]] = nmm_data
                nmm_data = padded
            
            # Resample to 500 time points if needed
            if nmm_data.shape[0] != 500:
                nmm_data = self._resample_data(nmm_data, 500)
            
            return nmm_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _resample_data(self, data, target_length):
        """Resample data to target length"""
        original_time = np.linspace(0, 1, data.shape[0])
        new_time = np.linspace(0, 1, target_length)
        resampled = np.zeros((target_length, data.shape[1]))
        
        for i in range(data.shape[1]):
            f = interpolate.interp1d(original_time, data[:, i], kind='linear')
            resampled[:, i] = f(new_time)
        
        return resampled
    
    def _add_white_noise(self, signal, snr_db):
        """Add white Gaussian noise based on SNR in dB"""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise


def create_dataloaders_from_metadata(
    train_metadata_path,
    test_metadata_path,
    fwd_matrix_path,
    batch_size=8,
    val_split=0.1,
    num_workers=4,
    nmm_spikes_dir=None,
    train_dataset_len=None,
    test_dataset_len=None,
):
    """
    Create dataloaders from metadata files (train_sample_source1.mat, test_sample_source1.mat)
    
    Parameters
    ----------
    train_metadata_path : str
        Path to training metadata file
    test_metadata_path : str
        Path to test metadata file
    fwd_matrix_path : str
        Path to forward matrix .mat file
    batch_size : int
        Batch size
    val_split : float
        Fraction of training data for validation
    num_workers : int
        Number of data loading workers
    nmm_spikes_dir : str, optional
        Path to nmm_spikes directory
    train_dataset_len : int, optional
        Number of training samples to use
    test_dataset_len : int, optional
        Number of test samples to use
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, None)
    """
    # Load forward matrix
    print(f"Loading forward matrix from {fwd_matrix_path}...")
    fwd_data = load_mat_file(fwd_matrix_path)
    
    if 'fwd' in fwd_data:
        fwd = fwd_data['fwd']
    elif 'forward' in fwd_data:
        fwd = fwd_data['forward']
    elif 'leadfield' in fwd_data:
        fwd = fwd_data['leadfield']
    else:
        # Try to find any matrix with the right shape
        for key, value in fwd_data.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                if value.shape[0] == 75 or value.shape[1] == 75:
                    fwd = value
                    if fwd.shape[1] == 75:  # Transpose if needed
                        fwd = fwd.T
                    break
        else:
            raise ValueError("Could not find forward matrix in file")
    
    print(f"Forward matrix shape: {fwd.shape}")
    
    # Create training dataset
    print(f"\nLoading training data from {train_metadata_path}...")
    train_full_dataset = SpikeEEGMetadataDataset(
        metadata_path=train_metadata_path,
        fwd=fwd,
        nmm_spikes_dir=nmm_spikes_dir,
        dataset_len=train_dataset_len,
        normalize=True,
    )
    
    # Split training into train and validation
    train_size = int((1 - val_split) * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test dataset
    print(f"\nLoading test data from {test_metadata_path}...")
    test_dataset = SpikeEEGMetadataDataset(
        metadata_path=test_metadata_path,
        fwd=fwd,
        nmm_spikes_dir=nmm_spikes_dir,
        dataset_len=test_dataset_len,
        normalize=True,
    )
    
    print(f"\nDataset splits - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
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
    
    return train_loader, val_loader, test_loader, None


def create_dataloaders_from_spikes(
    data_root,
    fwd_matrix_path,
    batch_size=8,
    train_split=0.8,
    val_split=0.1,
    num_workers=4,
    dataset_len=1000,
    num_sources=2,
    patch_size=20,
    snr_range=(0, 30),
):
    """
    Create dataloaders from nmm_spikes data
    
    Parameters
    ----------
    data_root : str
        Root directory containing 'source/nmm_spikes' or pointing to 'nmm_spikes'
    fwd_matrix_path : str
        Path to forward matrix .mat file (should contain 'fwd' field with shape (75, 994))
    batch_size : int
        Batch size
    train_split : float
        Fraction for training
    val_split : float
        Fraction for validation
    num_workers : int
        Number of data loading workers
    dataset_len : int
        Total dataset size
    num_sources : int
        Number of sources per sample
    patch_size : int
        Size of each source patch
    snr_range : tuple
        SNR range in dB
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # Load forward matrix
    print(f"Loading forward matrix from {fwd_matrix_path}...")
    fwd_data = load_mat_file(fwd_matrix_path)
    
    if 'fwd' in fwd_data:
        fwd = fwd_data['fwd']
    elif 'forward' in fwd_data:
        fwd = fwd_data['forward']
    elif 'leadfield' in fwd_data:
        fwd = fwd_data['leadfield']
    else:
        # Try to find any matrix with the right shape
        for key, value in fwd_data.items():
            if isinstance(value, np.ndarray) and value.shape == (75, 994):
                fwd = value
                break
        else:
            raise ValueError("Could not find forward matrix in file")
    
    print(f"Forward matrix shape: {fwd.shape}")
    
    # Create full dataset
    full_dataset = SpikeEEGDataset(
        data_root=data_root,
        fwd=fwd,
        num_sources=num_sources,
        patch_size=patch_size,
        dataset_len=dataset_len,
        snr_range=snr_range,
        normalize=True,
    )
    
    # Split into train/val/test
    train_size = int(train_split * dataset_len)
    val_size = int(val_split * dataset_len)
    test_size = dataset_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
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
    
    return train_loader, val_loader, test_loader, None

