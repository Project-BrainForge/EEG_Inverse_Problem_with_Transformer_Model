"""
Topological converter utilities

Provides a device-aware TopologicalConverter that converts batched EEG
arrays of shape (B, SEQ, CH) into topographic images (B, SEQ, 1, H, W).

This reuses the interpolation logic from the attached converter and
exposes a `to_image_tensor` helper for PyTorch workflows.
"""
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import torch
import mne
from mne.channels import make_dig_montage
from mne.channels.layout import _find_topomap_coords


class TopologicalConverter:
    """
    Convert EEG channel vectors to topographic 2D images.

    Parameters
    ----------
    electrode_file : str
        Path to electrode configuration (default: 'anatomy/electrode_75.mat').
    image_size : int
        Output height and width (square) (default: 64).
    sphere : float|str
        Sphere radius passed to MNE topomap coords (default: 'auto').
    normalize : bool
        Whether to normalize per-sample to [0,1].
    """

    def __init__(self, electrode_file='anatomy/electrode_75.mat', image_size=64, sphere='auto', normalize=True):
        self.image_size = (image_size, image_size)
        self.sphere = sphere
        self.normalize = normalize
        self._load_electrodes(electrode_file)
        self._setup_interpolation_grid()

    def _load_electrodes(self, electrode_file):
        data = scipy.io.loadmat(electrode_file)
        # The attached .mat uses 'eloc75' structured array
        eloc = data.get('eloc75', None)
        if eloc is None:
            # try common alternative key
            eloc = data.get('eloc', data.get('eloc75', None))

        if eloc is None:
            raise ValueError(f"Electrode file {electrode_file} missing expected key 'eloc75' or 'eloc'")

        # eloc may be an object array; iterate to extract positions
        eloc_arr = eloc[0]

        self.num_channels = len(eloc_arr)
        positions = {}
        self.channel_names = []

        for electrode in eloc_arr:
            ch_name = electrode['labels'][0]
            self.channel_names.append(ch_name)
            x = float(electrode['X'][0][0])
            y = float(electrode['Y'][0][0])
            z = float(electrode['Z'][0][0])
            positions[ch_name] = np.array([x, y, z])

        self.montage = make_dig_montage(ch_pos=positions, coord_frame='head')

        self.info = mne.create_info(ch_names=self.channel_names, sfreq=1000, ch_types='eeg')
        self.info.set_montage(self.montage)

        # Get 2D positions for interpolation
        self.pos = _find_topomap_coords(self.info, picks=None, sphere=self.sphere)

    def _setup_interpolation_grid(self):
        xi = np.linspace(-1, 1, self.image_size[0])
        yi = np.linspace(-1, 1, self.image_size[1])
        self.Xi, self.Yi = np.meshgrid(xi, yi)
        self.mask = (self.Xi**2 + self.Yi**2) <= 1.0

    def convert_single_timepoint(self, eeg_data, vmin=None, vmax=None):
        assert len(eeg_data) == self.num_channels, f"Expected {self.num_channels} channels, got {len(eeg_data)}"
        values = griddata(self.pos, eeg_data, (self.Xi, self.Yi), method='cubic', fill_value=0)
        values = np.where(self.mask, values, 0)
        if self.normalize:
            if vmin is None or vmax is None:
                absmax = np.abs(eeg_data).max()
                vmin = -absmax if vmin is None else vmin
                vmax = absmax if vmax is None else vmax
            values = (values - vmin) / (vmax - vmin + 1e-10)
            values = np.clip(values, 0, 1)
        return values.astype(np.float32)

    def convert_timeseries(self, eeg_data):
        # eeg_data: (time, channels) or (channels, time)
        if eeg_data.shape[0] == self.num_channels:
            eeg_data = eeg_data.T
        num_timepoints, num_channels = eeg_data.shape
        assert num_channels == self.num_channels, f"Expected {self.num_channels} channels, got {num_channels}"
        vmin = -np.abs(eeg_data).max()
        vmax = np.abs(eeg_data).max()
        output = np.zeros((num_timepoints, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for t in range(num_timepoints):
            output[t] = self.convert_single_timepoint(eeg_data[t], vmin, vmax)
        return output

    def convert_batch(self, eeg_data):
        # eeg_data: (batch_size, num_timepoints, num_channels)
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.detach().cpu().numpy()
        batch_size, num_timepoints, num_channels = eeg_data.shape
        assert num_channels == self.num_channels, f"Expected {self.num_channels} channels, got {num_channels}"
        output = np.zeros((batch_size, num_timepoints, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for b in range(batch_size):
            output[b] = self.convert_timeseries(eeg_data[b])
        return output

    def to_image_tensor(self, eeg_batch: torch.Tensor, device='cpu', verbose=False) -> torch.Tensor:
        """
        Convert a batched EEG tensor to topographic image tensor.

        Args:
            eeg_batch: torch.Tensor of shape (B, SEQ, CH)
            device: device to place output tensor on

        Returns:
            torch.Tensor of shape (B, SEQ, 1, H, W) dtype float32
        """
        numpy_batch = eeg_batch.detach().cpu().numpy()
        images = self.convert_batch(numpy_batch)  # (B, SEQ, H, W)
        # Add channel dim
        images = images[:, :, None, :, :]
        tensor = torch.from_numpy(images).to(device)
        return tensor


def _test_quick():
    import torch
    conv = TopologicalConverter(electrode_file='anatomy/electrode_75.mat', image_size=64)
    x = torch.randn(2, 16, conv.num_channels)
    imgs = conv.to_image_tensor(x)
    print('imgs', imgs.shape, imgs.dtype)


if __name__ == '__main__':
    _test_quick()
