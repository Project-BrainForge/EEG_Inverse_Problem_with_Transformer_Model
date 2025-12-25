"""
Data utility functions for EEG source localization
"""
import numpy as np


def add_white_noise(signal, snr_db):
    """Add white Gaussian noise to signal based on SNR in dB
    
    Parameters
    ----------
    signal : np.array
        Input signal
    snr_db : float
        Signal-to-noise ratio in decibels
    
    Returns
    -------
    np.array
        Noisy signal
    """
    # Calculate signal power
    signal_power = np.mean(signal ** 2)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / snr_linear
    
    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    return noisy_signal


def ispadding(arr):
    """Check which elements are padding values (marked as 999)
    
    Parameters
    ----------
    arr : np.array
        Input array to check for padding
    
    Returns
    -------
    np.array
        Boolean array where True indicates padding
    """
    return arr >= 999

