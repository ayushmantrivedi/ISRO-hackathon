import numpy as np
from scipy.signal import savgol_filter

def preprocess_multichannel_data(mc_data, flatten_window_length=101, polyorder=3):
    """
    Preprocesses the 5-channel data dictionary.
    """
    time = mc_data['time']
    flux = mc_data['flux']
    centroid_x = mc_data['centroid_x']
    centroid_y = mc_data['centroid_y']
    background = mc_data['background']
    quality = mc_data['quality']

    # 1. Flux processing: Detrend using Savitzky-Golay
    # Create mask of lower 3-sigma outliers
    std_flux = np.std(flux)
    med_flux = np.median(flux)
    mask = flux < (med_flux - 3 * std_flux)
    
    # Simple flattening using Sav-Gol if window is valid
    if len(flux) > flatten_window_length:
        trend = savgol_filter(flux, flatten_window_length, polyorder)
        # Avoid division by zero
        trend[trend == 0] = 1e-8
        flux_flat = flux / trend
    else:
        flux_flat = flux

    # 2. Standardization function
    def standardize(arr):
        std = np.std(arr)
        if std == 0: std = 1e-8
        return (arr - np.mean(arr)) / std

    # Standardize channels
    flux_flat = standardize(flux_flat)
    centroid_x = standardize(centroid_x)
    centroid_y = standardize(centroid_y)
    background = standardize(background)
    
    # 3. Quality Flags
    # Standardize so it's on the same numerical scale for neural networks
    quality = np.array(quality, dtype=np.float32)
    quality = standardize(quality)

    # Stack into shape: (Channels, Seq_Len) -> (5, Seq_Len)
    stacked_channels = np.vstack([flux_flat, centroid_x, centroid_y, background, quality])
    
    # Sanitize any remaining NaNs or Infs
    stacked_channels = np.nan_to_num(stacked_channels, nan=0.0, posinf=0.0, neginf=0.0)
    
    return stacked_channels

def extract_multichannel_sequences(stacked_channels, window_size=200, step_size=100):
    """
    Extracts fixed-length sequences for deep learning.
    
    Args:
        stacked_channels (np.ndarray): Shape (5, total_length)
        
    Returns:
        np.ndarray: Shape (num_sequences, 5, window_size)
    """
    num_channels, seq_len = stacked_channels.shape
    
    if seq_len < window_size:
        return np.array([])
        
    sequences = []
    for i in range(0, seq_len - window_size + 1, step_size):
        # Shape: (5, window_size)
        seq = stacked_channels[:, i:i+window_size]
        sequences.append(seq)
        
    return np.array(sequences)
