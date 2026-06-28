import pytest
import numpy as np
import warnings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_multichannel_data, extract_multichannel_sequences

def test_preprocess_pure_nans():
    # Construct a payload entirely of NaNs
    mc_data = {
        'time': np.array([1, 2, 3, 4, 5]),
        'flux': np.full(5, np.nan),
        'centroid_x': np.full(5, np.nan),
        'centroid_y': np.full(5, np.nan),
        'background': np.full(5, np.nan),
        'quality': np.full(5, np.nan)
    }
    
    # Should not throw any warnings and return an array of 0s
    stacked = preprocess_multichannel_data(mc_data, flatten_window_length=5)
    
    assert stacked.shape == (5, 5)
    assert not np.isnan(stacked).any()
    assert np.allclose(stacked, 0.0)

def test_preprocess_zero_variance():
    # Construct a payload of pure identical values (zero variance)
    mc_data = {
        'time': np.array([1, 2, 3, 4, 5]),
        'flux': np.ones(5),
        'centroid_x': np.ones(5),
        'centroid_y': np.ones(5),
        'background': np.ones(5),
        'quality': np.zeros(5)
    }
    
    stacked = preprocess_multichannel_data(mc_data, flatten_window_length=5)
    
    assert stacked.shape == (5, 5)
    assert not np.isnan(stacked).any()
    # It should standardize nicely without dividing by zero, resulting in approx 0.0 due to mean subtraction
    assert np.allclose(stacked, 0.0)
    
def test_extract_multichannel_sequences():
    # Create 5 channels of 250 length
    stacked = np.random.randn(5, 250)
    
    sequences = extract_multichannel_sequences(stacked, window_size=200, step_size=25)
    assert sequences.shape == (3, 5, 200) # (250-200)/25 + 1 = 3
    
    sequences_short = extract_multichannel_sequences(stacked, window_size=300)
    assert len(sequences_short) == 0
