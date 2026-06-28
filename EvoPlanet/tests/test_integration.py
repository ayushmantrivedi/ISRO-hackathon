import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import EvoPlanetPipeline
from src.preprocessing import preprocess_multichannel_data

def test_pipeline_integration():
    """
    End-to-End integration test simulating a fetched lightcurve,
    preprocessing it, and pushing it through the entire EvoPlanetPipeline.
    """
    # 1. Simulate raw fetched multi-channel data
    seq_len = 250
    mc_data = {
        'time': np.linspace(0, 10, seq_len),
        'flux': np.random.randn(seq_len) + 1.0,
        'centroid_x': np.random.randn(seq_len),
        'centroid_y': np.random.randn(seq_len),
        'background': np.random.randn(seq_len),
        'quality': np.zeros(seq_len),
        'metadata': np.array([1.0, 1.0, 5000, 4.4, 0.5, 0.1, 0.01, 10.0]) # 8-dim metadata
    }
    
    # 2. Preprocess
    stacked_channels = preprocess_multichannel_data(mc_data)
    assert stacked_channels.shape == (5, seq_len)
    
    # 3. Trim to exact window
    window = stacked_channels[:, :200]
    
    # 4. Initialize Pipeline
    pipeline = EvoPlanetPipeline(seq_len=200)
    
    # 5. Inference
    result = pipeline.process_sequence(window, mc_data['metadata'])
    
    # 6. Assertions on output structure and values
    assert 'candidate_probability' in result
    assert 'uncertainty' in result
    assert 'evolutionary_rank_score' in result
    assert 'saliency_map' in result
    assert 'physics_consistency' in result
    
    assert 0.0 <= result['candidate_probability'] <= 1.0
    assert result['uncertainty'] >= 0.0
    assert len(result['saliency_map']) == 200
    assert 0.0 <= result['physics_consistency'] <= 1.0
