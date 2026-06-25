import pytest
import torch
import numpy as np

# Ensure src path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic_data import generate_synthetic_lightcurve
from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector, CrossAttentionFusion
from src.pipeline import EvoPlanetPipeline
from src.evorank.optimizer import EvoNetEnvironment

def test_synthetic_data_generation():
    seq_len = 200
    lc_with = generate_synthetic_lightcurve(seq_len=seq_len, has_transit=True, snr=3.0)
    lc_without = generate_synthetic_lightcurve(seq_len=seq_len, has_transit=False, snr=0.0)
    
    assert len(lc_with) == seq_len
    assert len(lc_without) == seq_len
    # Ensure there is variation
    assert np.std(lc_with) > 0
    assert np.std(lc_without) > 0

def test_autoencoder_architecture():
    ae = TransitAutoencoder(seq_len=200)
    x = torch.randn(2, 5, 200) # Batch size 2, 5 channels
    
    # Test Reconstruction
    reconstruction = ae(x)
    assert reconstruction.shape == (2, 5, 200)
    
    # Test Feature Extraction
    features = ae.extract_features(x)
    assert 'l1' in features
    assert 'l2' in features
    assert 'bottleneck' in features
    
    assert features['l1'].shape == (2, 16, 100)
    assert features['l2'].shape == (2, 32, 50)
    assert features['bottleneck'].shape == (2, 128)

def test_detector_architectures():
    x_raw = torch.randn(2, 5, 200) # 5 channels
    metadata = torch.randn(2, 8) # 8 metadata dims
    ae_features = {
        'l1': torch.randn(2, 16, 100),
        'l2': torch.randn(2, 32, 50),
        'bottleneck': torch.randn(2, 128)
    }
    
    # Test Gated Fusion
    detector_gated = TransformerDetector(seq_len=200, fusion_type='gated')
    logits_gated = detector_gated(x_raw, ae_features, metadata=metadata)
    assert logits_gated.shape == (2, 2)
    
    # Test Cross Attention Fusion
    detector_attn = TransformerDetector(seq_len=200, fusion_type='cross_attention')
    logits_attn = detector_attn(x_raw, ae_features, metadata=metadata)
    assert logits_attn.shape == (2, 2)
    
    # Test Concat Fusion
    detector_concat = TransformerDetector(seq_len=200, fusion_type='concat')
    logits_concat = detector_concat(x_raw, ae_features, metadata=metadata)
    assert logits_concat.shape == (2, 2)

def test_evonet_environment():
    env = EvoNetEnvironment(proxy_epochs=0) # No actual training needed for the test
    
    # Test interpreting genome
    fusion_type, num_heads, hidden_dim, dropout = env.interpret_genome([1.2, 0.4, 2.8, 0.45])
    
    # Expected discretizations based on logic
    assert fusion_type == 'gated' # idx 1
    assert num_heads == 2 # idx 0
    assert hidden_dim == 128 # idx 2
    assert dropout == 0.45
