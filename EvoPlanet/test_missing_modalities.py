import torch
import numpy as np
import random
from src.pipeline import EvoPlanetPipeline
from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector

def simulate_missing_metadata(base_metadata, missing_indices):
    """
    Takes an 8-dim metadata vector, zeroes out missing indices, 
    and returns a 16-dim vector (8 imputed values + 8 presence flags).
    """
    metadata_16 = np.zeros(16, dtype=np.float32)
    for i in range(8):
        if i in missing_indices:
            metadata_16[i] = 0.0 # Impute missing with 0
            metadata_16[i+8] = 0.0 # Flag as missing
        else:
            metadata_16[i] = base_metadata[i]
            metadata_16[i+8] = 1.0 # Flag as present
    return metadata_16

def run_modality_tests():
    print("Initializing EvoPlanet Pipeline for Modality Dropout Tests...")
    
    # We initialize fresh models since we changed the architecture to 16-dim.
    # Note: For production, we would need to retrain.
    pipeline = EvoPlanetPipeline(seq_len=200)
    
    # Simulate a typical candidate sequence (5 channels, 200 time steps)
    mock_sequence = np.random.randn(5, 200)
    # Add a mock transit dip to channel 0
    mock_sequence[0, 90:110] -= 1.5 
    
    # Base 8-dim metadata (Radius, Mass, Teff, log g, Period, Duration, Depth, Power)
    base_metadata = np.array([1.0, 1.0, 5800.0, 4.4, 10.0, 3.0, 0.01, 15.0])
    
    test_cases = [
        {"name": "Ideal Case (No Missing Data)", "missing": []},
        {"name": "Missing Temperature (Teff)", "missing": [2]},
        {"name": "Missing Stellar Properties (Rad, Mass, Teff, logg)", "missing": [0, 1, 2, 3]},
        {"name": "Missing BLS Features (Per, Dur, Dep, Pow)", "missing": [4, 5, 6, 7]},
        {"name": "Catastrophic Failure (All Metadata Missing)", "missing": list(range(8))}
    ]
    
    print("\n--- Running Rigorous Missing Data Scenarios ---")
    
    for case in test_cases:
        print(f"\nScenario: {case['name']}")
        
        # 1. Apply Modality Masking to generate 16-dim input
        meta_16 = simulate_missing_metadata(base_metadata, case['missing'])
        
        # 2. Run inference
        try:
            res = pipeline.process_sequence(mock_sequence, meta_16)
            print(f"Success! Pipeline ingested the partial data seamlessly.")
            print(f"   Candidate Probability: {res['candidate_probability']:.4f}")
            print(f"   False Positive Prob: {res['false_positive_probability']:.4f}")
            print(f"   Epistemic Uncertainty: {res['uncertainty']:.6f}")
            print(f"   EvoRank Score: {res['evolutionary_rank_score']:.4f}")
            print(f"   Physics Consistency: {res['physics_consistency']:.4f}")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    run_modality_tests()
