import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_synthetic_lightcurve(seq_len=200, has_transit=False, snr=2.0):
    """
    Generates a synthetic noisy lightcurve, optionally with a transit dip.
    """
    # 1. Base stellar noise (Gaussian)
    noise_level = 0.01
    time = np.linspace(0, 1, seq_len)
    flux = np.ones(seq_len) + np.random.normal(0, noise_level, seq_len)
    
    # Add some low frequency stellar variability (sinusoid)
    variability = 0.02 * np.sin(2 * np.pi * 3 * time + np.random.uniform(0, 2*np.pi))
    flux += variability
    
    if has_transit:
        # 2. Inject Transit
        # Randomize transit parameters
        transit_duration = int(np.random.uniform(10, 30))
        transit_center = int(np.random.uniform(transit_duration, seq_len - transit_duration))
        transit_depth = noise_level * snr # Depth based on desired SNR
        
        # Create U-shape dip
        dip = np.zeros(seq_len)
        half_dur = transit_duration // 2
        
        # Simple inverted parabola for transit shape
        x = np.linspace(-1, 1, transit_duration)
        parabola = 1 - (x**2)
        
        start_idx = transit_center - half_dur
        end_idx = start_idx + transit_duration
        dip[start_idx:end_idx] = -transit_depth * parabola
        
        flux += dip
        
    return flux

class SyntheticTransitDataset(Dataset):
    """
    PyTorch Dataset for Synthetic Lightcurves.
    """
    def __init__(self, num_samples=1000, seq_len=200):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.data = []
        self.labels = []
        
        # Generate 50% positive, 50% negative
        for i in range(num_samples):
            has_transit = (i % 2 == 0)
            snr = np.random.uniform(1.5, 5.0) if has_transit else 0
            
            flux = generate_synthetic_lightcurve(seq_len, has_transit, snr)
            
            # Standardize (Zero mean, unit variance)
            flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)
            
            self.data.append(flux)
            
            # Label: 0 for Candidate (has transit), 1 for False Positive/Noise
            # Note: The problem often treats 1 as planet, but our detector outputs [P(Candidate), P(FP)]
            # We'll map label 0 to Candidate, label 1 to FP.
            self.labels.append(0 if has_transit else 1)
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(1) # (batch, 1, seq_len)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(batch_size=32, num_train=2000, num_val=500, seq_len=200):
    train_dataset = SyntheticTransitDataset(num_train, seq_len)
    val_dataset = SyntheticTransitDataset(num_val, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    dataset = SyntheticTransitDataset(num_samples=10)
    x, y = dataset[0]
    print(f"Shape: {x.shape}, Label: {y}")
