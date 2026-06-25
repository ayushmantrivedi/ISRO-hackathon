import torch
import torch.nn as nn

class EvoPhysicsInspector(nn.Module):
    """
    EvoNet-Driven Adaptive Physics Threshold Layer.
    Instead of hardcoding astrophysical constraints (e.g. max transit depth for a star),
    this layer learns the boundary conditions of reality dynamically.
    
    Inputs:
        - metadata: (batch, 8) Stellar properties + Derived BLS Periodogram physics
        - bottleneck_features: (batch, 128) Latent morphology representation from Autoencoder
        
    Outputs:
        - consistency_factor: (batch, 1) Scaling factor between 0.0 and 1.0.
          1.0 = Perfectly obeys physics.
          0.0 = Physically impossible (e.g., planet too big for star, duration > period)
    """
    def __init__(self, metadata_dim=8, bottleneck_dim=128, hidden_dim=64):
        super().__init__()
        
        # We fuse the metadata and bottleneck features
        self.fusion = nn.Sequential(
            nn.Linear(metadata_dim + bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Forces output to be a multiplier between 0 and 1
        )
        
    def forward(self, metadata, bottleneck_features):
        x = torch.cat([metadata, bottleneck_features], dim=-1)
        consistency_factor = self.fusion(x)
        return consistency_factor
