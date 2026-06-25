import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class TransitAutoencoder(nn.Module):
    """
    Self-Supervised Astronomical Representation Learner.
    Learns to compress lightcurves and extract transit morphology.
    Returns multi-level features instead of just the bottleneck.
    """
    def __init__(self, seq_len=200):
        super().__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.enc1 = ConvBlock(5, 16, pool=True)     # seq_len / 2
        self.enc2 = ConvBlock(16, 32, pool=True)    # seq_len / 4
        self.enc3 = ConvBlock(32, 64, pool=True)    # seq_len / 8
        
        # Bottleneck
        self.bottleneck_dim = (seq_len // 8) * 64
        self.fc_enc = nn.Linear(self.bottleneck_dim, 128)
        self.fc_dec = nn.Linear(128, self.bottleneck_dim)
        
        # Decoder
        self.dec3 = UpConvBlock(64, 32)
        self.dec2 = UpConvBlock(32, 16)
        self.dec1 = UpConvBlock(16, 5)
        
    def extract_features(self, x):
        """
        Extracts multi-level latent representations.
        Args:
            x (torch.Tensor): Input sequence (batch_size, 5, seq_len)
        Returns:
            dict: Layer 1, Layer 2, and Bottleneck features
        """
        # (batch_size, 1, seq_len)
        l1 = self.enc1(x)       # (batch_size, 16, seq_len/2)
        l2 = self.enc2(l1)      # (batch_size, 32, seq_len/4)
        l3 = self.enc3(l2)      # (batch_size, 64, seq_len/8)
        
        flat = l3.view(l3.size(0), -1)
        bottleneck = self.fc_enc(flat) # (batch_size, 128)
        
        return {
            "l1": l1,
            "l2": l2,
            "bottleneck": bottleneck
        }

    def forward(self, x):
        """
        Forward pass for self-supervised training (Reconstruction).
        """
        features = self.extract_features(x)
        l1, l2, bottleneck = features["l1"], features["l2"], features["bottleneck"]
        
        dec_input = self.fc_dec(bottleneck)
        dec_input = dec_input.view(x.size(0), 64, self.seq_len // 8)
        
        d3 = self.dec3(dec_input)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        
        return out
