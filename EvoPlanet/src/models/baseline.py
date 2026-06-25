import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SimpleCNNBaseline(nn.Module):
    """
    A simple 1D CNN baseline classifier.
    It does not use any autoencoder representations or transformer fusion.
    Takes raw sequence directly to Candidate / False Positive.
    """
    def __init__(self, seq_len=200):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        """
        x: (batch, 1, seq_len)
        """
        features = self.conv_layers(x) # (batch, 64, 1)
        features = features.view(features.size(0), -1) # (batch, 64)
        return self.classifier(features)


class MLBaseline:
    """
    A simple Random Forest Baseline using basic extracted statistics.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def _extract_features(self, sequences):
        """
        sequences: numpy array of shape (num_samples, seq_len)
        """
        features = []
        for seq in sequences:
            if not np.isfinite(seq).all():
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            f = [
                np.mean(seq),
                np.std(seq),
                np.min(seq),
                np.max(seq),
                np.percentile(seq, 5),
                np.percentile(seq, 95)
            ]
            features.append(f)
        return np.array(features)
        
    def fit(self, X_train, y_train):
        features = self._extract_features(X_train)
        self.model.fit(features, y_train)
        
    def predict_proba(self, X_test):
        features = self._extract_features(X_test)
        return self.model.predict_proba(features)
