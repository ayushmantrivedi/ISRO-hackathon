import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import sys
import os
import numpy as np

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector
from src.real_data_dataset import get_real_dataloaders

class EvoNetEnvironment:
    """
    Acts as the Environment hook for the EvoNet optimizer.
    EvoNet will spawn Genomes (which are lists/arrays of parameters).
    This environment interprets the Genome, builds the corresponding 
    neural architecture, trains it on a proxy task, and returns the fitness.
    """
    def __init__(self, device='cpu', proxy_epochs=3):
        self.device = device
        self.proxy_epochs = proxy_epochs
        
        print("Initializing EvoNet Proxy Environment...")
        # We use a smaller dataset for the proxy task to evaluate fitness quickly
        self.train_loader, self.val_loader = get_real_dataloaders(
            batch_size=32, seq_len=200, quarter=3
        )
        
        # The autoencoder doesn't need to be evolved for this, it's just a feature extractor
        # We assume it's pre-trained or we just use it initialized for proxy testing
        self.autoencoder = TransitAutoencoder(seq_len=200).to(self.device)
        self.autoencoder.eval()

    def interpret_genome(self, genome):
        """
        Maps a continuous genome vector to discrete architectural choices.
        Example Genome Layout (Length 4):
        [0]: Fusion Type (0 to 2)
        [1]: Num Heads (2, 4, 8)
        [2]: Hidden Dim Multiplier (32, 64, 128)
        [3]: Dropout Rate (0.1 to 0.5)
        """
        fusion_map = {0: 'cross_attention', 1: 'gated', 2: 'concat'}
        
        # Discretize values
        f_type_idx = int(round(max(0, min(2, genome[0]))))
        fusion_type = fusion_map[f_type_idx]
        
        heads_options = [2, 4, 8]
        head_idx = int(round(max(0, min(2, genome[1]))))
        num_heads = heads_options[head_idx]
        
        dim_options = [32, 64, 128]
        dim_idx = int(round(max(0, min(2, genome[2]))))
        hidden_dim = dim_options[dim_idx]
        
        dropout_rate = max(0.1, min(0.5, genome[3]))
        
        return fusion_type, num_heads, hidden_dim, dropout_rate

    def evaluate_fitness(self, genome):
        """
        Builds the model from the genome, trains it, and returns the validation F1-Score.
        This function should be called by the `evonet` optimizer.
        """
        fusion_type, num_heads, hidden_dim, dropout_rate = self.interpret_genome(genome)
        
        detector = TransformerDetector(
            seq_len=200,
            embed_dim=64,
            num_heads=num_heads,
            num_layers=2,
            fusion_type=fusion_type,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        optimizer = optim.Adam(detector.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Proxy Training
        detector.train()
        for epoch in range(self.proxy_epochs):
            for (batch_x, batch_meta), batch_y in self.train_loader:
                batch_x, batch_meta, batch_y = batch_x.to(self.device), batch_meta.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                with torch.no_grad():
                    ae_features = self.autoencoder.extract_features(batch_x)
                    
                logits = detector(batch_x, ae_features, metadata=batch_meta)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        # Proxy Validation
        detector.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for (batch_x, batch_meta), batch_y in self.val_loader:
                batch_x, batch_meta, batch_y = batch_x.to(self.device), batch_meta.to(self.device), batch_y.to(self.device)
                ae_features = self.autoencoder.extract_features(batch_x)
                logits = detector(batch_x, ae_features, metadata=batch_meta)
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        # Calculate F1 Score as fitness
        # Assuming label 0 is Candidate, 1 is FP
        binary_labels = 1 - np.array(all_labels)
        binary_preds = 1 - np.array(all_preds)
        
        fitness = f1_score(binary_labels, binary_preds, zero_division=0)
        
        return fitness

# Mock connection to EvoNet for demonstration
def run_mock_evonet_search():
    env = EvoNetEnvironment()
    
    # Simulate a population of random genomes
    population = [
        [1.0, 1.0, 1.0, 0.3], # Baseline Gated
        [0.0, 2.0, 2.0, 0.1], # Cross-Attn, 8 heads, 128 dim
        [2.0, 0.0, 0.0, 0.5], # Concat, 2 heads, 32 dim
    ]
    
    print("Running EvoNet Proxy Fitness Evaluations...")
    for i, genome in enumerate(population):
        fit = env.evaluate_fitness(genome)
        print(f"Genome {i} {env.interpret_genome(genome)} -> Fitness (F1): {fit:.4f}")

if __name__ == "__main__":
    run_mock_evonet_search()
