import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector
from src.real_data_dataset import get_real_dataloaders

def train_autoencoder(autoencoder, dataloader, num_epochs=5, device='cpu'):
    """
    Stage 2: Self-Supervised Training
    Trains the Transit-Preserving Autoencoder to reconstruct the sequence.
    """
    print("Starting Stage 2: Self-Supervised Representation Learning...")
    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        for (batch_x, _), _ in dataloader:  # We ignore labels and metadata for autoencoder
            x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstructed = autoencoder(x)
            
            loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} - AE Loss: {total_loss/len(dataloader):.4f}")
    return autoencoder


def train_detector(detector, autoencoder, dataloader, num_epochs=5, device='cpu'):
    """
    Stage 3: Detector & Fusion Training
    Trains the Transformer + Cross-Attention Fusion layer.
    """
    print("Starting Stage 3: Training Transformer Detector with Fusion...")
    detector.to(device)
    autoencoder.to(device)
    autoencoder.eval() # Freeze AE
    
    optimizer = optim.Adam(detector.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss() 
    
    for epoch in range(num_epochs):
        detector.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for (batch_x, batch_meta), batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Apply Modality Dropout (~20% chance to drop features dynamically per batch)
            mask = torch.rand_like(batch_meta[:, :8]) > 0.2
            batch_meta[:, :8] = batch_meta[:, :8] * mask # Zero out features
            batch_meta[:, 8:] = batch_meta[:, 8:] * mask # Zero out their flags
            batch_meta = batch_meta.to(device)
            
            optimizer.zero_grad()
            
            # Extract latent features
            with torch.no_grad():
                ae_features = autoencoder.extract_features(batch_x)
            
            # Forward pass (now includes metadata)
            logits = detector(batch_x, ae_features, metadata=batch_meta)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Detector Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")
    return detector

def main():
    # 1. Setup Data Pipeline
    print("Setting up Real Multi-Channel Data Pipeline...")
    train_loader, val_loader = get_real_dataloaders(batch_size=32, seq_len=200, quarter=3)
    
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize Models
    autoencoder = TransitAutoencoder(seq_len=200)
    detector = TransformerDetector(seq_len=200)
    
    # 3. Train Models
    print("\n--- Training Autoencoder ---")
    autoencoder = train_autoencoder(autoencoder, train_loader, num_epochs=10, device=device)
    
    print("\n--- Training Detector ---")
    detector = train_detector(detector, autoencoder, train_loader, num_epochs=15, device=device)
    
    # 4. Save Weights
    print("\nTraining Complete! Saving weights...")
    os.makedirs("weights", exist_ok=True)
    torch.save(autoencoder.state_dict(), "weights/autoencoder.pt")
    torch.save(detector.state_dict(), "weights/detector.pt")
    print("Weights saved in weights/")

if __name__ == "__main__":
    main()
