import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data_ingestion import CURATED_TARGETS, download_multi_channel_data
from src.preprocessing import preprocess_multichannel_data, extract_multichannel_sequences

class RealKeplerDataset(Dataset):
    def __init__(self, seq_len=200, quarter=3, download_dir="data/raw"):
        self.seq_len = seq_len
        self.data = []
        self.metadata = []
        self.labels = []
        
        print("Initializing RealKeplerDataset. Fetching data...")
        for target_id, label in CURATED_TARGETS.items():
            mc_data = download_multi_channel_data(target_id, quarter=quarter, download_dir=download_dir)
            
            if mc_data is None:
                continue
                
            # Preprocess to get stacked channels: shape (5, total_len)
            stacked_channels = preprocess_multichannel_data(mc_data)
            
            # Extract sliding windows: shape (num_seqs, 5, seq_len)
            sequences = extract_multichannel_sequences(stacked_channels, window_size=seq_len)
            
            if len(sequences) > 0:
                self.data.append(sequences)
                meta_array = mc_data['metadata']
                self.metadata.extend([meta_array] * len(sequences))
                
                # In CURATED_TARGETS: 1 = Planet, 0 = FP.
                # Model assumes: 0 = Candidate, 1 = FP
                model_label = 0 if label == 1 else 1
                self.labels.extend([model_label] * len(sequences))
                
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
            self.data = torch.tensor(self.data, dtype=torch.float32) # (N, 5, seq_len)
            base_meta = torch.tensor(np.array(self.metadata), dtype=torch.float32) # (N, 8)
            flags = torch.ones_like(base_meta) # (N, 8) all present initially
            self.metadata = torch.cat([base_meta, flags], dim=1) # (N, 16)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            print(f"Dataset ready: {len(self)} samples with shape {self.data.shape[1:]} and meta {self.metadata.shape[1:]}")
        else:
            print("Warning: No data loaded!")
            self.data = torch.zeros((0, 5, seq_len), dtype=torch.float32)
            self.metadata = torch.zeros((0, 16), dtype=torch.float32)
            self.labels = torch.zeros(0, dtype=torch.long)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return (self.data[idx], self.metadata[idx]), self.labels[idx]

def get_real_dataloaders(batch_size=32, seq_len=200, quarter=3):
    dataset = RealKeplerDataset(seq_len=seq_len, quarter=quarter)
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        return None, None
        
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_real_dataloaders()
    if train_loader:
        for (x, meta), y in train_loader:
            print(f"Batch X shape: {x.shape}, Batch Meta shape: {meta.shape}, Batch Y shape: {y.shape}")
            break
