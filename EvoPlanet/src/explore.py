import matplotlib.pyplot as plt
import os
from data_ingestion import download_kepler_data
from preprocessing import preprocess_lightcurve

def run_exploration():
    print("Downloading Kepler-10 data (Quarter 3)...")
    lcs = download_kepler_data("Kepler-10", quarter=3, download_dir="../data/raw")
    
    if lcs is None or len(lcs) == 0:
        print("Failed to download data.")
        return
        
    lc = lcs[0]
    print("Data downloaded. Preprocessing...")
    
    lc_flat = preprocess_lightcurve(lc)
    
    print("Preprocessing done. Generating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot raw normalized
    lc.normalize().plot(ax=axes[0], title="Raw Normalized Light Curve (Kepler-10 Quarter 3)", c='blue', alpha=0.5)
    
    # Plot flattened
    lc_flat.plot(ax=axes[1], title="Flattened Light Curve (Long-term trends removed)", c='red', alpha=0.5)
    
    plt.tight_layout()
    
    os.makedirs("../notebooks", exist_ok=True)
    out_path = "../notebooks/kepler10_exploration.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    
    # Fold it on known period of Kepler-10b (0.837 days) just to see the transit
    print("Folding on known period (0.837 days)...")
    folded = lc_flat.fold(period=0.837495)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    folded.plot(ax=ax, title="Phase-Folded Light Curve (Kepler-10b Transit)", c='black', alpha=0.3, marker='.', linestyle='none')
    out_path_folded = "../notebooks/kepler10_folded.png"
    plt.savefig(out_path_folded)
    print(f"Folded plot saved to {out_path_folded}")

if __name__ == "__main__":
    run_exploration()
