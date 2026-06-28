import lightkurve as lk
import numpy as np
import pandas as pd
import os
import sys

def safe_print(msg):
    try:
        print(msg)
    except Exception:
        pass

# Curated list of known planets (1) and false positives (0)
# We use only a few to keep data lightweight.
CURATED_TARGETS = {
    "Kepler-10": 1, # Planet
    "Kepler-11": 1, # Planet
    "KIC 8462852": 0, # Boyajian's Star (FP/Weird)
    "KIC 11446443": 0, # Eclipsing Binary / FP
    "Kepler-22": 1, # Planet
    "KIC 12557548": 0, # Disintegrating planet / Ambiguous Noise
    "Kepler-1625": 1,  # Exomoon Candidate
    "Kepler-452": 1    # Borderline Earth-like Candidate
}

# Stellar Metadata for global context (Radius, Mass, Teff, log g)
# Imputed with realistic values.
TARGET_METADATA = {
    "Kepler-10": [1.06, 0.91, 5627.0, 4.35],
    "Kepler-11": [1.06, 0.96, 5680.0, 4.30],
    "KIC 8462852": [1.58, 1.43, 6750.0, 4.00],
    "KIC 11446443": [1.10, 1.00, 5800.0, 4.40],
    "Kepler-22": [0.98, 0.97, 5518.0, 4.44],
    "KIC 12557548": [0.66, 0.67, 4400.0, 4.60],
    "Kepler-1625": [1.79, 1.08, 5548.0, 3.90],
    "Kepler-452": [1.11, 1.04, 5757.0, 4.32]
}

def download_multi_channel_data(target_id, quarter=3, author="Kepler", exptime="long", download_dir="data/raw"):
    """
    Downloads Kepler light curve data and extracts multi-channel features.
    Extracts: Flux, Centroid X, Centroid Y, Background, Quality Flags.
    """
    safe_print(f"Searching for {target_id} (Quarter {quarter})...")
    search_result = lk.search_lightcurve(target_id, author=author, exptime=exptime, quarter=quarter)
    
    if len(search_result) == 0:
        safe_print(f"No data found for {target_id} in Quarter {quarter}.")
        return None
        
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    safe_print(f"Downloading to {download_dir}...")
    lc_collection = search_result.download_all(download_dir=download_dir)
    
    if lc_collection is None or len(lc_collection) == 0:
        return None
        
    lc = lc_collection[0] # Take the first downloaded LC
    
    # Extract the 5 channels
    # Note: Lightkurve LC objects have these as astropy columns. We convert to numpy arrays.
    
    # Fill missing values with median for centroids/bkg before returning
    def safe_extract(column_name):
        arr = getattr(lc, column_name).value
        # Replace NaNs/Infs with median
        if not np.isfinite(arr).all():
            median_val = np.nanmedian(arr)
            if np.isnan(median_val): median_val = 0.0
            arr = np.nan_to_num(arr, nan=median_val, posinf=median_val, neginf=median_val)
        return arr

    flux = safe_extract('flux')
    centroid_col = safe_extract('centroid_col')
    centroid_row = safe_extract('centroid_row')
    sap_bkg = safe_extract('sap_bkg')
    quality = lc.quality.value # integer flags
    
    # Fetch static metadata
    meta = list(TARGET_METADATA.get(target_id, [1.0, 1.0, 5778.0, 4.44])) # Default to Solar if missing
    
    # Compute BLS derived physics
    safe_print("Computing BLS Periodogram...")
    try:
        clean_lc = lc.remove_nans().remove_outliers()
        bls = clean_lc.to_periodogram(method='bls')
        period = float(bls.period_at_max_power.value)
        duration = float(bls.duration_at_max_power.value)
        depth = float(bls.depth_at_max_power.value)
        power = float(bls.max_power.value)
        bls_features = [period, duration, depth, power]
        bls_features = [0.0 if not np.isfinite(f) else f for f in bls_features]
    except Exception as e:
        safe_print(f"BLS failed: {e}. Using fallback values.")
        bls_features = [1.0, 0.1, 0.001, 10.0]
        
    # Append dynamic features to static metadata
    meta.extend(bls_features) # Now 8 dimensions!
    
    # Create a dictionary to hold the multi-channel data
    multi_channel_data = {
        'time': lc.time.value,
        'flux': flux,
        'centroid_x': centroid_col,
        'centroid_y': centroid_row,
        'background': sap_bkg,
        'quality': quality,
        'metadata': np.array(meta, dtype=np.float32)
    }
    
    return multi_channel_data

if __name__ == "__main__":
    print("Testing Multi-Channel Data Ingestion with Metadata...")
    data = download_multi_channel_data("Kepler-10", quarter=3, download_dir="../data/raw")
    if data is not None:
        print("Successfully extracted channels!")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f" - {k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f" - {k}: {v}")
