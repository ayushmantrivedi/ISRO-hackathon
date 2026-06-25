import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import EvoPlanetPipeline
from src.data_ingestion import CURATED_TARGETS, download_multi_channel_data
from src.preprocessing import preprocess_multichannel_data, extract_multichannel_sequences

st.set_page_config(page_title="EvoPlanet v2", layout="wide", page_icon="🪐")

@st.cache_resource
def load_pipeline():
    pipeline = EvoPlanetPipeline(seq_len=200)
    # Load weights if available
    ae_path = "weights/autoencoder.pt"
    det_path = "weights/detector.pt"
    if os.path.exists(ae_path) and os.path.exists(det_path):
        pipeline.autoencoder.load_state_dict(torch.load(ae_path, map_location='cpu'))
        pipeline.detector.load_state_dict(torch.load(det_path, map_location='cpu'))
    return pipeline

def main():
    st.title("🪐 EvoPlanet v2: Multi-Modal Architecture")
    st.markdown("""
    This dashboard demonstrates the fully upgraded EvoPlanet pipeline. It fuses **5 parallel time-series channels** 
    with an **8-dimensional physics context vector** (combining static stellar metadata and dynamically derived BLS periodograms).
    
    **New Production Features:**
    - **Multi-Channel Ingestion**: Evaluates Flux, Centroids (X/Y), Background, and Quality Flags simultaneously.
    - **Derived Physics Injection**: Dynamically calculates Box Least Squares (BLS) metrics directly from the LightCurve.
    - **Global Context MLP**: Cross-attends the sequence with the star's Radius, Mass, Teff, and log g.
    """)
    
    pipeline = load_pipeline()
    
    st.sidebar.header("Data Source (Real World)")
    target_name = st.sidebar.selectbox("Select Curated Kepler Target:", list(CURATED_TARGETS.keys()))
    
    if st.sidebar.button("Fetch API & Process Target"):
        with st.spinner(f"Fetching Multi-Channel Data for {target_name} from MAST..."):
            mc_data = download_multi_channel_data(target_name, quarter=3, download_dir="data/raw")
            
        if mc_data is None:
            st.error(f"Failed to fetch data for {target_name}.")
            return
            
        # Extract features
        metadata = mc_data['metadata'] # 8 params
        stacked_channels = preprocess_multichannel_data(mc_data)
        sequences = extract_multichannel_sequences(stacked_channels, window_size=200)
        
        if len(sequences) == 0:
            st.error("Not enough data to extract a sequence.")
            return
            
        # Grab the first sequence for demonstration
        seq_idx = 0
        multi_channel_array = sequences[seq_idx]
        
        st.write("---")
        st.write("### 1. Multi-Modal Inputs (Tier 1 & Tier 2/3)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Multi-Channel Sequence Matrix [5, 200]**")
            fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
            
            axs[0].plot(multi_channel_array[0], color='black', alpha=0.8)
            axs[0].set_ylabel('Flux')
            
            axs[1].plot(multi_channel_array[1], color='red', alpha=0.6)
            axs[1].set_ylabel('Centroid X')
            
            axs[2].plot(multi_channel_array[2], color='blue', alpha=0.6)
            axs[2].set_ylabel('Centroid Y')
            
            axs[3].plot(multi_channel_array[3], color='purple', alpha=0.6)
            axs[3].set_ylabel('Background')
            
            axs[4].plot(multi_channel_array[4], color='orange', alpha=0.6)
            axs[4].set_ylabel('Quality Flags')
            axs[4].set_xlabel('Time Steps (Sliding Window)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Global Physics Context [8-Dim]**")
            # Create a 2-column layout for the metrics to look cleaner
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Radius (R_sun)", f"{metadata[0]:.2f}")
                st.metric("Mass (M_sun)", f"{metadata[1]:.2f}")
                st.metric("Teff (K)", f"{metadata[2]:.0f}")
                st.metric("log g", f"{metadata[3]:.2f}")
            with mc2:
                st.metric("BLS Period (d)", f"{metadata[4]:.2f}")
                st.metric("BLS Duration", f"{metadata[5]:.3f}")
                st.metric("BLS Depth", f"{metadata[6]:.4f}")
                st.metric("BLS Power (SNR)", f"{metadata[7]:.1f}")
                
        with st.spinner("Running Transformer & Fusion Inference..."):
            result = pipeline.process_sequence(multi_channel_array, metadata)
            
        st.write("---")
        st.write("### 2. Inference Results & Triage Ranking")
        
        true_label = "Confirmed Planet" if CURATED_TARGETS[target_name] == 1 else "Known False Positive"
        if target_name in ["KIC 12557548", "Kepler-1625", "Kepler-452"]:
            true_label = "Ambiguous/Unconfirmed (Stress Test)"
            
        st.info(f"**Astronomical Ground Truth:** {true_label}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidate Probability", f"{result['candidate_probability']*100:.2f}%")
        c2.metric("Epistemic Uncertainty", f"{result['uncertainty']:.4f}")
        c3.metric("EvoRank Score", f"{result['evolutionary_rank_score']:.2f}")
        c4.metric("Physics Consistency", f"{result.get('physics_consistency', 1.0)*100:.1f}%")
        
        st.write("---")
        st.write("### 3. Astronomical Explainability (Saliency Map)")
        st.markdown("The attention rollout highlights exactly which temporal features the Cross-Attention mechanism focused on when fusing the Physics Metadata with the Raw Light Curve.")
        
        saliency = result['saliency_map']
        # Normalize saliency for visualization
        saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
        
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(multi_channel_array[0], color='gray', alpha=0.5, label='Normalized Flux')
        scatter = ax2.scatter(np.arange(200), multi_channel_array[0], c=saliency, cmap='hot', s=60, zorder=5, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax2, label='Attention Activation Weight')
        ax2.set_title("Transformer Cross-Attention Overlay")
        ax2.legend()
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
