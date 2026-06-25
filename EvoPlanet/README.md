# 🪐 EvoPlanet v2: Multi-Modal Exoplanet Discovery

EvoPlanet is an advanced AI-driven pipeline for exoplanet candidate triage, utilizing a **Multi-Modal Transformer Architecture** combined with **Evolutionary Neural Architecture Search (NAS)**.

## 🚀 Key Features

*   **Multi-Channel Physics Processing (Tier 1):** Directly processes 5 parallel time-series signals from NASA Kepler (Flux, Centroids X/Y, Background, Quality) to rule out instrumental noise and eclipsing binaries natively.
*   **Global Physics Context (Tier 2 & 3):** Dynamically calculates Box Least Squares (BLS) periodograms and injects an 8-dimensional physics context vector (Radius, Mass, Teff, log g, BLS Period, Depth, Duration, SNR) into a Metadata MLP.
*   **Neural Architecture Search (NAS):** Uses an evolutionary algorithm (`EvoNet`) to dynamically find the optimal fusion strategy (Cross-Attention vs. Gated) and topology for the neural network.
*   **Dynamic Physics Inspector:** An adaptive neural layer that learns physical boundary conditions (e.g., Transit Depth vs. Stellar Radius) and penalizes physically impossible signals.
*   **Astronomical Explainability:** Employs MC-Dropout for epistemic uncertainty quantification and Attention Rollout to generate visual saliency maps for astronomers.

## 🛠️ Architecture
The architecture comprises four primary components:
1.  **Transit-Preserving Autoencoder:** Self-supervised learning to compress the multi-channel light curves into latent representations.
2.  **Transformer Detector & Metadata MLP:** Processes spatial sequential data while fusing the 8-dim stellar physics parameters using Cross-Attention.
3.  **EvoPhysicsInspector:** A constraint-learning layer that evaluates if the sequence obeys the astrophysical properties of the host star.
4.  **EvoRank Candidate Triage:** Ranks signals based on Model Confidence, Uncertainty, Signal-to-Noise, and False Positive Probability.

## 💻 Usage

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the Multi-Modal Pipeline**
```bash
python train.py
```

**3. Evaluate Production Metrics (ROC-AUC, PR-AUC)**
```bash
python evaluate.py
```

**4. Run EvoNet Architecture Search**
```bash
python src/evorank/optimizer.py
```

**5. Launch Dashboard**
```bash
streamlit run app/dashboard.py
```
