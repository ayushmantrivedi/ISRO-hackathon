import torch
import numpy as np

from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector
from src.evorank.ranker import EvolutionaryRanker
from src.explainability import calculate_mc_dropout_variance, generate_attention_rollout

class EvoPlanetPipeline:
    """
    End-to-End Inference Pipeline for EvoPlanet.
    """
    def __init__(self, seq_len=200):
        self.seq_len = seq_len
        self.autoencoder = TransitAutoencoder(seq_len=seq_len)
        self.detector = TransformerDetector(seq_len=seq_len)
        self.ranker = EvolutionaryRanker(w_conf=1.5, w_unc=2.0, w_snr=1.0, w_fp=2.0)
        
        # In a real scenario, we would load the trained weights here
        # self.autoencoder.load_state_dict(torch.load("weights/ae.pt"))
        # self.detector.load_state_dict(torch.load("weights/det.pt"))
        
        self.autoencoder.eval()
        self.detector.eval()

    def process_sequence(self, multi_channel_array, metadata_array):
        """
        Runs the full pipeline on a single light curve sequence.
        
        Args:
            multi_channel_array (np.array): Shape (5, seq_len)
            metadata_array (np.array): Shape (8,)
            
        Returns:
            dict: Results containing score, uncertainty, rank, and saliency map
        """
        # 1. Prepare Tensors
        x = torch.tensor(multi_channel_array, dtype=torch.float32).unsqueeze(0) # (1, 5, seq_len)
        meta = torch.tensor(metadata_array, dtype=torch.float32).unsqueeze(0) # (1, 8) or (1, 16)
        
        # Convert to 16-dim (Indicator Masking) if it's 8-dim
        if meta.shape[-1] == 8:
            flags = torch.ones_like(meta)
            meta = torch.cat([meta, flags], dim=-1)
            
        # Move inputs to the same device as the model
        device = next(self.detector.parameters()).device
        x = x.to(device)
        meta = meta.to(device)
        
        # Ensure we are in eval mode (MC-Dropout can sometimes alter this state)
        self.autoencoder.eval()
        self.detector.eval()
        
        # 2. Extract Latent Representations (Self-Supervised Module)
        with torch.no_grad():
            ae_features = self.autoencoder.extract_features(x)
            
            # 3. Detect & Fuse
            logits, attns = self.detector(x, ae_features, metadata=meta, return_attention=True)
            probs = torch.softmax(logits, dim=1)[0]
            prob_candidate = probs[0].item()
            prob_fp = probs[1].item()
            
            consistency_factor = attns.get('consistency_factor', torch.tensor([[1.0]]))[0, 0].item()
            
        # 4. Quantify Uncertainty
        uncertainty_score = calculate_mc_dropout_variance(self.detector, x, ae_features, metadata=meta)
        
        # 5. Explain
        saliency_map = generate_attention_rollout(self.detector, x, ae_features, metadata=meta)
        
        # 6. Rank
        # Estimate SNR simply as variance of flux channel (index 0)
        snr = np.var(multi_channel_array[0]) 
        
        rank_score = self.ranker.score_candidate(
            confidence=prob_candidate,
            uncertainty=uncertainty_score,
            snr=snr,
            fp_score=prob_fp
        )
        
        return {
            "candidate_probability": prob_candidate,
            "false_positive_probability": prob_fp,
            "uncertainty": uncertainty_score,
            "evolutionary_rank_score": rank_score,
            "saliency_map": saliency_map,
            "physics_consistency": consistency_factor
        }
