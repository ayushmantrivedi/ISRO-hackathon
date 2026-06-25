import numpy as np

class EvolutionaryRanker:
    """
    Ranks exoplanet candidates using a parameterized scoring function.
    The parameters (weights) of this function are intended to be optimized
    using an Evolutionary Algorithm (e.g., from the `evonet` package).
    """
    def __init__(self, w_conf=1.0, w_unc=1.0, w_snr=1.0, w_fp=1.0):
        # Weights optimized via Evolutionary Algorithm
        self.w_conf = w_conf
        self.w_unc = w_unc
        self.w_snr = w_snr
        self.w_fp = w_fp

    def get_weights(self):
        return [self.w_conf, self.w_unc, self.w_snr, self.w_fp]

    def set_weights(self, weights):
        self.w_conf, self.w_unc, self.w_snr, self.w_fp = weights

    def score_candidate(self, confidence, uncertainty, snr, fp_score):
        """
        Calculates the evolutionary fitness/rank score of a candidate.
        
        Args:
            confidence (float): Probability from the detector (0 to 1).
            uncertainty (float): MC Dropout variance/uncertainty score (lower is better).
            snr (float): Signal-to-noise ratio of the dip.
            fp_score (float): False positive probability.
            
        Returns:
            float: The final rank score.
        """
        # Rank = w1*Confidence + w2*(1-Uncertainty) + w3*SNR - w4*FP_Score
        # Note: We penalize high uncertainty and high false positive score
        score = (self.w_conf * confidence) + \
                (self.w_unc * (1.0 - uncertainty)) + \
                (self.w_snr * snr) - \
                (self.w_fp * fp_score)
        return score

    def rank_batch(self, candidates_metrics):
        """
        Ranks a batch of candidates.
        
        Args:
            candidates_metrics (list of dicts): List containing 'conf', 'unc', 'snr', 'fp'
            
        Returns:
            list of tuples: (index, score) sorted by highest score first.
        """
        scores = []
        for i, metrics in enumerate(candidates_metrics):
            s = self.score_candidate(
                metrics.get('conf', 0.0),
                metrics.get('unc', 1.0),
                metrics.get('snr', 0.0),
                metrics.get('fp', 1.0)
            )
            scores.append((i, s))
            
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# To connect to evonet, you would define an Environment in evonet that:
# 1. Spawns networks (or genomes) representing these 4 weights.
# 2. The fitness function evaluates how many True Positives are in the Top-K ranks.
