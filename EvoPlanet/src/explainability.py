import torch
import numpy as np

def calculate_mc_dropout_variance(model, raw_seq, ae_features, metadata=None, num_passes=10):
    """
    Quantifies epistemic uncertainty using Monte Carlo Dropout.
    
    Args:
        model: The trained TransformerDetector (must have Dropout layers).
        raw_seq: (batch, 5, seq_len)
        ae_features: dict of latent features
        metadata: (batch, 8) tensor of context features
        num_passes: Number of stochastic forward passes
        
    Returns:
        float: The variance of the predictions (uncertainty score).
    """
    was_training = model.training
    model.eval() # Ensure BatchNorm stays in eval mode for single-batch inference
    
    # Enable ONLY Dropout layers for MC-Dropout
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    model.apply(apply_dropout)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            # Forward pass
            logits = model(raw_seq, ae_features, metadata=metadata)
            probs = torch.softmax(logits, dim=1)[:, 0].item() # Candidate prob
            predictions.append(probs)
            
    if not was_training:
        model.eval() # Restore eval mode
            
    variance = np.var(predictions)
    return variance

def generate_attention_rollout(model, raw_seq, ae_features, metadata=None):
    """
    Generates a saliency/attention map to explain WHICH part of the light curve
    caused the model to flag a transit.
    
    Extracts the attention weights from the Cross-Attention Fusion layer.
    """
    model.eval()
    with torch.no_grad():
        logits, attns = model(raw_seq, ae_features, metadata=metadata, return_attention=True)
        
    # By default, PyTorch MultiheadAttention averages across heads.
    # attns['l1'] has shape (batch, target_seq_len, source_seq_len)
    # Average across the latent dimension to get a 1D attention map for the raw sequence
    
    attn_l1 = attns['l1'].mean(dim=2).squeeze().cpu().numpy()
    
    return attn_l1
