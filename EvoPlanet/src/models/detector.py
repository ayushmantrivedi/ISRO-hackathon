import torch
import torch.nn as nn
from .physics_inspector import EvoPhysicsInspector

class CrossAttentionFusion(nn.Module):
    """
    Fuses raw sequence representations with latent multi-level features
    using Cross Attention.
    """
    def __init__(self, embed_dim, num_heads=4, fusion_type='gated'):
        super().__init__()
        self.fusion_type = fusion_type
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        if self.fusion_type == 'gated':
            # Gating mechanism to learn *when* to trust the autoencoder features
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        
    def forward(self, query, key_value):
        """
        query: Features from the raw sequence (batch, seq_len, embed_dim)
        key_value: Features from the autoencoder (batch, enc_len, embed_dim)
        """
        # Cross attention: query attends to key_value
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value)
        
        if self.fusion_type == 'gated':
            # Calculate gate based on both query and attention output
            gate_input = torch.cat([query, attn_out], dim=-1)
            g = self.gate(gate_input)
            out = self.layer_norm(query + g * attn_out)
        elif self.fusion_type == 'cross_attention':
            # Simple residual connection
            out = self.layer_norm(query + attn_out)
        else: # fallback or 'concat' (handled outside if needed, but we'll default to cross_attn logic)
            out = self.layer_norm(query + attn_out)
            
        return out, attn_weights

class TransformerDetector(nn.Module):
    """
    Dynamic Primary Detector model for EvoNet NAS.
    Processes the raw sequence using a Transformer Encoder,
    then uses dynamic Fusion to integrate latent representations from the Autoencoder.
    """
    def __init__(self, seq_len=200, embed_dim=64, num_heads=4, num_layers=2, 
                 fusion_type='gated', hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        # 1. Project raw sequence into embedding space
        self.input_proj = nn.Linear(5, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer for the raw sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projections for the Autoencoder Latent features to match embed_dim
        self.proj_l1 = nn.Linear(16, embed_dim)
        self.proj_l2 = nn.Linear(32, embed_dim)
        self.proj_bn = nn.Linear(128, embed_dim)
        
        # Fusion Layers
        self.fusion_l1 = CrossAttentionFusion(embed_dim, num_heads, fusion_type)
        self.fusion_l2 = CrossAttentionFusion(embed_dim, num_heads, fusion_type)
        self.fusion_bn = CrossAttentionFusion(embed_dim, num_heads, fusion_type)
        
        # Metadata MLP for Global Context
        self.meta_mlp = nn.Sequential(
            nn.Linear(8, 32), # 8 inputs: 4 static (Radius, Mass, Teff, logg) + 4 derived BLS (Period, Duration, Depth, Power)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Classification Head (Dynamic Topology)
        self.flatten = nn.Flatten()
        
        # We can dynamically construct the classifier based on hidden_dim
        # Input to classifier is flattened Transformer sequence + Metadata MLP output
        classifier_input_dim = (seq_len * embed_dim) + 32
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Output: [P(Candidate), P(FalsePositive)]
        )
        
        # Physics Consistency Layer
        self.physics_inspector = EvoPhysicsInspector(metadata_dim=8, bottleneck_dim=128)
        
    def forward(self, raw_seq, ae_features, metadata=None, return_attention=False):
        """
        raw_seq: (batch, 5, seq_len)
        ae_features: dict containing 'l1', 'l2', 'bottleneck'
        metadata: (batch, 8) tensor containing stellar + derived metadata
        """
        # --- 1. Process Raw Sequence ---
        x = raw_seq.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x) # (batch, seq_len, embed_dim)
        
        # --- 2. Prepare Autoencoder Features ---
        l1 = ae_features['l1'].transpose(1, 2)
        l1 = self.proj_l1(l1)
        
        l2 = ae_features['l2'].transpose(1, 2)
        l2 = self.proj_l2(l2)
        
        bn = ae_features['bottleneck'].unsqueeze(1)
        bn = self.proj_bn(bn)
        
        # --- 3. Dynamic Fusion ---
        if self.fusion_type == 'concat':
            # If concat, we ignore attention weights and just concat
            # Because sequence lengths differ, we'll pool them first.
            # (This is an example of handling different structural paths)
            x_pool = x.mean(dim=1)
            l1_pool = l1.mean(dim=1)
            l2_pool = l2.mean(dim=1)
            bn_pool = bn.mean(dim=1)
            fused = torch.cat([x_pool, l1_pool, l2_pool, bn_pool], dim=-1)
            if metadata is not None:
                meta_emb = self.meta_mlp(metadata)
                fused = torch.cat([fused, meta_emb], dim=-1)
                
            # Override classifier for concat (hacky but works for demo)
            if not hasattr(self, 'concat_classifier'):
                self.concat_classifier = nn.Linear(fused.shape[-1], 2).to(x.device)
            logits = self.concat_classifier(fused)
            
            if return_attention:
                # Fake attention since concat doesn't have it
                return logits, {"l1": torch.ones(x.size(0), x.size(1), l1.size(1)).to(x.device)}
            return logits
            
        else:
            # Cross Attention or Gated
            x, attn_l1 = self.fusion_l1(x, l1)
            x, attn_l2 = self.fusion_l2(x, l2)
            x, attn_bn = self.fusion_bn(x, bn)
            
            # --- 4. Global Context Fusion & Classification ---
            flat = self.flatten(x)
            
            if metadata is not None:
                meta_emb = self.meta_mlp(metadata)
                flat = torch.cat([flat, meta_emb], dim=-1)
            else:
                dummy_meta = torch.zeros(x.size(0), 32).to(x.device)
                flat = torch.cat([flat, dummy_meta], dim=-1)
                
            logits = self.classifier(flat)
            
            # --- 5. EvoPhysicsInspector ---
            # Apply dynamic physics consistency penalty
            bn_feat = ae_features['bottleneck']
            if metadata is not None:
                consistency_factor = self.physics_inspector(metadata, bn_feat) # (batch, 1)
            else:
                consistency_factor = torch.ones(logits.size(0), 1).to(logits.device)
                
            # If consistency is 1.0 (perfect), penalty is 0.
            # If consistency is 0.0 (impossible), penalty is 5.0.
            penalty = (1.0 - consistency_factor) * 5.0
            
            # Subtract penalty from Candidate logit (0) and add to FP logit (1)
            mod = torch.cat([-penalty, penalty], dim=-1)
            logits = logits + mod
            
            if return_attention:
                return logits, {"l1": attn_l1, "l2": attn_l2, "bn": attn_bn, "consistency_factor": consistency_factor}
                
            return logits
