import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os

from src.real_data_dataset import get_real_dataloaders
from src.models.autoencoder import TransitAutoencoder
from src.models.detector import TransformerDetector
from src.pipeline import EvoPlanetPipeline

from sklearn.metrics import roc_auc_score

def evaluate_detector():
    print("Loading test data...")
    # Load real multi-channel data
    _, test_loader = get_real_dataloaders(batch_size=32, seq_len=200, quarter=3)
    if test_loader is None:
        print("Test data not found.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pipeline which contains AE, Detector, Ranker
    pipeline = EvoPlanetPipeline(seq_len=200)
    
    print("Loading trained weights...")
    try:
        pipeline.autoencoder.load_state_dict(torch.load("weights/autoencoder.pt"))
        pipeline.detector.load_state_dict(torch.load("weights/detector.pt"))
    except FileNotFoundError:
        print("Weights not found! Ensure train.py has completed.")
        return
        
    pipeline.autoencoder.to(device)
    pipeline.detector.to(device)
    
    all_labels = []
    all_probs = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for (batch_x, batch_meta), batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_meta = batch_meta.to(device)
            
            # Forward pass
            ae_features = pipeline.autoencoder.extract_features(batch_x)
            logits = pipeline.detector(batch_x, ae_features, metadata=batch_meta)
            
            probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy() # Probability of Candidate
            
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs)
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate Metrics
    # Label 0 is candidate, 1 is FP. 
    # For standard metrics, we treat candidate as the positive class (1).
    binary_labels = 1 - all_labels 
    
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(binary_labels, preds)
    f1 = f1_score(binary_labels, preds)
    
    precision, recall, _ = precision_recall_curve(binary_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    try:
        roc_auc = roc_auc_score(binary_labels, all_probs)
    except ValueError:
        roc_auc = 0.0 # Handle case with only one class in test set
        
    print("\n--- Production-Grade Evaluation Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    
    print("\nTesting EvoPlanetPipeline End-to-End on a single sample...")
    # Test end-to-end pipeline on one sample to verify Ranking and Saliency
    (sample_seq, sample_meta), sample_label = test_loader.dataset[0]
    sample_seq_np = sample_seq.numpy()
    sample_meta_np = sample_meta.numpy()
    
    # pipeline.process_sequence expects CPU numpy array
    pipeline.autoencoder.to('cpu')
    pipeline.detector.to('cpu')
    
    result = pipeline.process_sequence(sample_seq_np, sample_meta_np)
    print("\nPipeline Result for Sample 0:")
    print(f"True Label: {'Candidate' if sample_label == 0 else 'False Positive'}")
    print(f"Candidate Prob: {result['candidate_probability']:.4f}")
    print(f"Uncertainty:    {result['uncertainty']:.6f}")
    print(f"EvoRank Score:  {result['evolutionary_rank_score']:.4f}")
    print(f"Saliency Map Length: {len(result['saliency_map'])}")

if __name__ == "__main__":
    evaluate_detector()
