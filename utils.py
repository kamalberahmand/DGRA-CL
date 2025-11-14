import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def inject_anomalies(edges, timestamps, nodes, anomaly_ratio=0.15):
    """Inject synthetic anomalies into test set"""
    num_anomalies = int(len(edges) * anomaly_ratio)
    anomaly_indices = np.random.choice(len(edges), num_anomalies, replace=False)
    
    labels = np.zeros(len(edges))
    labels[anomaly_indices] = 1
    
    # Structural anomalies: rewire edges
    structural_idx = anomaly_indices[:num_anomalies//3]
    for idx in structural_idx:
        edges[idx] = (np.random.choice(nodes), np.random.choice(nodes))
    
    # Temporal anomalies: shift timestamps
    temporal_idx = anomaly_indices[num_anomalies//3:2*num_anomalies//3]
    for idx in temporal_idx:
        timestamps[idx] *= np.random.uniform(0.5, 1.5)
    
    # Contextual anomalies: substitute with low-frequency nodes
    contextual_idx = anomaly_indices[2*num_anomalies//3:]
    rare_nodes = nodes[:len(nodes)//10]  # Bottom 10%
    for idx in contextual_idx:
        edges[idx] = (np.random.choice(rare_nodes), edges[idx][1])
    
    return edges, timestamps, labels

def evaluate(scores, labels):
    """Evaluate anomaly detection performance"""
    auc = roc_auc_score(labels, scores)
    
    # Find optimal threshold
    thresholds = np.percentile(scores, [90, 95, 99])
    best_f1 = 0
    best_thresh = thresholds[0]
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    preds = (scores > best_thresh).astype(int)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    return {
        'AUC': auc,
        'F1': best_f1,
        'Precision': precision,
        'Recall': recall
    }

def augment_sequence(seq, mask_ratio=0.6, crop_ratio=0.8):
    """Augment sequence for context-aware contrastive learning"""
    seq_len = len(seq)
    
    # Masked view
    mask_indices = np.random.choice(seq_len, int(seq_len * mask_ratio), replace=False)
    masked_seq = seq.clone()
    masked_seq[mask_indices] = 0  # Mask token
    
    # Cropped view
    crop_len = int(seq_len * crop_ratio)
    start = np.random.randint(0, seq_len - crop_len + 1)
    cropped_seq = seq[start:start + crop_len]
    
    return masked_seq, cropped_seq
