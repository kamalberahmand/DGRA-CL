import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def inject_anomalies(edges, timestamps, nodes, anomaly_ratio=0.15):
    """Inject realistic anomalies into test set"""
    num_anomalies = int(len(edges) * anomaly_ratio)
    anomaly_indices = np.random.choice(len(edges), num_anomalies, replace=False)
    
    labels = np.zeros(len(edges))
    labels[anomaly_indices] = 1
    
    # Degree bursts: sudden increases in node connectivity
    degree_burst_idx = anomaly_indices[:num_anomalies//3]
    avg_degree = len(edges) / len(nodes)
    burst_multiplier = np.random.uniform(5, 10)  # 5-10x increase
    for idx in degree_burst_idx:
        target_node = edges[idx][0]
        num_new_edges = int(avg_degree * burst_multiplier)
        # Add multiple edges from this node
        for _ in range(num_new_edges):
            random_target = np.random.choice(nodes)
            if idx + 1 < len(edges):
                edges = np.insert(edges, idx + 1, [target_node, random_target], axis=0)
                timestamps = np.insert(timestamps, idx + 1, timestamps[idx])
    
    # Clique injection: densely connected subgraphs
    clique_idx = anomaly_indices[num_anomalies//3:2*num_anomalies//3]
    clique_size = np.random.randint(3, 6)  # 3-5 nodes
    for idx in clique_idx:
        clique_nodes = np.random.choice(nodes, clique_size, replace=False)
        # Create dense connections (90% edge density)
        for i in range(clique_size):
            for j in range(i+1, clique_size):
                if np.random.random() < 0.9:  # 90% density
                    edges[idx] = (clique_nodes[i], clique_nodes[j])
    
    # Temporal shifts: timing anomalies (2σ deviation)
    temporal_idx = anomaly_indices[2*num_anomalies//3:]
    mean_time = np.mean(timestamps)
    std_time = np.std(timestamps)
    for idx in temporal_idx:
        # Shift beyond 2 standard deviations
        shift = np.random.choice([-1, 1]) * np.random.uniform(2, 4) * std_time
        timestamps[idx] = timestamps[idx] + shift
    
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
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    
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
