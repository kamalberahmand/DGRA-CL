import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def inject_anomalies(edges, timestamps, nodes, anomaly_ratio: float = 0.10, random_state: int | None = None):
    """
    Inject synthetic anomalies into a dynamic edge stream without changing
    the number of edges. This follows the three anomaly types described in
    the paper (degree burst, cliques, temporal shifts) at a high level,
    while keeping labels aligned with edges.

    Parameters
    ----------
    edges : np.ndarray, shape [E, 2]
        Edge list (source, target) for the test split.
    timestamps : np.ndarray, shape [E]
        Timestamps aligned with `edges`.
    nodes : np.ndarray or list
        Array of node ids.
    anomaly_ratio : float, optional
        Fraction of edges to mark as anomalous (default 0.10).
    random_state : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    edges_aug : np.ndarray, shape [E, 2]
        Edge list with anomaly patterns injected (in-place modifications).
    timestamps_aug : np.ndarray, shape [E]
        Possibly perturbed timestamps.
    labels : np.ndarray, shape [E]
        Binary anomaly labels (1 = anomalous edge).
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    edges = np.asarray(edges).copy()
    timestamps = np.asarray(timestamps).copy()
    nodes = np.asarray(nodes)

    E = len(edges)
    if E == 0:
        return edges, timestamps, np.zeros(0, dtype=int)

    num_anomalies = max(1, int(E * anomaly_ratio))
    anomaly_indices = rng.choice(E, num_anomalies, replace=False)

    labels = np.zeros(E, dtype=int)
    labels[anomaly_indices] = 1

    # Split indices into three anomaly types
    third = num_anomalies // 3
    degree_burst_idx = anomaly_indices[:third]
    clique_idx = anomaly_indices[third:2*third]
    temporal_idx = anomaly_indices[2*third:]

    # 1) Degree bursts: create short windows where a single node fires many edges
    for idx in degree_burst_idx:
        center = int(idx)
        window = rng.integers(3, 7)  # window length
        start = max(0, center - window // 2)
        end = min(E, start + window)
        burst_node = rng.choice(nodes)
        edges[start:end, 0] = burst_node
        labels[start:end] = 1  # mark all edges in the burst as anomalous

    # 2) Clique-like patterns: overwrite small blocks of edges with dense pairs
    for idx in clique_idx:
        center = int(idx)
        clique_size = rng.integers(3, 6)  # 3-5 nodes
        clique_nodes = rng.choice(nodes, size=clique_size, replace=False)

        # generate up to (clique_size choose 2) pairs but clamp to small block
        pairs = []
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                pairs.append((clique_nodes[i], clique_nodes[j]))
        rng.shuffle(pairs)

        block_len = min(len(pairs), 5)
        start = max(0, center - block_len // 2)
        end = min(E, start + block_len)
        for k, edge_idx in enumerate(range(start, end)):
            edges[edge_idx] = pairs[k]
            labels[edge_idx] = 1

    # 3) Temporal shifts: move selected edges far into the past or future
    if len(temporal_idx) > 0:
        mean_time = float(timestamps.mean())
        std_time = float(timestamps.std()) if timestamps.std() > 0 else 1.0
        for idx in temporal_idx:
            shift = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0) * std_time
            timestamps[int(idx)] = timestamps[int(idx)] + shift
            labels[int(idx)] = 1

    return edges, timestamps, labels

def evaluate(scores, labels):
    """
    Evaluate anomaly detection performance using AUC and F1 with a
    percentile-based threshold search, as done in the paper.

    Parameters
    ----------
    scores : np.ndarray, shape [N]
        Anomaly scores (higher = more anomalous).
    labels : np.ndarray, shape [N]
        Binary ground-truth labels.

    Returns
    -------
    metrics : dict
        Dictionary with AUC, F1, Precision, Recall and chosen threshold.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    auc = roc_auc_score(labels, scores)

    # Candidate thresholds from high percentiles of scores
    percentiles = [90, 95, 99]
    thresholds = np.percentile(scores, percentiles)

    best_f1 = 0.0
    best_thresh = thresholds[0]

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (scores > best_thresh).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    return {
        "AUC": float(auc),
        "F1": float(best_f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "Threshold": float(best_thresh),
    }

def augment_sequence(seq, mask_ratio: float = 0.6, crop_ratio: float = 0.8):
    """
    Generate two augmented views of a sequence for context-aware
    contrastive learning, as described in Section 3.2.

    Parameters
    ----------
    seq : torch.Tensor, shape [L]
        1D tensor of token ids.
    mask_ratio : float
        Fraction of positions to mask.
    crop_ratio : float
        Fraction of the sequence length to keep in the cropped view.

    Returns
    -------
    masked_seq : torch.Tensor, shape [L]
        Sequence with a subset of tokens replaced by 0 (mask token).
    cropped_seq : torch.Tensor, shape [L']
        Contiguous subsequence (no padding is applied here).
    """
    import torch  # imported lazily to avoid hard dependency when only using metrics

    if not isinstance(seq, torch.Tensor):
        raise TypeError("`seq` must be a 1D torch.Tensor of token ids.")

    seq_len = seq.size(0)
    if seq_len == 0:
        return seq, seq

    # Masked view
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_indices = torch.randperm(seq_len)[:num_mask]
    masked_seq = seq.clone()
    masked_seq[mask_indices] = 0  # assume 0 is the [MASK] / padding token

    # Cropped view
    crop_len = max(1, int(seq_len * crop_ratio))
    if crop_len >= seq_len:
        cropped_seq = seq.clone()
    else:
        start = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
        cropped_seq = seq[start:start + crop_len]

    return masked_seq, cropped_seq
