import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceEncoder(nn.Module):
    """
    Transformer-based sequence encoder for temporal interaction traces.
    This corresponds to the prefix encoder f(·) in the paper.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, seq_len]  ->  [batch, hidden_dim]
        """
        emb = self.embedding(x)                     # [B, L, H]
        out = self.transformer(emb)                 # [B, L, H]
        pooled = out.mean(dim=1)                    # simple mean pooling
        return self.output_layer(pooled)            # [B, H]


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention module to fuse retrieved normal
    exemplars with a query representation (Eq. 7).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        query : [B, H] or [H]
        keys  : [B, K, H] or [K, H]
        values: [B, K, H] or [K, H]
        """
        # Support both batched and single-sample usage
        single = False
        if query.dim() == 1:
            query = query.unsqueeze(0)   # [1, H]
            keys = keys.unsqueeze(0)     # [1, K, H]
            values = values.unsqueeze(0) # [1, K, H]
            single = True

        # [B, 1, H] @ [B, H, K] -> [B, 1, K]
        attn_scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)      # [B, 1, K]
        fused = torch.matmul(attn_weights, values).squeeze(1)  # [B, H]

        if single:
            fused = fused.squeeze(0)  # [H]
        return fused


class DGRACL(nn.Module):
    """
    Main DGRA-CL model implementing:
      - sequence encoding
      - time-aware similarity and retrieval
      - cross-attention fusion
      - deviation-based anomaly scoring
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 4, K: int = 7, n_heads: int = 8):
        super().__init__()
        self.encoder = SequenceEncoder(vocab_size, hidden_dim, num_layers, n_heads)
        self.attention_fusion = CrossAttentionFusion(hidden_dim)
        self.K = K
        self.hidden_dim = hidden_dim

    # ---------- Representation learning ----------
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, L]  ->  [B, H]"""
        return self.encoder(x)

    # ---------- Retrieval and similarity ----------
    @staticmethod
    def time_aware_similarity(
        query_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
        query_time: torch.Tensor,
        candidate_time: torch.Tensor,
        lambda_decay: float,
    ) -> torch.Tensor:
        """
        Time-decayed cosine similarity h(x_q, x_p) (Eq. 2).

        query_emb     : [Bq, H]
        candidate_emb : [Bn, H]
        query_time    : [Bq]
        candidate_time: [Bn]
        """
        # Cosine similarity
        query_norm = F.normalize(query_emb, dim=-1)
        cand_norm = F.normalize(candidate_emb, dim=-1)
        sim = torch.matmul(query_norm, cand_norm.T)  # [Bq, Bn]

        # Time decay
        time_diff = torch.abs(query_time.unsqueeze(1) - candidate_time.unsqueeze(0))  # [Bq, Bn]
        time_weight = torch.exp(-lambda_decay * time_diff)
        return sim * time_weight

    def retrieve_demonstrations(
        self,
        query_emb: torch.Tensor,
        query_time: torch.Tensor,
        pool_emb: torch.Tensor,
        pool_time: torch.Tensor,
        lambda_decay: float,
    ) -> torch.Tensor:
        """
        Retrieve top-K causally valid normal patterns from the memory.

        Causality: only candidates with time < query_time are allowed.
        """
        device = query_emb.device
        similarities = self.time_aware_similarity(
            query_emb, pool_emb, query_time, pool_time, lambda_decay
        )  # [Bq, Bn]

        # Causality mask: forbid retrieval from the future
        causal_mask = pool_time.unsqueeze(0) < query_time.unsqueeze(1)  # [Bq, Bn]
        similarities = similarities.masked_fill(~causal_mask.to(device), float("-inf"))

        # Edge case: if a row is all -inf, fall back to unconstrained similarities
        row_all_inf = torch.isinf(similarities).all(dim=1)
        if row_all_inf.any():
            similarities[row_all_inf] = self.time_aware_similarity(
                query_emb[row_all_inf], pool_emb, query_time[row_all_inf], pool_time, lambda_decay
            )

        top_k = min(self.K, pool_emb.size(0))
        top_k_indices = torch.topk(similarities, k=top_k, dim=1).indices  # [Bq, K]
        return top_k_indices

    def fuse_demonstrations(self, query_emb: torch.Tensor, retrieved_embs: torch.Tensor) -> torch.Tensor:
        """
        query_emb     : [B, H] or [H]
        retrieved_embs: [B, K, H] or [K, H]
        """
        return self.attention_fusion(query_emb, retrieved_embs, retrieved_embs)

    # ---------- Anomaly scoring ----------
    @staticmethod
    def compute_anomaly_score(
        query_emb: torch.Tensor,
        fused_baseline: torch.Tensor,
        gamma: float = 0.5,
        delta: float = 0.5,
    ) -> torch.Tensor:
        """
        Deviation-based anomaly score (Eq. 8).

        query_emb     : [B, H] or [H]
        fused_baseline: [B, H] or [H]
        """
        single = False
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            fused_baseline = fused_baseline.unsqueeze(0)
            single = True

        cos_dist = 1.0 - F.cosine_similarity(query_emb, fused_baseline, dim=-1)  # [B]
        l2_dist = torch.norm(query_emb - fused_baseline, p=2, dim=-1)           # [B]
        scores = gamma * cos_dist + delta * l2_dist

        if single:
            scores = scores.squeeze(0)
        return scores


def contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss used for both time-aware and
    context-aware contrastive learning.

    query    : [N, H]
    positive : [N, H]
    negatives: [M, H]
    """
    # Normalize
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarity
    pos_sim = torch.exp(torch.sum(query * positive, dim=-1) / temperature)  # [N]

    # Negative similarities: every query sees the same negative pool
    logits_neg = torch.matmul(query, negatives.T) / temperature             # [N, M]
    neg_sim = torch.exp(logits_neg).sum(dim=-1)                             # [N]

    loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
    return loss.mean()
