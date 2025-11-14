import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SequenceEncoder(nn.Module):
    """Sequence encoder for temporal graph sequences"""
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, hidden_dim]
        out = self.transformer(emb)  # [batch, seq_len, hidden_dim]
        return self.output_layer(out.mean(dim=1))  # [batch, hidden_dim]

class GCNFusion(nn.Module):
    """GCN for fusing retrieved demonstrations"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        # x: node features, edge_index: graph structure
        return self.gcn(x, edge_index)

class DGRACL(nn.Module):
    """Main DGRA-CL model"""
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, K=7):
        super().__init__()
        self.encoder = SequenceEncoder(vocab_size, hidden_dim, num_layers)
        self.gcn_fusion = GCNFusion(hidden_dim)
        self.K = K
        self.hidden_dim = hidden_dim
        
    def encode_sequence(self, x):
        """Encode a sequence to embedding"""
        return self.encoder(x)
    
    def time_aware_similarity(self, query_emb, candidate_emb, query_time, candidate_time, lambda_decay):
        """Compute time-decayed similarity"""
        # Cosine similarity
        sim = F.cosine_similarity(query_emb.unsqueeze(1), candidate_emb.unsqueeze(0), dim=-1)
        # Time decay
        time_diff = torch.abs(query_time.unsqueeze(1) - candidate_time.unsqueeze(0))
        time_weight = torch.exp(-lambda_decay * time_diff)
        return sim * time_weight
    
    def retrieve_demonstrations(self, query_emb, query_time, pool_emb, pool_time, lambda_decay):
        """Retrieve top-K similar normal patterns"""
        similarities = self.time_aware_similarity(query_emb, pool_emb, query_time, pool_time, lambda_decay)
        top_k_indices = torch.topk(similarities, self.K, dim=1).indices
        return top_k_indices
    
    def fuse_demonstrations(self, retrieved_embs):
        """Fuse retrieved demonstrations using GCN"""
        # Create a simple chain graph for retrieved demonstrations
        num_demos = retrieved_embs.size(0)
        edge_index = torch.stack([
            torch.arange(num_demos - 1),
            torch.arange(1, num_demos)
        ], dim=0).to(retrieved_embs.device)
        
        fused = self.gcn_fusion(retrieved_embs, edge_index)
        return fused.mean(dim=0)  # Mean pooling
    
    def compute_anomaly_score(self, query_emb, fused_baseline, alpha=0.6, beta=0.4):
        """Compute anomaly score"""
        # Cosine distance
        cos_dist = 1 - F.cosine_similarity(query_emb, fused_baseline, dim=0)
        # Euclidean distance
        l2_dist = torch.norm(query_emb - fused_baseline, p=2)
        return alpha * cos_dist + beta * l2_dist

def contrastive_loss(query, positive, negatives, temperature=0.1):
    """Time-aware contrastive loss"""
    # Normalize
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    # Positive similarity
    pos_sim = torch.exp(torch.sum(query * positive, dim=-1) / temperature)
    
    # Negative similarities
    neg_sim = torch.exp(torch.matmul(query, negatives.T) / temperature).sum(dim=-1)
    
    # Loss
    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    return loss.mean()
