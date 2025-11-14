import torch
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from model import DGRACL, contrastive_loss
from utils import evaluate, augment_sequence, inject_anomalies

def train_epoch(model, train_loader, optimizer, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        sequences, timestamps = batch
        sequences = sequences.to(args.device)
        timestamps = timestamps.to(args.device)
        
        # Encode sequences
        embeddings = model.encode_sequence(sequences)
        
        # Time-aware contrastive loss
        batch_size = embeddings.size(0)
        time_sim = model.time_aware_similarity(
            embeddings, embeddings, timestamps, timestamps, args.lambda_time
        )
        
        # Create positive pairs (same sequence)
        loss_tcl = 0
        for i in range(batch_size):
            positive = embeddings[i]
            negatives = torch.cat([embeddings[:i], embeddings[i+1:]], dim=0)
            loss_tcl += contrastive_loss(embeddings[i], positive, negatives, args.tau)
        loss_tcl /= batch_size
        
        # Context-aware contrastive loss
        loss_ccl = 0
        for seq in sequences:
            masked, cropped = augment_sequence(seq, args.mask_ratio, args.crop_ratio)
            emb_masked = model.encode_sequence(masked.unsqueeze(0))
            emb_cropped = model.encode_sequence(cropped.unsqueeze(0))
            
            negatives = embeddings[torch.randperm(batch_size)[:batch_size-1]]
            loss_ccl += contrastive_loss(emb_masked, emb_cropped, negatives, args.tau)
        loss_ccl /= batch_size
        
        # Total loss
        loss = loss_tcl + args.alpha * loss_ccl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def test(model, test_loader, pool_embeddings, pool_timestamps, args):
    """Test anomaly detection"""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sequences, timestamps, labels = batch
            sequences = sequences.to(args.device)
            timestamps = timestamps.to(args.device)
            
            # Encode query
            query_emb = model.encode_sequence(sequences)
            
            # Retrieve top-K demonstrations
            retrieved_indices = model.retrieve_demonstrations(
                query_emb, timestamps, pool_embeddings, pool_timestamps, args.lambda_time
            )
            
            # Fuse retrieved demonstrations
            for i in range(len(query_emb)):
                retrieved_embs = pool_embeddings[retrieved_indices[i]]
                fused_baseline = model.fuse_demonstrations(retrieved_embs)
                
                # Compute anomaly score
                score = model.compute_anomaly_score(query_emb[i], fused_baseline)
                all_scores.append(score.item())
                all_labels.append(labels[i].item())
    
    # Evaluate
    results = evaluate(np.array(all_scores), np.array(all_labels))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UCI')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--K', type=int, default=7)
    parser.add_argument('--lambda_time', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--crop_ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data (simplified - you need to implement actual data loading)
    print(f"Loading {args.dataset} dataset...")
    # train_loader, test_loader, vocab_size = load_data(args.dataset)
    
    # Initialize model
    model = DGRACL(vocab_size=10000, hidden_dim=args.hidden_dim, 
                   num_layers=args.num_layers, K=args.K).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    print("Training DGRA-CL...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, args)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
    
    # Build normal pattern pool
    print("Building normal pattern pool...")
    pool_embeddings = []
    pool_timestamps = []
    with torch.no_grad():
        for batch in train_loader:
            sequences, timestamps = batch
            emb = model.encode_sequence(sequences.to(args.device))
            pool_embeddings.append(emb)
            pool_timestamps.append(timestamps)
    pool_embeddings = torch.cat(pool_embeddings, dim=0)
    pool_timestamps = torch.cat(pool_timestamps, dim=0)
    
    # Test
    print("Testing...")
    results = test(model, test_loader, pool_embeddings, pool_timestamps, args)
    
    print("\nResults:")
    print(f"AUC-ROC: {results['AUC']:.4f}")
    print(f"F1-Score: {results['F1']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")

if __name__ == "__main__":
    main()
