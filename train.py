import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import DGRACL, contrastive_loss
from utils import augment_sequence, evaluate


class SequenceDataset(Dataset):
    """
    Simple dataset wrapper around pre-tokenized interaction sequences.

    Expected npz file structure:
        - train_sequences: int64 array [N_train, L]
        - train_times    : float32 array [N_train]
        - test_sequences : int64 array [N_test, L]
        - test_times     : float32 array [N_test]
        - test_labels    : int64 / bool array [N_test] (optional)
    """
    def __init__(self, sequences: np.ndarray, times: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.times = torch.as_tensor(times, dtype=torch.float32)

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.times[idx]


def load_npz_dataset(path: str | Path):
    """Load preprocessed dynamic graph sequences from an .npz file."""
    data = np.load(path)
    train_seq = data["train_sequences"]
    train_time = data["train_times"]
    test_seq = data["test_sequences"]
    test_time = data["test_times"]
    test_labels = data["test_labels"] if "test_labels" in data.files else None
    return train_seq, train_time, test_seq, test_time, test_labels


def build_memory(model: DGRACL, loader: DataLoader, device: torch.device):
    """
    Build prefix-aligned memory D from the training set.
    Memory stores (embedding, timestamp) pairs.
    """
    model.eval()
    all_embs = []
    all_times = []
    with torch.no_grad():
        for seq, t in tqdm(loader, desc="Building memory"):
            seq = seq.to(device)
            t = t.to(device)
            emb = model.encode_sequence(seq)  # [B, H]
            all_embs.append(emb.cpu())
            all_times.append(t.cpu())
    mem_emb = torch.cat(all_embs, dim=0)   # [N_train, H]
    mem_time = torch.cat(all_times, dim=0) # [N_train]
    return mem_emb, mem_time


def train_epoch(
    model: DGRACL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_ccl: float = 0.4,
    temperature: float = 0.1,
):
    """
    One training epoch optimizing:
        L_ret = L_tcl + alpha_ccl * L_ccl
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for seq, t in tqdm(loader, desc="Training", leave=False):
        seq = seq.to(device)           # [B, L]
        t = t.to(device)               # [B]

        optimizer.zero_grad()

        # Encode original sequences
        z = model.encode_sequence(seq)  # [B, H]
        B, H = z.size()

        # ----- Time-aware contrastive (Ltcl) -----
        with torch.no_grad():
            time_diff = torch.abs(t.unsqueeze(1) - t.unsqueeze(0))          # [B, B]
            time_diff = time_diff + torch.eye(B, device=device) * 1e9       # avoid self
            pos_indices = torch.argmin(time_diff, dim=1)                    # [B]

        pos_emb = z[pos_indices]                                           # [B, H]

        # Negatives: all other embeddings in the batch
        neg_mask = torch.ones(B, B, dtype=torch.bool, device=device)
        neg_mask[torch.arange(B), pos_indices] = False
        neg_mask[torch.arange(B), torch.arange(B)] = False
        negatives = z[neg_mask].view(-1, H)                                # [B*(B-2), H]

        if negatives.size(0) == 0:
            continue

        loss_tcl = contrastive_loss(z, pos_emb, negatives, temperature=temperature)

        # ----- Context-aware contrastive (Lccl) -----
        aug1_list = []
        aug2_list = []
        for i in range(B):
            masked, cropped = augment_sequence(seq[i])
            # pad cropped to original length
            if cropped.size(0) < seq.size(1):
                pad_len = seq.size(1) - cropped.size(0)
                cropped = torch.cat(
                    [cropped, torch.zeros(pad_len, dtype=cropped.dtype, device=cropped.device)],
                    dim=0,
                )
            aug1_list.append(masked.unsqueeze(0))
            aug2_list.append(cropped.unsqueeze(0))
        aug1 = torch.cat(aug1_list, dim=0)  # [B, L]
        aug2 = torch.cat(aug2_list, dim=0)  # [B, L]

        z1 = model.encode_sequence(aug1)    # [B, H]
        z2 = model.encode_sequence(aug2)    # [B, H]

        neg_ccl = torch.cat([z1, z2], dim=0)  # [2B, H]
        loss_ccl = contrastive_loss(z1, z2, neg_ccl, temperature=temperature)

        loss = loss_tcl + alpha_ccl * loss_ccl
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate_model(
    model: DGRACL,
    mem_emb: torch.Tensor,
    mem_time: torch.Tensor,
    test_loader: DataLoader,
    device: torch.device,
    lambda_decay: float = 0.1,
    gamma: float = 0.5,
    delta: float = 0.5,
    labels: np.ndarray | None = None,
):
    """
    Run retrieval-augmented anomaly scoring on the test set and
    (optionally) compute AUC/F1 if labels are available.
    """
    model.eval()
    mem_emb = mem_emb.to(device)
    mem_time = mem_time.to(device)

    all_scores = []
    with torch.no_grad():
        for seq, t in tqdm(test_loader, desc="Testing"):
            seq = seq.to(device)
            t = t.to(device)  # [B]

            z_q = model.encode_sequence(seq)  # [B, H]
            indices = model.retrieve_demonstrations(z_q, t, mem_emb, mem_time, lambda_decay)  # [B, K]

            fused_list = []
            for i in range(z_q.size(0)):
                idx_i = indices[i]                           # [K]
                retrieved_embs = mem_emb[idx_i]             # [K, H]
                fused = model.fuse_demonstrations(z_q[i], retrieved_embs)  # [H]
                fused_list.append(fused.unsqueeze(0))
            fused_batch = torch.cat(fused_list, dim=0)       # [B, H]

            scores = model.compute_anomaly_score(z_q, fused_batch, gamma=gamma, delta=delta)  # [B]
            all_scores.append(scores.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)

    if labels is None:
        return all_scores, None

    metrics = evaluate(all_scores, labels)
    return all_scores, metrics


def main():
    parser = argparse.ArgumentParser(description="DGRA-CL training script")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset file")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size for token ids")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_ccl", type=float, default=0.4, help="Weight for context CL loss")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_decay", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ Load data ------------
    train_seq, train_time, test_seq, test_time, test_labels = load_npz_dataset(args.data)

    train_dataset = SequenceDataset(train_seq, train_time)
    test_dataset = SequenceDataset(test_seq, test_time)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------ Initialize model ------------
    model = DGRACL(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        K=args.K,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------ Training loop ------------
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            alpha_ccl=args.alpha_ccl,
            temperature=args.temperature,
        )
        print(f"[Epoch {epoch:03d}] Loss = {avg_loss:.4f}")

    # ------------ Build memory ------------
    mem_emb, mem_time = build_memory(model, train_loader, device)

    # ------------ Evaluation ------------
    labels = test_labels if test_labels is not None else None
    scores, metrics = evaluate_model(
        model,
        mem_emb,
        mem_time,
        test_loader,
        device,
        lambda_decay=args.lambda_decay,
        gamma=args.gamma,
        delta=args.delta,
        labels=labels,
    )

    if metrics is not None:
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print("  {}: {:.4f}".format(k, v))
    else:
        print("No test_labels found in dataset; only anomaly scores were computed.")


if __name__ == "__main__":
    main()
