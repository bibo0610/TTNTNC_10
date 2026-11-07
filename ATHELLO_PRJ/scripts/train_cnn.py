"""Supervised CNN training, game-level validation split, and best checkpoint."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from agents.cnn_agent import CNNPolicy, masked_logits


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        x = data["states"].copy()
        y = data["labels"].copy()
        masks = data["masks"].copy()
        ids = data["game_ids"].copy()
    if x.ndim != 4 or x.shape[1:] != (3, 8, 8) or y.shape != (len(x),) or masks.shape != (len(x), 64):
        raise ValueError("Invalid data shapes")
    if not masks[np.arange(len(y)), y].all():
        raise ValueError("Teacher action must be legal")
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("Need at least 2 games for train/validation split")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    val_games = max(1, int(round(len(unique) * args.val_split)))
    val_games = min(len(unique) - 1, val_games)
    is_val = np.isin(ids, unique[:val_games])
    def loader(indices, shuffle):
        idx = np.flatnonzero(indices)
        ds = TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]), torch.from_numpy(masks[idx]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)
    train_loader = loader(~is_val, True)
    val_loader = loader(is_val, False)
    device = torch.device(args.device)
    model = CNNPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1
    checkpoint = Path(args.out)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, count = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = masked_logits(model(xb), mb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            count += len(xb)
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = masked_logits(model(xb), mb)
                val_loss += loss_fn(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(xb)
        acc = correct / total
        print(f"Epoch {epoch:03d} | train_loss={train_loss/count:.4f} | val_loss={val_loss/total:.4f} | val_accuracy={acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_accuracy": acc, "seed": args.seed}, checkpoint)
    print(f"Best validation accuracy: {best_acc:.2%}; checkpoint: {checkpoint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cnn_heuristic.npz")
    ap.add_argument("--out", default="checkpoints/cnn_policy.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.epochs < 1 or not 0 < args.val_split < 1:
        ap.error("epochs must be >= 1 and val-split must be between 0 and 1")
    train(args)

if __name__ == "__main__":
    main()
