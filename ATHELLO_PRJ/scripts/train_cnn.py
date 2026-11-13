"""Train the Othello CNN from a teacher dataset, with game-level validation."""
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agents.cnn_agent import CNNPolicy, masked_logits


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args):
    """Train from scratch; save only the epoch with lowest validation loss."""
    data_path = Path(args.data)
    checkpoint = Path(args.out)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}. Use a new --out path, "
            "or pass --overwrite explicitly."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(data_path, allow_pickle=False) as data:
        x = data["states"].copy()
        y = data["labels"].copy()
        masks = data["masks"].copy()
        ids = data["game_ids"].copy()

    if (x.ndim != 4 or x.shape[1:] != (3, 8, 8)
            or y.shape != (len(x),) or masks.shape != (len(x), 64)
            or ids.shape != (len(x),) or len(x) == 0):
        raise ValueError("Invalid dataset shapes or empty dataset")
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Dataset contains invalid states or labels")
    if (y < 0).any() or (y >= 64).any() or not masks[np.arange(len(y)), y].all():
        raise ValueError("Dataset contains out-of-range or illegal teacher actions")

    game_ids = np.unique(ids)
    if len(game_ids) < 2:
        raise ValueError("Need at least 2 games to split train/validation")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(game_ids)
    val_count = max(1, min(len(game_ids) - 1, int(round(len(game_ids) * args.val_split))))
    validation = np.isin(ids, game_ids[:val_count])

    def make_loader(select, shuffle):
        idx = np.flatnonzero(select)
        dataset = TensorDataset(
            torch.from_numpy(x[idx].astype(np.float32)),
            torch.from_numpy(y[idx].astype(np.int64)),
            torch.from_numpy(masks[idx].astype(np.bool_)),
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(~validation, shuffle=True)
    val_loader = make_loader(validation, shuffle=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine. Use --device cpu.")

    model = CNNPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "data": str(data_path), "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "val_split": args.val_split,
        "seed": args.seed, "device": str(device),
        "train_samples": int((~validation).sum()),
        "val_samples": int(validation.sum()),
        "train_games": int(len(game_ids) - val_count), "val_games": int(val_count),
    }
    dataset_sha256 = sha256_file(data_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, seen = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            loss = criterion(masked_logits(model(xb), mb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(xb)
            seen += len(xb)

        model.eval()
        val_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = masked_logits(model(xb), mb)
                val_sum += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += len(xb)
        val_loss = val_sum / total
        val_accuracy = correct / total
        print(f"Epoch {epoch:03d} | train_loss={train_sum/seen:.4f} | "
              f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.2%}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
                "seed": args.seed, "training_config": config,
                "dataset_sha256": dataset_sha256,
            }, checkpoint)
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
                break
    print(f"Best validation loss: {best_loss:.4f} (epoch {best_epoch}); checkpoint: {checkpoint}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cnn_heuristic.npz")
    ap.add_argument("--out", default="checkpoints/cnn_experiment.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6,
                    help="Stop after this many epochs without lower validation loss; 0 disables")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--overwrite", action="store_true",
                    help="Explicitly replace the checkpoint at --out")
    args = ap.parse_args()
    if (args.epochs < 1 or args.patience < 0 or args.batch_size < 1
            or not 0 < args.val_split < 1 or args.lr <= 0 or args.weight_decay < 0):
        ap.error("Invalid training parameters; check epochs, patience, batch-size, lr, weight-decay, val-split")
    train(args)


if __name__ == "__main__":
    main()
