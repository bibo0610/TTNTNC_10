"""Train the Othello CNN from a teacher dataset, with game-level validation."""
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.cnn_agent import ARCHITECTURES, make_policy, masked_logits
from scripts.cnn_augmentation import augment_d4_batch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_by_source_game(ids, seed, val_split, teacher_ids=None):
    """Stratify game-level holdouts per teacher when provenance is available."""
    rng = np.random.default_rng(seed)
    validation = np.zeros(len(ids), dtype=np.bool_)
    if teacher_ids is None:
        game_ids = np.unique(ids)
        if len(game_ids) < 2:
            raise ValueError("Need at least 2 games for validation")
        rng.shuffle(game_ids)
        count = max(1, min(len(game_ids) - 1, int(round(len(game_ids) * val_split))))
        validation = np.isin(ids, game_ids[:count])
        return validation
    for tid in np.unique(teacher_ids):
        within = teacher_ids == tid
        groups = np.unique(ids[within])
        if len(groups) < 2:
            raise ValueError(f"Teacher {tid}: need at least 2 games for validation")
        rng.shuffle(groups)
        count = max(1, min(len(groups) - 1, int(round(len(groups) * val_split))))
        validation |= within & np.isin(ids, groups[:count])
    return validation


def exclude_train_overlaps(states, train_select, val_select):
    """Remove (not move) validation states that occur verbatim in training."""
    train_keys = {np.ascontiguousarray(s).tobytes() for s in states[train_select]}
    to_check = np.flatnonzero(val_select)
    overlap = np.fromiter((np.ascontiguousarray(states[i]).tobytes() in train_keys
                           for i in to_check), dtype=np.bool_, count=len(to_check))
    val_select = val_select.copy()
    val_select[to_check[overlap]] = False
    if not val_select.any():
        raise ValueError("All validation states overlap training; cannot validate")
    return val_select, int(overlap.sum())



def weighted_teacher_ce(logits, labels, teacher_ids, teacher_weights):
    """Weighted supervised loss, normalized by sum of sample weights.

    teacher_weights has exactly three positive values for
    0=Heuristic, 1=Minimax, 2=MCTS. None reproduces ordinary CE.
    The normalization means 0.6/1.5/0.9 and 0.2/0.5/0.3 are equivalent.
    """
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    if teacher_weights is None:
        return per_sample.mean()
    weights = teacher_weights[teacher_ids.long()]
    return (per_sample * weights).sum() / weights.sum()


def validate_teacher_weights(teacher_weights, teacher_ids):
    """Validate provenance before giving any source more influence."""
    if teacher_weights is None:
        return None
    if teacher_ids is None:
        raise ValueError("--teacher-weights requires dataset field teacher_ids (build V4 dataset first)")
    weights = np.asarray(teacher_weights, dtype=np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("--teacher-weights must provide exactly 3 finite, positive numbers")
    if not np.isin(teacher_ids, (0, 1, 2)).all():
        raise ValueError("teacher_ids must be 0=Heuristic, 1=Minimax, 2=MCTS")
    counts = np.bincount(teacher_ids.astype(np.int64), minlength=3)
    if (counts == 0).any():
        raise ValueError("Weighted Multi-Teacher requires samples from all three teachers")
    return weights

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
        teacher_ids = data["teacher_ids"].copy() if "teacher_ids" in data.files else None

    if (x.ndim != 4 or x.shape[1:] != (3, 8, 8)
            or y.shape != (len(x),) or masks.shape != (len(x), 64)
            or ids.shape != (len(x),) or len(x) == 0):
        raise ValueError("Invalid dataset shapes or empty dataset")
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Dataset contains invalid states or labels")
    if (y < 0).any() or (y >= 64).any() or not masks[np.arange(len(y)), y].all():
        raise ValueError("Dataset contains out-of-range or illegal teacher actions")

    game_ids = np.unique(ids)
    if teacher_ids is not None and (teacher_ids.shape != ids.shape or
                                      not np.issubdtype(teacher_ids.dtype, np.integer)):
        raise ValueError("Invalid teacher_ids field")
    source_weights = validate_teacher_weights(getattr(args, "teacher_weights", None), teacher_ids)
    validation = split_by_source_game(ids, args.seed, args.val_split, teacher_ids)
    train_select = ~validation
    val_select = validation.copy()
    removed_overlaps = 0
    if args.drop_shared_val_states:
        val_select, removed_overlaps = exclude_train_overlaps(x, train_select, val_select)
        print(f"Removed {removed_overlaps} validation examples sharing an exact training state")

    def make_loader(select, shuffle):
        idx = np.flatnonzero(select)
        tensors = [
            torch.from_numpy(x[idx].astype(np.float32)),
            torch.from_numpy(y[idx].astype(np.int64)),
            torch.from_numpy(masks[idx].astype(np.bool_)),
        ]
        if source_weights is not None:
            tensors.append(torch.from_numpy(teacher_ids[idx].astype(np.int64)))
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_select, shuffle=True)
    val_loader = make_loader(val_select, shuffle=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine. Use --device cpu.")

    architecture = getattr(args, "architecture", "cnn_v1")
    model = make_policy(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()  # validation remains unweighted, as in V4.0
    teacher_weights_tensor = (torch.tensor(source_weights, dtype=torch.float32, device=device)
                              if source_weights is not None else None)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "data": str(data_path), "epochs": args.epochs, "patience": args.patience,
        "architecture": architecture,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "val_split": args.val_split,
        "seed": args.seed, "device": str(device),
        "train_samples": int(train_select.sum()),
        "val_samples": int(val_select.sum()),
        "train_games": int(len(np.unique(ids[train_select]))),
        "val_games": int(len(np.unique(ids[val_select]))),
        "source_stratified": bool(teacher_ids is not None),
        "drop_shared_val_states": bool(args.drop_shared_val_states),
        "dropped_val_overlaps": int(removed_overlaps),
        "augmentation": "random_d4_train_only" if args.augment_d4 else "none",
        "teacher_weights": source_weights.tolist() if source_weights is not None else None,
        "teacher_order": ["heuristic", "minimax", "mcts"] if source_weights is not None else None,
        "validation_loss": "unweighted_cross_entropy",
    }
    dataset_sha256 = sha256_file(data_path)
    print(f"Augmentation: {config['augmentation']}; validation remains unchanged")
    if source_weights is not None:
        counts = np.bincount(teacher_ids[train_select], minlength=3)
        proportions = (counts * source_weights) / np.sum(counts * source_weights)
        print(f"Teacher weights [Heuristic, Minimax, MCTS]: {source_weights.tolist()}")
        print(f"Train teacher counts: {counts.tolist()}; weighted contribution: {proportions.round(4).tolist()}")
        print("Training: weighted Cross-Entropy; validation: ordinary Cross-Entropy")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_weight_sum, seen = 0.0, 0.0, 0
        for batch in train_loader:
            xb, yb, mb = (v.to(device) for v in batch[:3])
            tids = batch[3].to(device) if teacher_weights_tensor is not None else None
            if args.augment_d4:
                xb, yb, mb = augment_d4_batch(xb, yb, mb)
            logits = masked_logits(model(xb), mb)
            loss = weighted_teacher_ce(logits, yb, tids, teacher_weights_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if teacher_weights_tensor is None:
                batch_weight = float(len(xb))
            else:
                batch_weight = teacher_weights_tensor[tids].sum().item()
            train_sum += loss.item() * batch_weight
            train_weight_sum += batch_weight
            seen += len(xb)

        model.eval()
        val_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb, mb = (v.to(device) for v in batch[:3])
                logits = masked_logits(model(xb), mb)
                val_sum += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += len(xb)
        val_loss = val_sum / total
        val_accuracy = correct / total
        print(f"Epoch {epoch:03d} | train_loss={train_sum/train_weight_sum:.4f} | "
              f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.2%}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "architecture": architecture,
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
    ap.add_argument("--architecture", choices=ARCHITECTURES, default="cnn_v1",
                    help="cnn_v1 preserves V1-V4.1; residual_v1 trains the new V5 model")
    ap.add_argument("--augment-d4", action="store_true",
                    help="Apply random D4 board symmetry to TRAIN batches only (V3.2)")
    ap.add_argument("--drop-shared-val-states", action="store_true",
                    help="Discard exact duplicate validation states also present in training (recommended for V4)")
    ap.add_argument("--teacher-weights", nargs=3, type=float, metavar=("HEURISTIC", "MINIMAX", "MCTS"),
                    help="Optional positive source weights, e.g. 0.6 1.5 0.9 (V4.1). Requires teacher_ids.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Explicitly replace the checkpoint at --out")
    args = ap.parse_args()
    if (args.epochs < 1 or args.patience < 0 or args.batch_size < 1
            or not 0 < args.val_split < 1 or args.lr <= 0 or args.weight_decay < 0):
        ap.error("Invalid training parameters; check epochs, patience, batch-size, lr, weight-decay, val-split")
    if args.teacher_weights is not None:
        vals = np.asarray(args.teacher_weights, dtype=np.float64)
        if not np.isfinite(vals).all() or (vals <= 0).any():
            ap.error("--teacher-weights values must be positive and finite")
    train(args)


if __name__ == "__main__":
    main()
