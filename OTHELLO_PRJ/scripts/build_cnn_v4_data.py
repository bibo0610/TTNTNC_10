"""Construct a sample-balanced CNN V4 dataset from three teacher datasets.

The datasets must already exist; MCTS 128 is NOT regenerated here.
Use game-level provenance, inspect conflicting teacher labels, and avoid overwrites.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TEACHERS = ("heuristic", "minimax", "mcts")
REQUIRED = ("states", "labels", "masks", "game_ids")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_teacher(path, name):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} dataset missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED) - set(data.files)
        if missing:
            raise ValueError(f"{name}: missing arrays {sorted(missing)}")
        d = {k: data[k].copy() for k in REQUIRED}
    states, labels, masks, games = (d[k] for k in REQUIRED)
    n = len(labels)
    if (n < 1 or states.shape != (n, 3, 8, 8)
            or masks.shape != (n, 64) or games.shape != (n,)):
        raise ValueError(f"{name}: invalid input array shapes")
    if (not np.issubdtype(labels.dtype, np.integer)
            or not np.issubdtype(games.dtype, np.integer)
            or not np.isfinite(states).all()):
        raise ValueError(f"{name}: invalid states, labels or game ids")
    if (labels < 0).any() or (labels >= 64).any():
        raise ValueError(f"{name}: labels must be in 0..63")
    masks = masks.astype(np.bool_, copy=False)
    if (not masks.any(axis=1).all()
            or not masks[np.arange(n), labels].all()):
        raise ValueError(f"{name}: label not legal according to mask")
    d["masks"] = masks
    return d


def find_conflicting_states(datasets):
    """Conflicts count distinct exactly equal encoded states with >1 label."""
    actions = defaultdict(set)
    teachers = defaultdict(set)
    for tid, data in enumerate(datasets):
        for state, label in zip(data["states"], data["labels"]):
            key = np.ascontiguousarray(state).tobytes()
            actions[key].add(int(label))
            teachers[key].add(tid)
    conflicts = {state for state, labels in actions.items() if len(labels) > 1}
    cross = sum(len(teachers[k]) > 1 for k in conflicts)
    return conflicts, len(actions), cross


def make_balanced(paths, out, per_teacher=0, seed=42, conflicts="keep"):
    """Build the dataset; return written metadata. per_teacher=0 means minimum size."""
    if conflicts not in ("keep", "drop"):
        raise ValueError("conflicts must be keep or drop")
    if per_teacher < 0:
        raise ValueError("per_teacher must be >=0")
    out = Path(out)
    meta_path = out.with_suffix(".json")
    if out.exists() or meta_path.exists():
        raise FileExistsError(f"Output already exists: {out} or {meta_path}; use another filename")
    datasets = [read_teacher(p, name) for name, p in zip(TEACHERS, paths)]
    conflicting, unique_states, cross_conflicts = find_conflicting_states(datasets)
    eligible = []
    for d in datasets:
        if conflicts == "drop":
            ids = np.fromiter((i for i, s in enumerate(d["states"])
                               if np.ascontiguousarray(s).tobytes() not in conflicting), dtype=np.int64)
        else:
            ids = np.arange(len(d["labels"]), dtype=np.int64)
        eligible.append(ids)
    target = per_teacher or min(len(x) for x in eligible)
    if target < 2 or any(len(x) < target for x in eligible):
        raise ValueError(f"Not enough samples to balance at {target}; available {[len(x) for x in eligible]}")
    rng = np.random.default_rng(seed)
    collected = {k: [] for k in REQUIRED}
    collected["teacher_ids"] = []
    next_game_id = 0
    chosen_stats = []
    for tid, (name, d, possible) in enumerate(zip(TEACHERS, datasets, eligible)):
        selected = np.sort(rng.choice(possible, size=target, replace=False))
        original_games = d["game_ids"][selected]
        _, remapped = np.unique(original_games, return_inverse=True)
        collected["game_ids"].append((remapped + next_game_id).astype(np.int32))
        next_game_id += len(np.unique(original_games))
        for k in ("states", "labels", "masks"):
            collected[k].append(d[k][selected])
        collected["teacher_ids"].append(np.full(target, tid, dtype=np.int8))
        chosen_stats.append({"teacher": name, "available_samples": int(len(d["labels"])),
                             "eligible_samples": int(len(possible)),
                             "selected_samples": int(target),
                             "selected_games": int(len(np.unique(original_games))),
                             "source_path": str(paths[tid]), "sha256": sha256(paths[tid])})

    combined = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    shuffled = rng.permutation(len(combined["labels"]))
    combined = {k: v[shuffled] for k, v in combined.items()}
    metadata = {
        "method": "balanced_without_replacement", "teachers": list(TEACHERS),
        "seed": int(seed), "per_teacher": int(target), "total_samples": int(len(combined["labels"])),
        "total_games": int(next_game_id), "conflict_policy": conflicts,
        "source_unique_encoded_states": int(unique_states),
        "source_conflicting_states": int(len(conflicting)),
        "cross_teacher_conflicting_states": int(cross_conflicts),
        "sources": chosen_stats,
        "notes": "teacher_ids 0=Heuristic, 1=Minimax, 2=MCTS; game_ids remapped per source; sample balance does not ensure equal number of unique states",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **combined)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"V4 dataset: {len(combined['labels'])} samples; {target} per teacher; {next_game_id} source games")
    print(f"Conflicting distinct states in sources: {len(conflicting)} (cross-teacher: {cross_conflicts}); policy={conflicts}")
    print(f"Saved: {out} and {meta_path}")
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heuristic", default="data/cnn_v1_heuristic.npz")
    ap.add_argument("--minimax", default="data/cnn_v2_minimax.npz")
    ap.add_argument("--mcts", default="data/cnn_v3_1_mcts_128_final.npz")
    ap.add_argument("--out", default="data/cnn_v4_multi_teacher.npz")
    ap.add_argument("--per-teacher", type=int, default=0,
                    help="0 = automatically use smallest eligible source; otherwise samples per teacher")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conflicts", choices=("keep", "drop"), default="keep",
                    help="keep and report disagreeing labels (default), or discard all conflicting states")
    args = ap.parse_args()
    make_balanced([args.heuristic, args.minimax, args.mcts], args.out,
                  args.per_teacher, args.seed, args.conflicts)


if __name__ == "__main__":
    main()
