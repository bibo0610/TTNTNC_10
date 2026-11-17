"""Dataset balancing and train/validation isolation for CNN V4."""
import json
import numpy as np
import pytest

from scripts.build_cnn_v4_data import make_balanced
from scripts.train_cnn import split_by_source_game, exclude_train_overlaps


def write_source(path, teacher, n=12):
    states = np.zeros((n, 3, 8, 8), dtype=np.float32)
    labels = np.arange(n, dtype=np.int64) % 64
    # Same encoded state occurs in each teacher, with a different label.
    for i in range(n):
        if i == 0:
            states[i, 0, 0, 0] = 1
            labels[i] = teacher + 3
        else:
            states[i, 0, i // 8, i % 8] = 1
            states[i, 2] = (teacher + 1) / 4
    masks = np.ones((n, 64), dtype=bool)
    games = np.repeat(np.arange(n // 3, dtype=np.int32), 3)
    np.savez_compressed(path, states=states, labels=labels, masks=masks, game_ids=games)


def test_balanced_dataset_provenance_and_legal_labels(tmp_path):
    src = [tmp_path / f"teacher{i}.npz" for i in range(3)]
    for i, p in enumerate(src):
        write_source(p, i)
    out = tmp_path / "v4.npz"
    meta = make_balanced(src, out, per_teacher=9, seed=42)
    assert meta["total_samples"] == 27
    assert meta["source_conflicting_states"] >= 1
    assert json.loads(out.with_suffix('.json').read_text())['per_teacher'] == 9
    with np.load(out, allow_pickle=False) as data:
        assert data['states'].shape == (27, 3, 8, 8)
        assert data['labels'].shape == (27,)
        assert np.array_equal(np.bincount(data['teacher_ids'], minlength=3), [9, 9, 9])
        assert data['masks'][np.arange(27), data['labels']].all()
        groups = [set(data['game_ids'][data['teacher_ids'] == i].tolist()) for i in range(3)]
        assert not groups[0] & groups[1]
        assert not groups[1] & groups[2]
    with pytest.raises(FileExistsError):
        make_balanced(src, out, per_teacher=9)


def test_deterministic_balancing_and_discard_conflicting_states(tmp_path):
    src = [tmp_path / f"teacher{i}.npz" for i in range(3)]
    for i, p in enumerate(src):
        write_source(p, i)
    paths = [tmp_path / 'one.npz', tmp_path / 'two.npz']
    for out in paths:
        make_balanced(src, out, per_teacher=6, seed=123, conflicts='drop')
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
        assert len(a['labels']) == 18
        assert not np.any(np.all(a['states'][:, 0] == np.pad(np.array([[1.]], dtype=np.float32), [(0,7),(0,7)]), axis=(1,2)))


def test_stratified_game_split_and_overlap_filter():
    games = np.concatenate([np.repeat(np.arange(10), 3), np.repeat(np.arange(10, 20), 3)])
    teachers = np.concatenate([np.zeros(30, dtype=np.int8), np.ones(30, dtype=np.int8)])
    val = split_by_source_game(games, 42, 0.2, teachers)
    assert np.count_nonzero(val) == 12
    for g in np.unique(games):
        assert val[games == g].all() or not val[games == g].any()
    assert val[teachers == 0].sum() == val[teachers == 1].sum() == 6
    states = np.zeros((6, 3, 8, 8), dtype=np.float32)
    states[:, 0, 0, 0] = [1, 2, 3, 1, 4, 5]
    train_select = np.array([True, True, True, False, False, False])
    val_select = ~train_select
    filtered, dropped = exclude_train_overlaps(states, train_select, val_select)
    assert dropped == 1
    assert np.array_equal(filtered, [False, False, False, False, True, True])
