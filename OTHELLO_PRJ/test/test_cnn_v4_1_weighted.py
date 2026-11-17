"""V4.1 source-weighted CE, compatibility and checkpoint tests."""
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts.train_cnn import weighted_teacher_ce, validate_teacher_weights, train


def test_weighted_teacher_loss_matches_hand_calculation():
    logits = torch.tensor([[3., 0.], [0., 3.], [3., 0.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    sources = torch.tensor([0, 1, 2], dtype=torch.long)
    weights = torch.tensor([0.6, 1.5, 0.9])
    ce = F.cross_entropy(logits, labels, reduction='none')
    expected = (ce * weights).sum() / weights.sum()
    actual = weighted_teacher_ce(logits, labels, sources, weights)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, ce.mean())
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.allclose(
        weighted_teacher_ce(logits.detach(), labels, sources, weights * 7),
        actual.detach())


def test_none_or_uniform_teacher_weights_preserve_baseline_loss():
    logits = torch.randn(5, 64)
    labels = torch.tensor([0, 1, 2, 3, 4])
    sources = torch.tensor([0, 1, 2, 0, 2])
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(weighted_teacher_ce(logits, labels, None, None), expected)
    assert torch.allclose(weighted_teacher_ce(
        logits, labels, sources, torch.tensor([1., 1., 1.])), expected)


@pytest.mark.parametrize('w', [[0, 1, 1], [0.6, -1, 0.9], [float('nan'), 1, 1],
                               [float('inf'), 1, 1], [0.5, 0.5], [1, 1, 1, 1]])
def test_reject_invalid_weight_values(w):
    with pytest.raises(ValueError):
        validate_teacher_weights(w, np.array([0, 1, 2]))


def test_reject_missing_or_invalid_provenance():
    with pytest.raises(ValueError, match='teacher_ids'):
        validate_teacher_weights([0.6, 1.5, 0.9], None)
    with pytest.raises(ValueError, match='teacher_ids'):
        validate_teacher_weights([0.6, 1.5, 0.9], np.array([0, 1, 3]))
    with pytest.raises(ValueError, match='three teachers'):
        validate_teacher_weights([0.6, 1.5, 0.9], np.array([0, 1, 1]))


def make_data(tmp_path, with_provenance=True):
    n = 36
    x = np.zeros((n, 3, 8, 8), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    masks = np.ones((n, 64), dtype=bool)
    ids = np.repeat(np.arange(12), 3)
    tids = ids // 4
    for i in range(n):
        x[i, 0, i // 8, i % 8] = 1
        x[i, 2] = tids[i] % 2
        y[i] = (i + int(tids[i])) % 64
    d = dict(states=x, labels=y, masks=masks, game_ids=ids)
    if with_provenance:
        d['teacher_ids'] = tids.astype(np.int8)
    path = tmp_path / 'synthetic.npz'
    np.savez_compressed(path, **d)
    return path


def args_for(path, out, weights):
    return Namespace(data=str(path), out=str(out), epochs=1, patience=1,
                     batch_size=8, lr=0.0005, weight_decay=0.0001,
                     val_split=0.25, seed=42, device='cpu',
                     augment_d4=False, drop_shared_val_states=True,
                     teacher_weights=weights, overwrite=False)


def test_weighted_training_writes_checkpoint_and_keeps_validation_unweighted(tmp_path):
    data = make_data(tmp_path)
    out = tmp_path / 'v4_1.pt'
    train(args_for(data, out, [0.6, 1.5, 0.9]))
    assert out.exists()
    checkpoint = torch.load(out, map_location='cpu', weights_only=True)
    meta = checkpoint['training_config']
    assert meta['teacher_weights'] == [0.6, 1.5, 0.9]
    assert meta['teacher_order'] == ['heuristic', 'minimax', 'mcts']
    assert meta['validation_loss'] == 'unweighted_cross_entropy'
    assert meta['source_stratified'] is True
    assert meta['drop_shared_val_states'] is True
    with pytest.raises(FileExistsError):
        train(args_for(data, out, [0.6, 1.5, 0.9]))


def test_original_training_without_teacher_ids_still_supported(tmp_path):
    data = make_data(tmp_path, with_provenance=False)
    out = tmp_path / 'ordinary.pt'
    train(args_for(data, out, None))
    assert out.exists()
    with pytest.raises(ValueError, match='teacher_ids'):
        train(args_for(data, tmp_path / 'bad.pt', [0.6, 1.5, 0.9]))
