"""Tests for train-only D4 symmetry augmentation of CNN policy data."""

import numpy as np
import torch

from agents.cnn_agent import encode_state, legal_mask
from core.othello_engine import init_board, get_valid_moves, make_move
from scripts.cnn_augmentation import transform_d4_batch, augment_d4_batch


def test_all_eight_symmetries_transform_move_and_mask_together():
    states = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    states[0, 0, 2, 3] = 1.0
    states[0, 2, :, :] = 1.0
    labels = torch.tensor([19], dtype=torch.long)
    masks = torch.zeros((1, 64), dtype=torch.bool)
    masks[0, 19] = True
    masks[0, 26] = True
    distinct = set()
    for k in range(4):
        for flip in (False, True):
            s, y, m = transform_d4_batch(states, labels, masks, k, flip)
            assert m[0, y[0]].item()
            assert m.sum().item() == 2
            assert torch.argmax(s[0, 0].reshape(-1)).item() == y.item()
            assert s[0, 2].eq(1).all()
            distinct.add(int(y.item()))
    assert len(distinct) == 8


def test_transforms_match_othello_engine_legal_moves():
    board = init_board()
    for player in (1, -1, 1, -1):
        options = get_valid_moves(board, player)
        board = make_move(board, *options[0], player)
    player = 1
    mask = legal_mask(get_valid_moves(board, player))
    states = torch.from_numpy(encode_state(board, player)).unsqueeze(0)
    labels = torch.tensor([int(np.flatnonzero(mask)[0])])
    masks = torch.from_numpy(mask).unsqueeze(0)
    for k in range(4):
        for flip in (False, True):
            s, y, m = transform_d4_batch(states, labels, masks, k, flip)
            b = np.rot90(np.array(board), k)
            if flip:
                b = np.fliplr(b)
            expected = legal_mask(get_valid_moves(b.tolist(), player))
            assert np.array_equal(m[0].numpy(), expected)
            assert m[0, y[0]].item()
            assert np.array_equal(s[0].numpy(), encode_state(b.tolist(), player))


def test_online_augmentation_retains_legal_targets_and_shapes():
    n = 32
    states = torch.arange(n * 3 * 8 * 8, dtype=torch.float32).reshape(n, 3, 8, 8)
    labels = torch.arange(n) * 2
    masks = torch.zeros((n, 64), dtype=torch.bool)
    masks[torch.arange(n), labels] = True
    torch.manual_seed(42)
    sa, ya, ma = augment_d4_batch(states, labels, masks)
    assert sa.shape == states.shape and ya.shape == labels.shape
    assert ma.shape == masks.shape
    assert ma[torch.arange(n), ya].all()
    assert ma.sum(dim=1).eq(1).all()
    assert torch.equal(torch.sort(sa.flatten(1), dim=1)[0],
                       torch.sort(states.flatten(1), dim=1)[0])
