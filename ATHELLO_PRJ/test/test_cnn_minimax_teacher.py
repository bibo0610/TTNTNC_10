"""CNN V2 tests: Minimax teacher and separate GUI entry point."""
import numpy as np
import pytest
import torch

from agents.cnn_agent import CNNPolicy, cnn_minimax_agent, _minimax_cnn_agent
from core.othello_engine import get_valid_moves, init_board
from scripts.generate_cnn_data import generate


def test_minimax_teacher_legal_samples():
    dataset = generate(2, seed=42, epsilon=0.1, teacher="minimax", minimax_depth=1)
    assert dataset["states"].shape[1:] == (3, 8, 8)
    assert len(dataset["labels"]) > 0
    assert dataset["masks"][np.arange(len(dataset["labels"])), dataset["labels"]].all()
    assert set(dataset["game_ids"]) == {0, 1}


def test_minimax_teacher_reproducible():
    first = generate(2, seed=123, teacher="minimax", minimax_depth=1)
    second = generate(2, seed=123, teacher="minimax", minimax_depth=1)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_invalid_minimax_depth():
    with pytest.raises(ValueError, match="minimax_depth"):
        generate(2, teacher="minimax", minimax_depth=0)


def test_v2_agent_independent_checkpoint(tmp_path, monkeypatch):
    import agents.cnn_agent as cnn_module
    checkpoint = tmp_path / "v2.pt"
    torch.save({"model_state_dict": CNNPolicy().state_dict()}, checkpoint)
    cnn_module._minimax_cnn_agent.cache_clear()
    monkeypatch.setattr(cnn_module, "MINIMAX_CHECKPOINT", checkpoint)
    board = init_board()
    assert cnn_minimax_agent(board, 1) in get_valid_moves(board, 1)
    assert cnn_minimax_agent([[1] * 8 for _ in range(8)], 1) is None
    cnn_module._minimax_cnn_agent.cache_clear()
