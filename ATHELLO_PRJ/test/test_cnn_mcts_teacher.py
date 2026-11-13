"""V3 teacher, checkpoint isolation, dataset reproducibility, evaluation."""
import random
import numpy as np
import pytest
import torch

from agents.cnn_agent import CNNPolicy
from core.othello_engine import get_valid_moves, init_board
from scripts.generate_cnn_data import generate
from scripts.evaluate_cnn_extended import evaluate
from agents.mcts_agent import mcts_agent


def test_mcts_teacher_generates_legal_samples():
    d = generate(2, seed=42, epsilon=0.15, teacher="mcts", mcts_simulations=5)
    assert d["states"].shape[1:] == (3, 8, 8)
    assert len(d["labels"]) > 0
    assert d["masks"][np.arange(len(d["labels"])), d["labels"]].all()
    assert set(d["game_ids"]) == {0, 1}


def test_mcts_teacher_reproducible_without_changing_global_rng():
    random.seed(777)
    before = random.getstate()
    one = generate(2, seed=123, teacher="mcts", mcts_simulations=3)
    assert random.getstate() == before
    two = generate(2, seed=123, teacher="mcts", mcts_simulations=3)
    for field in one:
        assert np.array_equal(one[field], two[field])


def test_invalid_mcts_simulations():
    with pytest.raises(ValueError, match="mcts_simulations"):
        generate(2, teacher="mcts", mcts_simulations=0)


def test_v3_agent_uses_independent_checkpoint(tmp_path, monkeypatch):
    import agents.cnn_agent as cnn_module
    path = tmp_path / "v3.pt"
    torch.save({"model_state_dict": CNNPolicy().state_dict()}, path)
    cnn_module._mcts_cnn_agent.cache_clear()
    monkeypatch.setattr(cnn_module, "MCTS_CHECKPOINT", path)
    board = init_board()
    assert cnn_module.cnn_mcts_agent(board, 1) in get_valid_moves(board, 1)
    assert cnn_module.cnn_mcts_agent([[1] * 8 for _ in range(8)], 1) is None
    cnn_module._mcts_cnn_agent.cache_clear()


def test_mcts_evaluation_opponent_legal():
    # Lightweight integration smoke test, two paired games.
    from functools import partial
    def first_legal(board, player):
        return get_valid_moves(board, player)[0]
    _, overall, openings = evaluate(first_legal, partial(mcts_agent, simulations=3), 2, 42, 4)
    assert sum(overall.values()) == 2
    assert openings == 1
