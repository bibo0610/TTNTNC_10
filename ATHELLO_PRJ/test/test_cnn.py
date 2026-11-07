import numpy as np
import torch
from agents.cnn_agent import CNNAgent, CNNPolicy, encode_state, legal_mask, masked_logits
from core.othello_engine import init_board, get_valid_moves, make_move
from scripts.generate_cnn_data import generate


def test_initial_encoding_and_mask():
    board = init_board()
    x = encode_state(board, 1)
    moves = get_valid_moves(board, 1)
    assert x.shape == (3, 8, 8)
    assert x[0].sum() == x[1].sum() == 2
    assert x[2].sum() == 64
    assert legal_mask(moves).sum() == 4
    assert masked_logits(torch.zeros(1,64), torch.tensor(legal_mask(moves))[None]).argmax(1).item() in [r*8+c for r,c in moves]


def test_agent_always_legal_and_pass(tmp_path):
    checkpoint = tmp_path / 'policy.pt'
    torch.save({'model_state_dict': CNNPolicy().state_dict()}, checkpoint)
    agent = CNNAgent(checkpoint)
    board = init_board()
    for _ in range(12):
        move = agent(board, 1)
        if not get_valid_moves(board, 1):
            assert move is None
            break
        assert move in get_valid_moves(board, 1)
        board = make_move(board, *move, 1)
        moves = get_valid_moves(board, -1)
        if not moves:
            break
        board = make_move(board, *moves[0], -1)
    assert agent([[1]*8 for _ in range(8)], 1) is None


def test_generated_samples_are_legal():
    d = generate(2, seed=7)
    assert len(d['labels']) > 0
    assert d['masks'][np.arange(len(d['labels'])), d['labels']].all()
    assert set(d['game_ids']) == {0,1}
