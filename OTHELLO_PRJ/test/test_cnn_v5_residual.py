"""V5 residual architecture, old checkpoint compatibility and training smoke tests."""
from argparse import Namespace

import numpy as np
import pytest
import torch

from agents.cnn_agent import (CNNAgent, CNNPolicy, ResidualCNNPolicy,
                              make_policy, load_policy, masked_logits)
from core.othello_engine import init_board, get_valid_moves
from scripts.train_cnn import train


def test_residual_network_shapes_and_parameter_budget():
    model = ResidualCNNPolicy()
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 64)
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in model.parameters()) == 486320
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in CNNPolicy().parameters())
    assert make_policy('cnn_v1').__class__ is CNNPolicy
    assert make_policy('residual_v1').__class__ is ResidualCNNPolicy
    with pytest.raises(ValueError, match='Unsupported'):
        make_policy('something_else')


def test_legacy_checkpoint_still_loads_and_is_legal(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save({'model_state_dict': CNNPolicy().state_dict()}, path)
    agent = CNNAgent(path)
    board = init_board()
    assert isinstance(agent.model, CNNPolicy)
    assert agent(board, 1) in get_valid_moves(board, 1)
    assert agent([[1] * 8 for _ in range(8)], 1) is None


def test_v5_checkpoint_auto_detects_and_is_legal(tmp_path):
    path = tmp_path / 'v5.pt'
    torch.save({'architecture': 'residual_v1',
                'training_config': {'architecture': 'residual_v1'},
                'model_state_dict': ResidualCNNPolicy().state_dict()}, path)
    agent = CNNAgent(path)
    assert isinstance(agent.model, ResidualCNNPolicy)
    board = init_board()
    legal = get_valid_moves(board, 1)
    assert agent(board, 1) in legal
    assert agent(board, 1) == agent(board, 1)  # eval() disables dropout
    assert agent([[1] * 8 for _ in range(8)], 1) is None
    # Invalid action scores must be masked out even for residual checkpoints.
    mask = torch.zeros((1,64), dtype=torch.bool)
    mask[0, [r * 8 + c for r, c in legal]] = True
    with torch.no_grad():
        logits = masked_logits(agent.model(torch.zeros(1,3,8,8)), mask)
    assert logits.argmax(1).item() in [r * 8 + c for r, c in legal]


def test_refuse_unknown_checkpoint_architecture(tmp_path):
    path = tmp_path / 'bad.pt'
    torch.save({'model_state_dict': {}, 'architecture': 'unknown'}, path)
    with pytest.raises(ValueError, match='Unsupported checkpoint architecture'):
        load_policy(path)


def test_v5_training_writes_model_architecture_and_respects_old_default(tmp_path):
    n = 36
    x = np.zeros((n,3,8,8), dtype=np.float32)
    y = np.arange(n, dtype=np.int64) % 64
    masks = np.ones((n,64), dtype=bool)
    ids = np.repeat(np.arange(12),3)
    teacher_ids = (ids // 4).astype(np.int8)
    for i in range(n):
        x[i,0,i//8,i%8] = 1
    dataset = tmp_path / 'dataset.npz'
    np.savez_compressed(dataset, states=x, labels=y, masks=masks,
                        game_ids=ids, teacher_ids=teacher_ids)
    out = tmp_path / 'model.pt'
    args = Namespace(data=str(dataset), out=str(out), epochs=1, patience=1,
                     batch_size=16, lr=0.0005, weight_decay=0.0001,
                     val_split=0.25, seed=42, device='cpu', augment_d4=False,
                     drop_shared_val_states=True, teacher_weights=[0.6,1.5,0.9],
                     architecture='residual_v1', overwrite=False)
    train(args)
    saved = torch.load(out, weights_only=True, map_location='cpu')
    assert saved['architecture'] == 'residual_v1'
    assert saved['training_config']['architecture'] == 'residual_v1'
    assert saved['training_config']['teacher_weights'] == [0.6,1.5,0.9]
    assert isinstance(load_policy(out), ResidualCNNPolicy)
    with pytest.raises(FileExistsError):
        train(args)


def test_v5_gui_registration():
    from gui.othello_gui import MODE_LABELS, AI_AGENTS
    assert 'hvcnnresidual' in MODE_LABELS
    assert callable(AI_AGENTS['hvcnnresidual'])
