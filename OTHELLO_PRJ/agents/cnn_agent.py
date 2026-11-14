"""CNN policy agent for the existing Othello engine. Requires PyTorch."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from core.othello_engine import get_valid_moves

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cnn_v1_heuristic.pt"
MINIMAX_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cnn_v2_minimax.pt"
MCTS_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cnn_v3_mcts.pt"


def encode_state(board, player):
    """3x8x8: current player's stones, opponent's stones, black-to-move flag."""
    if player not in (1, -1):
        raise ValueError("player must be 1 or -1")
    b = np.asarray(board, dtype=np.int8)
    if b.shape != (8, 8):
        raise ValueError("board must be 8x8")
    return np.stack(((b == player).astype(np.float32),
                     (b == -player).astype(np.float32),
                     np.full((8, 8), float(player == 1), dtype=np.float32)))


def legal_mask(moves):
    mask = np.zeros(64, dtype=np.bool_)
    for r, c in moves:
        mask[r * 8 + c] = True
    return mask


class CNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128),
                                  nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        return self.head(self.features(x))


def masked_logits(logits, mask):
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("All samples must have at least one legal move")
    return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)


def load_policy(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    # Load only trusted checkpoints: use weights_only to avoid arbitrary pickles.
    data = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model = CNNPolicy().to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model


class CNNAgent:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
        self.device = device
        self.model = load_policy(checkpoint, device)

    @torch.no_grad()
    def __call__(self, board, player):
        moves = get_valid_moves(board, player)
        if not moves:
            return None  # PASS is handled by the engine/environment
        x = torch.from_numpy(encode_state(board, player)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal_mask(moves)).unsqueeze(0).to(self.device)
        index = masked_logits(self.model(x), mask).argmax(dim=1).item()
        return divmod(index, 8)


@lru_cache(maxsize=1)
def _default_cnn_agent():
    """Load the default checkpoint once; reuse the model between GUI moves."""
    return CNNAgent()


def cnn_agent(board, player):
    """Choose a move with the trained CNN. Same API as heuristic/minimax_agent.

    Returns (row, col), or None when the player must pass.
    Raises FileNotFoundError if the checkpoint has not been trained yet.
    """
    if not get_valid_moves(board, player):
        return None
    return _default_cnn_agent()(board, player)


@lru_cache(maxsize=1)
def _minimax_cnn_agent():
    """Load V2 once, independently of the original heuristic-trained CNN."""
    return CNNAgent(MINIMAX_CHECKPOINT)


def cnn_minimax_agent(board, player):
    """CNN V2 trained on Minimax labels. Same API as other agents."""
    if not get_valid_moves(board, player):
        return None
    return _minimax_cnn_agent()(board, player)


@lru_cache(maxsize=1)
def _mcts_cnn_agent():
    """Load V3 once, independently of V1 and V2."""
    return CNNAgent(MCTS_CHECKPOINT)


def cnn_mcts_agent(board, player):
    """CNN V3 trained on MCTS labels; (board, player) -> move or None."""
    if not get_valid_moves(board, player):
        return None
    return _mcts_cnn_agent()(board, player)
