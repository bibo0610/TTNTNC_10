"""Generate supervised (position, expert move) examples from self-play."""
import argparse
import random
from pathlib import Path
import numpy as np
from agents.cnn_agent import encode_state, legal_mask
from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from agents.mcts_agent import mcts_agent
from core.othello_engine import init_board, get_valid_moves, make_move, is_game_over


def generate(games, seed=42, epsilon=0.15, teacher="heuristic", minimax_depth=3, mcts_simulations=100):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    if teacher not in ("heuristic", "minimax", "mcts"):
        raise ValueError("teacher must be heuristic, minimax or mcts")
    if minimax_depth < 1:
        raise ValueError("minimax_depth must be >= 1")
    if mcts_simulations < 1:
        raise ValueError("mcts_simulations must be >= 1")
    if games < 1:
        raise ValueError("games must be >= 1")
    rng = np.random.default_rng(seed)

    def choose_teacher(board, player):
        if teacher == "minimax":
            return minimax_agent(board, player, depth=minimax_depth)
        if teacher == "mcts":
            # Existing MCTS uses Python's global random. Seed each decision
            # from a local RNG and restore the external RNG state so datasets
            # are reproducible without disturbing other project components.
            saved_rng_state = random.getstate()
            try:
                random.seed(int(rng.integers(0, 2**32)))
                return mcts_agent(board, player, simulations=mcts_simulations)
            finally:
                random.setstate(saved_rng_state)
        return heuristic_agent(board, player)
    states, labels, masks, game_ids = [], [], [], []
    for game_id in range(games):
        board = init_board()
        player = 1
        while not is_game_over(board):
            moves = get_valid_moves(board, player)
            if not moves:
                player = -player
                continue
            expert = choose_teacher(board, player)
            if expert not in moves:
                raise ValueError(f"Teacher returned illegal move {expert} in game {game_id}")
            states.append(encode_state(board, player))
            labels.append(expert[0] * 8 + expert[1])
            masks.append(legal_mask(moves))
            game_ids.append(game_id)
            # Exploration varies encountered boards without corrupting expert labels.
            move = moves[int(rng.integers(len(moves)))] if rng.random() < epsilon else expert
            board = make_move(board, *move, player)
            player = -player
    return {"states": np.asarray(states, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "masks": np.asarray(masks, dtype=np.bool_),
            "game_ids": np.asarray(game_ids, dtype=np.int32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--teacher", choices=["heuristic", "minimax", "mcts"], default="heuristic")
    parser.add_argument("--minimax-depth", type=int, default=3,
                        help="Search depth for minimax teacher (default: 3)")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                        help="MCTS rollouts per expert decision (default: 100)")
    parser.add_argument("--out", default="data/cnn_heuristic.npz")
    args = parser.parse_args()
    if args.games < 2:
        parser.error("--games must be >= 2 for train/validation split")
    if args.minimax_depth < 1:
        parser.error("--minimax-depth must be >= 1")
    if args.mcts_simulations < 1:
        parser.error("--mcts-simulations must be >= 1")
    if args.teacher == "mcts" and args.out == "data/cnn_heuristic.npz":
        parser.error("Specify --out data/cnn_v3_mcts.npz to avoid overwriting V1")
    if args.teacher == "minimax" and args.out == "data/cnn_heuristic.npz":
        parser.error("Specify --out data/cnn_v2_minimax.npz to avoid overwriting V1")
    data = generate(args.games, args.seed, args.epsilon, args.teacher,
                    args.minimax_depth, args.mcts_simulations)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Teacher={args.teacher}, depth={args.minimax_depth if args.teacher == 'minimax' else 'n/a'}, "
          f"mcts_simulations={args.mcts_simulations if args.teacher == 'mcts' else 'n/a'}, "
          f"seed={args.seed}, epsilon={args.epsilon}")
    print(f"Saved {len(data['labels'])} samples from {args.games} games to {path}")

if __name__ == "__main__":
    main()
