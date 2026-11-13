"""Paired Othello evaluation from shared random openings, with results split by colour.

Run from the directory containing main.py, e.g.
python -m scripts.evaluate_cnn_extended --checkpoint checkpoints/cnn_v2_minimax.pt --opponent minimax --depth 2 --games 20 --opening-plies 6
"""
import argparse
import random
from collections import Counter
from functools import partial

from agents.cnn_agent import CNNAgent
from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from agents.mcts_agent import mcts_agent
from core.othello_engine import (
    black, white, init_board, get_valid_moves, make_move, is_game_over,
    get_winner, random_agent,
)


def make_opening(seed, plies):
    rng = random.Random(seed)
    board, player = init_board(), black
    for _ in range(plies):
        if is_game_over(board):
            break
        legal = get_valid_moves(board, player)
        if legal:
            board = make_move(board, *rng.choice(legal), player)
        player = -player
    return board, player


def play_from(board, player, black_agent, white_agent):
    board = [row[:] for row in board]
    agents = {black: black_agent, white: white_agent}
    while not is_game_over(board):
        legal = get_valid_moves(board, player)
        if legal:
            move = agents[player](board, player)
            if move not in legal:
                raise ValueError(f"Agent returned illegal move {move!r} for player {player}; legal={legal}")
            board = make_move(board, *move, player)
        player = -player
    return get_winner(board)


def tally(results, colour, winner):
    if winner == 0:
        results[colour]['D'] += 1
    elif winner == colour:
        results[colour]['W'] += 1
    else:
        results[colour]['L'] += 1


def format_line(label, counts):
    n = sum(counts.values())
    w, d, l = (counts[k] for k in ('W', 'D', 'L'))
    return (f"{label}: W={w} D={d} L={l}; "
            f"win_rate={w/n:.2%}; score_rate={(w+0.5*d)/n:.2%}")


def evaluate(agent, opponent, games, seed, opening_plies):
    results = {black: Counter({'W': 0, 'D': 0, 'L': 0}),
               white: Counter({'W': 0, 'D': 0, 'L': 0})}
    openings = set()
    # One pair uses exactly the same opening twice, swapping which side CNN plays.
    for i in range(games // 2):
        board, to_move = make_opening(seed + i, opening_plies)
        openings.add(tuple(cell for row in board for cell in row))
        # Seed global RNG separately for reproducible random-agent continuation.
        random.seed(seed + 2*i + 100000)
        tally(results, black, play_from(board, to_move, agent, opponent))
        random.seed(seed + 2*i + 100001)
        tally(results, white, play_from(board, to_move, opponent, agent))
    overall = results[black] + results[white]
    return results, overall, len(openings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/cnn_v2_minimax.pt')
    parser.add_argument('--opponent', choices=('random', 'heuristic', 'minimax', 'mcts', 'cnn'), required=True)
    parser.add_argument('--opponent-checkpoint', help='Required when --opponent cnn')
    parser.add_argument('--depth', type=int, default=2, help='Minimax depth when opponent is minimax')
    parser.add_argument('--mcts-simulations', type=int, default=100,
                        help='Number of MCTS rollouts per decision when MCTS is opponent')
    parser.add_argument('--games', type=int, default=20, help='Even number; half black and half white')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opening-plies', type=int, default=6, help='Random legal moves before agent play (even preferred)')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error('--games must be an even integer >= 2')
    if args.opening_plies < 0 or args.opening_plies > 50:
        parser.error('--opening-plies must be between 0 and 50')
    if args.depth < 1:
        parser.error('--depth must be >= 1')
    if args.mcts_simulations < 1:
        parser.error('--mcts-simulations must be >= 1')
    if args.opponent == 'cnn' and not args.opponent_checkpoint:
        parser.error('--opponent-checkpoint is required for --opponent cnn')

    agent = CNNAgent(args.checkpoint)
    opponents = {
        'random': lambda: random_agent,
        'heuristic': lambda: heuristic_agent,
        'minimax': lambda: partial(minimax_agent, depth=args.depth),
        'mcts': lambda: partial(mcts_agent, simulations=args.mcts_simulations),
        'cnn': lambda: CNNAgent(args.opponent_checkpoint),
    }
    opponent = opponents[args.opponent]()
    by_colour, overall, unique = evaluate(agent, opponent, args.games, args.seed, args.opening_plies)
    print(f"CNN checkpoint: {args.checkpoint}")
    print(f"Opponent: {args.opponent}"
          + (f" depth={args.depth}" if args.opponent == 'minimax' else '')
          + (f" simulations={args.mcts_simulations}" if args.opponent == 'mcts' else ''))
    print(f"Games: {args.games}; seed={args.seed}; opening_plies={args.opening_plies}; unique_openings={unique}/{args.games//2}")
    print(format_line('CNN BLACK', by_colour[black]))
    print(format_line('CNN WHITE', by_colour[white]))
    print(format_line('TOTAL', overall))


if __name__ == '__main__':
    main()