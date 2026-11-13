"""Evaluate CNN policy against CNN and random, alternating colours."""
import argparse
import random
from agents.cnn_agent import CNNAgent
from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from core.othello_engine import play_game, random_agent
from functools import partial

opponents = [
    ("Random", random_agent),
    ("Heuristic", heuristic_agent),
    ("Minimax (Depth 3)", partial(minimax_agent, depth=3)),
]

def evaluate(agent, opponent, games, seed):
    random.seed(seed)
    wins = draws = losses = 0
    for i in range(games):
        cnn_black = i % 2 == 0
        winner, _ = play_game(agent if cnn_black else opponent,
                              opponent if cnn_black else agent, verbose=False)
        if winner == 0:
            draws += 1
        elif winner == (1 if cnn_black else -1):
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cnn_policy.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.games < 2 or args.games % 2:
        ap.error("--games must be an even number >= 2 for balanced colours")
    agent = CNNAgent(args.checkpoint)
    for name, opponent in opponents:
        w, d, l = evaluate(agent, opponent, args.games, args.seed)
        print(f"CNN vs {name}: W={w} D={d} L={l}; win_rate={w/args.games:.2%}; score_rate={(w+0.5*d)/args.games:.2%}")

if __name__ == "__main__":
    main()
