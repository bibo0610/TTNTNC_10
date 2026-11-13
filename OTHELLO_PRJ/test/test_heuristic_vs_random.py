import random

from core.othello_env import OthelloEnv
from agents.heuristic_agent import heuristic_agent


def random_agent(board, player):
    from core.othello_engine import get_valid_moves

    moves = get_valid_moves(board, player)
    return random.choice(moves) if moves else None


def play_match(heuristic_color):
    env = OthelloEnv()

    while not env.is_terminal():
        player = env.current_player

        if player == heuristic_color:
            action = heuristic_agent(env.board, player)
        else:
            action = random_agent(env.board, player)

        # OthelloEnv kiểm tra nước đi hợp lệ,
        # bao gồm cả PASS (None).
        env.step(action)

    return env.result()


def main():
    results = {
        "win": 0,
        "draw": 0,
        "loss": 0
    }

    total_games = 100

    for i in range(total_games):
        # Luân phiên màu quân
        heuristic_color = 1 if i % 2 == 0 else -1

        winner = play_match(heuristic_color)

        if winner == heuristic_color:
            results["win"] += 1
        elif winner == 0:
            results["draw"] += 1
        else:
            results["loss"] += 1

        print(f"Tran {i + 1}/{total_games}: "
              f"{'Win' if winner == heuristic_color else 'Draw' if winner == 0 else 'Loss'}")

    win_rate = results["win"] / total_games * 100

    print("\n=== HEURISTIC VS RANDOM ===")
    print("Win:", results["win"])
    print("Draw:", results["draw"])
    print("Loss:", results["loss"])
    print(f"Win rate: {win_rate:.2f}%")

    assert sum(results.values()) == total_games


if __name__ == "__main__":
    main()
