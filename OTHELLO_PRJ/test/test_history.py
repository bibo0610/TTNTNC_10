import random

from core.othello_env import OthelloEnv
from history.game_history import GameHistory


def main():
    env = OthelloEnv()
    history = GameHistory()

    history.start(env.board)

    while not env.is_terminal():
        player = env.current_player
        actions = env.legal_actions()

        # Nếu không có nước đi, legal_actions trả về [None]
        action = random.choice(actions)

        _, reward, done = env.step(action)

        history.record(
            player=player,
            action=action,
            board=env.board
        )

    winner = env.result()
    history.finish(winner)

    path = history.save()

    print("=== KET QUA TRAN DAU ===")
    winner_name = {
    1: "Den (Black)",
    -1: "Trang (White)",
    0: "Hoa (Draw)"
    }
    print("Nguoi thang:", winner_name[winner])
    print("Tong so luot:", len(history.data["moves"]))
    print("File lich su:", path)

    assert history.data["initial_board"] is not None
    assert history.data["result"] is not None
    assert len(history.data["moves"]) > 0

    print("\nPASS: Game History da duoc luu!")


if __name__ == "__main__":
    main()