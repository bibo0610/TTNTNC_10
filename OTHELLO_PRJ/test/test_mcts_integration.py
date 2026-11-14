
import csv
import time

from pathlib import Path
from datetime import datetime

from core.othello_engine import (
    black,
    white,
    init_board,
    is_game_over,
    get_winner,
    random_agent,
    play_game
)

from agents.mcts_agent import agent


# =========================================================
# CONFIGURATION
# =========================================================

SIMULATIONS = 1000

# Duong dan project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Thu muc luu ket qua
RESULT_DIR = PROJECT_ROOT / "results" / "mcts"

# File CSV
RESULT_FILE = RESULT_DIR / "mcts_integration_results.csv"


# =========================================================
# SAVE RESULTS
# =========================================================

def save_result(
    timestamp,
    simulations,
    winner,
    elapsed,
    black_count,
    white_count,
    result
):
    """
    Luu ket qua integration test vao CSV.
    Moi lan chay se them mot dong moi.
    """

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = (
        RESULT_FILE.exists()
        and RESULT_FILE.stat().st_size > 0
    )

    with open(
        RESULT_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        # Tao header neu file chua ton tai
        if not file_exists:
            writer.writerow([
                "timestamp",
                "agent",
                "opponent",
                "simulations",
                "winner",
                "black_pieces",
                "white_pieces",
                "elapsed_seconds",
                "result"
            ])

        writer.writerow([
            timestamp,
            "MCTS",
            "Random",
            simulations,
            winner,
            black_count,
            white_count,
            round(elapsed, 2),
            result
        ])

    print("\nCSV saved to:")
    print(RESULT_FILE)


# =========================================================
# INTEGRATION TEST
# =========================================================

def test_mcts_integration():

    print("=" * 60)
    print("MCTS AGENT INTEGRATION TEST")
    print("=" * 60)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print("\nConfiguration:")
    print("MCTS simulations:", SIMULATIONS)
    print("Black player: MCTS")
    print("White player: Random")

    print("\nStarting game...")

    start_time = time.perf_counter()

    try:
        # Chay mot tran MCTS vs Random
        winner, final_board = play_game(
            agent,
            random_agent,
            verbose=False
        )

        elapsed = time.perf_counter() - start_time

        # Dem so quan tren ban co
        black_count = sum(
            row.count(black) for row in final_board
        )

        white_count = sum(
            row.count(white) for row in final_board
        )

        # Kiem tra ban co ket thuc
        assert len(final_board) == 8
        assert all(len(row) == 8 for row in final_board)

        assert is_game_over(final_board)

        # Kiem tra nguoi chien thang
        assert winner in (black, white, 0)

        assert winner == get_winner(final_board)

        # Xac dinh ket qua
        if winner == black:
            winner_name = "Black"
        elif winner == white:
            winner_name = "White"
        else:
            winner_name = "Draw"

        print("\n========== GAME RESULT ==========")

        print("Winner:", winner_name)
        print("Winner code:", winner)

        print("Black pieces:", black_count)
        print("White pieces:", white_count)

        print("Elapsed time:", round(elapsed, 2), "seconds")

        print("Board size:", len(final_board), "x", len(final_board))

        print("\nIntegration test: PASS")

        # Luu ket qua
        save_result(
            timestamp=timestamp,
            simulations=SIMULATIONS,
            winner=winner,
            elapsed=elapsed,
            black_count=black_count,
            white_count=white_count,
            result="PASS"
        )

        return True

    except Exception as error:

        elapsed = time.perf_counter() - start_time

        print("\nIntegration test: FAIL")
        print("Error:", error)

        # Luu lai lan kiem thu that bai
        save_result(
            timestamp=timestamp,
            simulations=SIMULATIONS,
            winner="ERROR",
            elapsed=elapsed,
            black_count="",
            white_count="",
            result="FAIL"
        )

        raise


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    success = test_mcts_integration()

    print("\n" + "=" * 60)

    if success:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL")

    print("=" * 60)