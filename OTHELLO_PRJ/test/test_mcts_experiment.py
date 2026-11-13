#Thực nghiệm, đấu với Random, so sánh 100/500/1000 simulations và lưu kết quả.

import csv
import time
from pathlib import Path
from datetime import datetime

from core.othello_engine import (
    black,
    white,
    empty,
    get_valid_moves,
    random_agent,
    play_game
)

from agents.mcts_agent import mcts_agent


# ==========================================
# 1. CAU HINH THUC NGHIEM
# ==========================================

SIMULATION_LEVELS = [100, 500, 1000]
GAMES_PER_LEVEL = 4

# Thu muc luu ket qua
RESULT_DIR = Path(__file__).resolve().parent.parent / "results" / "mcts"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Tao ma cho moi lan chay
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================
# 2. CAC FILE KET QUA
# ==========================================

POSITION_FILE = RESULT_DIR / "mcts_position_results.csv"
TOURNAMENT_FILE = RESULT_DIR / "mcts_tournament_results.csv"
GAME_DETAILS_FILE = RESULT_DIR / "mcts_game_details.csv"


def save_csv(file_path, fieldnames, row):
    """Them mot dong vao CSV, tu tao header neu file chua ton tai."""

    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or file_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


# ==========================================
# 3. TAO BAN CO BAN DAU
# ==========================================

def create_initial_board():
    board = [[empty for _ in range(8)] for _ in range(8)]

    board[3][3] = white
    board[3][4] = black
    board[4][3] = black
    board[4][4] = white

    return board


# ==========================================
# 4. KIEM TRA TREN MOT BAN CO
# ==========================================

def test_single_position(simulations):
    board = create_initial_board()
    valid_moves = get_valid_moves(board, black)

    start_time = time.perf_counter()

    move = mcts_agent(
        board,
        black,
        simulations=simulations
    )

    elapsed_time = time.perf_counter() - start_time

    legal = move in valid_moves

    row = {
        "run_id": RUN_ID,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "simulations": simulations,
        "selected_move": str(move),
        "decision_time_seconds": round(elapsed_time, 4),
        "legal_move": legal
    }

    save_csv(
        POSITION_FILE,
        [
            "run_id",
            "date_time",
            "simulations",
            "selected_move",
            "decision_time_seconds",
            "legal_move"
        ],
        row
    )

    print(f"Selected move: {move}")
    print(f"Decision time: {elapsed_time:.4f} seconds")
    print(f"Legal move: {'PASSED' if legal else 'FAILED'}")
    print(f"Saved to: {POSITION_FILE}")

    if not legal:
        raise AssertionError(
            f"MCTS chon nuoc di khong hop le: {move}"
        )


# ==========================================
# 5. TOURNAMENT MCTS VS RANDOM
# ==========================================

def run_tournament(simulations):
    stats = {
        "mcts_wins": 0,
        "random_wins": 0,
        "draws": 0,
        "total_time": 0.0,
        "total_moves": 0
    }

    def timed_mcts_agent(board, player):
        start = time.perf_counter()

        move = mcts_agent(
            board,
            player,
            simulations=simulations
        )

        stats["total_time"] += time.perf_counter() - start
        stats["total_moves"] += 1

        return move

    for game_number in range(1, GAMES_PER_LEVEL + 1):

        # Luân phiên màu quân của MCTS
        if game_number % 2 == 1:
            winner, final_board = play_game(
                timed_mcts_agent,
                random_agent,
                verbose=False
            )
            mcts_color = black
            mcts_color_name = "Black"

        else:
            winner, final_board = play_game(
                random_agent,
                timed_mcts_agent,
                verbose=False
            )
            mcts_color = white
            mcts_color_name = "White"

        if winner == mcts_color:
            stats["mcts_wins"] += 1
            result = "MCTS WIN"

        elif winner == empty:
            stats["draws"] += 1
            result = "DRAW"

        else:
            stats["random_wins"] += 1
            result = "RANDOM WIN"

        print(
            f"Game {game_number}/{GAMES_PER_LEVEL}: "
            f"MCTS {mcts_color_name} - {result}"
        )

        # Luu ngay sau moi van dau
        game_row = {
            "run_id": RUN_ID,
            "date_time": datetime.now().isoformat(timespec="seconds"),
            "simulations": simulations,
            "game_number": game_number,
            "mcts_color": mcts_color_name,
            "winner": (
                "MCTS" if winner == mcts_color
                else "DRAW" if winner == empty
                else "RANDOM"
            ),
            "result": result
        }

        save_csv(
            GAME_DETAILS_FILE,
            [
                "run_id",
                "date_time",
                "simulations",
                "game_number",
                "mcts_color",
                "winner",
                "result"
            ],
            game_row
        )

    # Tong hop ket qua
    win_rate = stats["mcts_wins"] / GAMES_PER_LEVEL * 100

    avg_time = (
        stats["total_time"] / stats["total_moves"]
        if stats["total_moves"] > 0 else 0
    )

    summary_row = {
        "run_id": RUN_ID,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "simulations": simulations,
        "games": GAMES_PER_LEVEL,
        "mcts_wins": stats["mcts_wins"],
        "random_wins": stats["random_wins"],
        "draws": stats["draws"],
        "win_rate_percent": round(win_rate, 2),
        "total_mcts_moves": stats["total_moves"],
        "total_thinking_time_seconds": round(stats["total_time"], 4),
        "average_decision_time_seconds": round(avg_time, 4)
    }

    save_csv(
        TOURNAMENT_FILE,
        [
            "run_id",
            "date_time",
            "simulations",
            "games",
            "mcts_wins",
            "random_wins",
            "draws",
            "win_rate_percent",
            "total_mcts_moves",
            "total_thinking_time_seconds",
            "average_decision_time_seconds"
        ],
        summary_row
    )

    return summary_row


# ==========================================
# 6. CHAY THUC NGHIEM
# ==========================================

def main():
    print("=" * 55)
    print("MCTS/UCT EXPERIMENT")
    print("Run ID:", RUN_ID)
    print("=" * 55)

    all_results = []

    for simulations in SIMULATION_LEVELS:
        print("\n" + "=" * 55)
        print(f"SIMULATIONS = {simulations}")
        print("=" * 55)

        # A. Kiem tra tren mot ban co
        print("\n[1] SINGLE POSITION TEST")
        test_single_position(simulations)

        # B. Tournament
        print("\n[2] TOURNAMENT: MCTS VS RANDOM")
        result = run_tournament(simulations)
        all_results.append(result)

        print("\nTournament result:")
        print(f"MCTS wins: {result['mcts_wins']}")
        print(f"Random wins: {result['random_wins']}")
        print(f"Draws: {result['draws']}")
        print(f"Win rate: {result['win_rate_percent']:.2f}%")
        print(
            "Average decision time:",
            result["average_decision_time_seconds"],
            "seconds"
        )

    # ======================================
    # 7. IN BANG TONG HOP
    # ======================================

    print("\n\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Simulations':<15}"
        f"{'MCTS Wins':<12}"
        f"{'Random Wins':<14}"
        f"{'Draws':<8}"
        f"{'Win Rate':<12}"
        f"{'Avg Time':<12}"
    )

    for result in all_results:
        print(
            f"{result['simulations']:<15}"
            f"{result['mcts_wins']:<12}"
            f"{result['random_wins']:<14}"
            f"{result['draws']:<8}"
            f"{result['win_rate_percent']:<12.2f}"
            f"{result['average_decision_time_seconds']:<12.4f}"
        )

    print("\nEXPERIMENT COMPLETED!")
    print("\nAll results saved in:")
    print(RESULT_DIR.resolve())

    print("\nGenerated files:")
    print("-", POSITION_FILE.name)
    print("-", TOURNAMENT_FILE.name)
    print("-", GAME_DETAILS_FILE.name)


if __name__ == "__main__":
    main()