#9 kiểm thử kỹ thuật
import csv
import copy
import unittest
import time
import traceback

from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from core.othello_engine import (
    black,
    white,
    empty,
    init_board,
    get_valid_moves,
    is_game_over
)

from agents.mcts_agent import (
    MCTSNode,
    mcts_agent,
    simulation
)


# =========================================================
# CAU HINH FILE CSV
# =========================================================

RESULT_DIR = Path("results/mcts")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULT_DIR / "mcts_test_results.csv"


# =========================================================
# BAN CO KIEM THU PASS
# Black khong co nuoc di, White van con nuoc di.
# =========================================================

PASS_BOARD = [
    [0, 0, 1, 1, 1, 1, -1, 1],
    [1, 1, 1, 1, 1, -1, -1, 1],
    [1, -1, 1, -1, -1, -1, -1, 1],
    [1, -1, 1, -1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, 1, 1, 1, 1]
]


# =========================================================
# PHAN 1: CAC KIEM THU KY THUAT MCTS
# =========================================================

class TestMCTS(unittest.TestCase):

    def setUp(self):
        self.board = init_board()

    # TEST 1: Nuoc di cua Black phai hop le

    def test_black_move_is_legal(self):
        move = mcts_agent(
            self.board,
            black,
            simulations=20
        )

        valid_moves = get_valid_moves(self.board, black)

        self.assertIsNotNone(move)
        self.assertIn(move, valid_moves)

    # TEST 2: Nuoc di cua White phai hop le

    def test_white_move_is_legal(self):
        move = mcts_agent(
            self.board,
            white,
            simulations=20
        )

        valid_moves = get_valid_moves(self.board, white)

        self.assertIsNotNone(move)
        self.assertIn(move, valid_moves)

    # TEST 3: MCTS khong duoc thay doi ban co goc

    def test_original_board_is_unchanged(self):
        original_board = copy.deepcopy(self.board)

        mcts_agent(
            self.board,
            black,
            simulations=20
        )

        self.assertEqual(self.board, original_board)

    # TEST 4: Xu ly PASS

    def test_pass_returns_none(self):
        self.assertEqual(
            get_valid_moves(PASS_BOARD, black),
            []
        )

        self.assertGreater(
            len(get_valid_moves(PASS_BOARD, white)),
            0
        )

        self.assertFalse(is_game_over(PASS_BOARD))

        move = mcts_agent(
            PASS_BOARD,
            black,
            simulations=20
        )

        self.assertIsNone(move)

    # TEST 5: Expansion xu ly PASS

    def test_expansion_handles_pass(self):
        root = MCTSNode(
            board=PASS_BOARD,
            player_to_move=black
        )

        child = root.expand()

        self.assertIsNotNone(child)
        self.assertIsNone(child.move)
        self.assertEqual(child.player_to_move, white)
        self.assertEqual(child.board, PASS_BOARD)

    # TEST 6: Simulation xu ly PASS

    def test_simulation_handles_pass(self):
        node = MCTSNode(
            board=PASS_BOARD,
            player_to_move=black
        )

        winner = simulation(node)

        self.assertIn(winner, [black, white, empty])
        self.assertEqual(node.board, PASS_BOARD)

    # TEST 7: Ban co ket thuc khong duoc tra ve nuoc di

    def test_terminal_board_returns_none(self):
        terminal_board = [
            [black for _ in range(8)]
            for _ in range(8)
        ]

        self.assertTrue(is_game_over(terminal_board))

        self.assertIsNone(
            mcts_agent(terminal_board, black, simulations=20)
        )

        self.assertIsNone(
            mcts_agent(terminal_board, white, simulations=20)
        )

    # TEST 8: Kiem tra dung so lan Simulation

    def test_simulation_count(self):
        number_of_simulations = 30

        with patch(
            "agents.mcts_agent.simulation",
            return_value=black
        ) as mock_simulation:

            move = mcts_agent(
                self.board,
                black,
                simulations=number_of_simulations
            )

        self.assertEqual(
            mock_simulation.call_count,
            number_of_simulations
        )

        self.assertIn(
            move,
            get_valid_moves(self.board, black)
        )

    # TEST 9: Khong chap nhan simulations <= 0

    def test_invalid_simulation_count(self):
        with self.assertRaises(ValueError):
            mcts_agent(self.board, black, simulations=0)

        with self.assertRaises(ValueError):
            mcts_agent(self.board, black, simulations=-1)


# =========================================================
# PHAN 2: GHI KET QUA UNITTEST VAO CSV
# =========================================================

class CSVTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_records = {}
        self.start_times = {}

    def startTest(self, test):
        self.start_times[test] = time.perf_counter()
        self.test_records[test] = {
            "status": "RUNNING",
            "message": ""
        }
        super().startTest(test)

    def addSuccess(self, test):
        self.test_records[test]["status"] = "PASS"
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.test_records[test]["status"] = "FAIL"
        self.test_records[test]["message"] = "".join(
            traceback.format_exception(*err)
        )
        super().addFailure(test, err)

    def addError(self, test, err):
        self.test_records[test]["status"] = "ERROR"
        self.test_records[test]["message"] = "".join(
            traceback.format_exception(*err)
        )
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.test_records[test]["status"] = "SKIP"
        self.test_records[test]["message"] = reason
        super().addSkip(test, reason)

    def stopTest(self, test):
        elapsed = time.perf_counter() - self.start_times[test]

        self.test_records[test]["duration_seconds"] = round(elapsed, 6)

        super().stopTest(test)


# =========================================================
# PHAN 3: XUAT CSV
# =========================================================

def export_results(result):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    passed = sum(
        1 for record in result.test_records.values()
        if record["status"] == "PASS"
    )

    failed = sum(
        1 for record in result.test_records.values()
        if record["status"] == "FAIL"
    )

    errors = sum(
        1 for record in result.test_records.values()
        if record["status"] == "ERROR"
    )

    skipped = sum(
        1 for record in result.test_records.values()
        if record["status"] == "SKIP"
    )

    # Kiem tra file CSV da ton tai va co du lieu chua
    file_exists = CSV_FILE.exists() and CSV_FILE.stat().st_size > 0

    # Mo file o che do ghi noi tiep
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as file:

        writer = csv.writer(file)

        # Chi ghi header neu file chua co du lieu
        if not file_exists:
            writer.writerow([
                "Test Name",
                "Status",
                "Duration (seconds)",
                "Message",
                "Timestamp"
            ])

        # Ghi ket qua tung test
        for test, record in result.test_records.items():

            writer.writerow([
                test.id(),
                record["status"],
                record.get("duration_seconds", 0),
                record["message"],
                timestamp
            ])

        # Ghi tong ket moi lan chay
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total tests", result.testsRun])
        writer.writerow(["Passed", passed])
        writer.writerow(["Failed", failed])
        writer.writerow(["Errors", errors])
        writer.writerow(["Skipped", skipped])
        writer.writerow([
            "Overall result",
            "PASS" if result.wasSuccessful() else "FAIL"
        ])

    print("\nCSV results appended to:", CSV_FILE)

# =========================================================
# PHAN 4: CHAY KIEM THU
# =========================================================


if __name__ == "__main__":

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMCTS)

    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=CSVTestResult
    )

    result = runner.run(suite)

    export_results(result)

    print("\n========== TEST SUMMARY ==========")

    passed = sum(
        1 for record in result.test_records.values()
        if record["status"] == "PASS"
    )

    failed = sum(
        1 for record in result.test_records.values()
        if record["status"] == "FAIL"
    )

    errors = sum(
        1 for record in result.test_records.values()
        if record["status"] == "ERROR"
    )

    skipped = sum(
        1 for record in result.test_records.values()
        if record["status"] == "SKIP"
    )

    print("Total tests:", result.testsRun)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Errors:", errors)
    print("Skipped:", skipped)

    if result.wasSuccessful():
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL")
