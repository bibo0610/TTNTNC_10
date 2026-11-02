import json
from pathlib import Path

from core.othello_engine import (
    get_valid_moves,
    make_move,
    is_game_over,
    get_winner,
)

class GameReplay:
    def __init__(self, filepath):
        self.filepath = Path(filepath)

        with open(self.filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.moves = self.data["moves"]
        self.boards = [self.data["initial_board"]]
        self.actions = []

        board = self.boards[0]

        if not isinstance(board, list) or len(board) != 8:
            raise ValueError("Ban co ban dau khong hop le.")

        for index, record in enumerate(self.moves, start=1):
            player = record["player"]
            action = record["action"]

            if action == "PASS":
                if get_valid_moves(board, player):
                    raise ValueError(f"PASS sai tai luot {index}")
                next_board = [row[:] for row in board]
            else:
                row, col = action

                if (row, col) not in get_valid_moves(board, player):
                    raise ValueError(f"Nuoc di sai tai luot {index}")

                next_board = make_move(board, row, col, player)

            if next_board != record["board"]:
                raise ValueError(
                    f"Ban co khong khop tai luot {index}"
                )

            self.actions.append((player, action))
            self.boards.append(next_board)
            board = next_board

        if not is_game_over(board):
            raise ValueError("Lich su chua ket thuc tran dau.")

        if get_winner(board) != self.data["result"]["winner"]:
            raise ValueError("Ket qua tran dau khong khop.")

        self.index = 0

    def show(self):
        board = self.boards[self.index]
        symbols = {1: "B", -1: "W", 0: "."}

        print(f"\n=== LUOT {self.index}/{len(self.moves)} ===")
        print("  a b c d e f g h")

        for i, row in enumerate(board):
            print(
                f"{i+1} " +
                " ".join(symbols[x] for x in row)
            )

        if self.index > 0:
            player, action = self.actions[self.index - 1]
            color = "Den" if player == 1 else "Trang"
            print(f"Vua di: {color} - {action}")

    def next(self):
        if self.index < len(self.moves):
            self.index += 1
        self.show()

    def previous(self):
        if self.index > 0:
            self.index -= 1
        self.show()

    def go_to(self, index):
        if not 0 <= index <= len(self.moves):
            raise ValueError("Chi so luot khong hop le.")
        self.index = index
        self.show()
