from copy import deepcopy

from core.othello_engine import (
    init_board,
    get_valid_moves,
    make_move,
    is_game_over,
    get_winner,
)


class OthelloEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = init_board()
        self.current_player = 1
        return deepcopy(self.board)

    def legal_actions(self):
        if self.is_terminal():
            return []

        moves = get_valid_moves(
            self.board,
            self.current_player
        )

        # PASS được biểu diễn bằng None
        return moves if moves else [None]

    def step(self, action):
        if self.is_terminal():
            raise ValueError("Trận đấu đã kết thúc.")

        legal = self.legal_actions()

        if action not in legal:
            raise ValueError(
                f"Nước đi không hợp lệ: {action}"
            )

        if action is not None:
            self.board = make_move(
                self.board,
                action[0],
                action[1],
                self.current_player
            )

        self.current_player *= -1

        done = self.is_terminal()
        reward = self.result() if done else 0

        return deepcopy(self.board), reward, done

    def is_terminal(self):
        return is_game_over(self.board)

    def result(self):
        if not self.is_terminal():
            return None

        return get_winner(self.board)

    def clone(self):
        return deepcopy(self)

    def render(self):
        symbols = {
            1: "B",
            -1: "W",
            0: "."
        }

        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
