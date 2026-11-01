import tkinter as tk
from tkinter import messagebox

from othello_engine import (
    init_board as create_board,
    get_valid_moves,
    make_move,
    count_pieces,
    is_game_over,
    in_bounds,
    black as BLACK,
    white as WHITE,
    empty as EMPTY,
    size as SIZE,
)

CELL_SIZE = 70

class OthelloGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Othello")

        self.board = create_board()

        # BLACK đi trước
        self.player = BLACK

        # -------------------------
        # Thông tin
        # -------------------------

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(pady=10)

        self.turn_label = tk.Label(
            self.info_frame,
            text="Lượt: Đen",
            font=("Arial", 16, "bold")
        )

        self.turn_label.pack()

        self.score_label = tk.Label(
            self.info_frame,
            text="Đen: 2 | Trắng: 2",
            font=("Arial", 13)
        )

        self.score_label.pack(pady=5)

        # -------------------------
        # Bàn cờ
        # -------------------------

        self.canvas = tk.Canvas(
            root,
            width=SIZE * CELL_SIZE,
            height=SIZE * CELL_SIZE,
            bg="green"
        )

        self.canvas.pack()

        self.canvas.bind(
            "<Button-1>",
            self.handle_click
        )

        # -------------------------
        # Button
        # -------------------------

        self.restart_button = tk.Button(
            root,
            text="Chơi lại",
            font=("Arial", 12),
            command=self.restart_game
        )

        self.restart_button.pack(pady=10)

        self.draw_board()


    # =========================
    # DRAW BOARD
    # =========================

    def draw_board(self):

        self.canvas.delete("all")

        # Vẽ các ô bàn cờ
        for row in range(SIZE):

            for col in range(SIZE):

                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE

                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#228B22",
                    outline="black"
                )

        # -------------------------
        # Hiển thị nước đi hợp lệ
        # -------------------------

        valid_moves = get_valid_moves(
            self.board,
            self.player
        )

        for row, col in valid_moves:

            x = col * CELL_SIZE + CELL_SIZE / 2
            y = row * CELL_SIZE + CELL_SIZE / 2

            self.canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill="yellow",
                outline=""
            )

        # -------------------------
        # Vẽ quân
        # -------------------------

        margin = 6

        for row in range(SIZE):

            for col in range(SIZE):

                value = self.board[row][col]

                if value == EMPTY:
                    continue

                x1 = col * CELL_SIZE + margin
                y1 = row * CELL_SIZE + margin

                x2 = (col + 1) * CELL_SIZE - margin
                y2 = (row + 1) * CELL_SIZE - margin

                if value == BLACK:
                    color = "black"
                else:
                    color = "white"

                self.canvas.create_oval(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    outline="black"
                )

        self.update_info()


    # =========================
    # CLICK
    # =========================

    def handle_click(self, event):

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if not in_bounds(row, col):
            return

        # Nước đi không hợp lệ
        if not get_valid_moves(
            self.board,
            self.player
        ):
            return

        # Thực hiện nước đi
        self.board = make_move(self.board, row, col, self.player)

        # Đổi lượt
        self.player = -self.player

        # Kiểm tra kết thúc
        if is_game_over(self.board):

            self.draw_board()
            self.show_result()

            return

        # -------------------------
        # PASS
        # -------------------------

        if not get_valid_moves(
            self.board,
            self.player
        ):

            player_name = (
                "Đen"
                if self.player == BLACK
                else "Trắng"
            )

            messagebox.showinfo(
                "Pass",
                f"{player_name} không có nước đi hợp lệ.\nBỏ lượt!"
            )

            self.player = -self.player

        self.draw_board()


    # =========================
    # SCORE
    # =========================

    def update_info(self):

        black, white = count_pieces(
            self.board
        )

        self.score_label.config(
            text=f"Đen: {black} | Trắng: {white}"
        )

        if self.player == BLACK:
            turn = "Đen"
        else:
            turn = "Trắng"

        self.turn_label.config(
            text=f"Lượt: {turn}"
        )


    # =========================
    # GAME OVER
    # =========================

    def show_result(self):

        black, white = count_pieces(
            self.board
        )

        if black > white:
            result = "Đen thắng!"

        elif white > black:
            result = "Trắng thắng!"

        else:
            result = "Hòa!"

        messagebox.showinfo(
            "Kết thúc",
            f"""
Trò chơi kết thúc!

Đen: {black}
Trắng: {white}

{result}
"""
        )


    # =========================
    # RESTART
    # =========================

    def restart_game(self):

        self.board = create_board()

        self.player = BLACK

        self.draw_board()


# =========================
# MAIN
# =========================

def run_gui():
    root = tk.Tk()
    OthelloGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()