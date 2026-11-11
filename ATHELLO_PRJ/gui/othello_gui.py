import tkinter as tk
from tkinter import messagebox, filedialog
from copy import deepcopy
from history.game_replay import GameReplay
from history.game_history import GameHistory
from core.othello_engine import get_winner
from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from agents.cnn_agent import cnn_agent, cnn_minimax_agent

AI_AGENTS = {
    "hva": heuristic_agent,
    "hvm": minimax_agent,
    "hvcnn": cnn_agent,
    "hvcnnminimax": cnn_minimax_agent,
}

from core.othello_engine import (
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

CELL_SIZE = 50

# Pillow dùng để vẽ quân cờ khử răng cưa (anti-aliasing).
# Chưa cài thì GUI vẫn chạy, chỉ quay về cách vẽ create_oval cũ.
try:
    from PIL import Image, ImageDraw, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SUPERSAMPLE = 4   # vẽ to gấp 4 lần rồi thu nhỏ -> viền tròn mượt


def make_disc_image(size_px, diameter, fill, outline, shadow=True):
    """Tạo ảnh một quân cờ tròn mượt, nền trong suốt, kích thước size_px x size_px."""
    s = SUPERSAMPLE
    big = size_px * s
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x0 = (big - diameter * s) // 2
    x1 = x0 + diameter * s

    if shadow:   # bóng đổ nhẹ lệch xuống dưới bên phải
        off = 2 * s
        draw.ellipse((x0 + off, x0 + off, x1 + off, x1 + off), fill=(0, 0, 0, 90))

    draw.ellipse((x0, x0, x1, x1), fill=fill, outline=outline, width=s)

    # Thu nhỏ bằng bộ lọc LANCZOS: các điểm ở mép được pha màu -> hết răng cưa
    img = img.resize((size_px, size_px), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class OthelloGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Othello")

        # Ảnh quân cờ và chấm gợi ý (tạo 1 lần, dùng lại mỗi lần vẽ).
        # Phải giữ trong self, nếu không Python sẽ xóa ảnh và quân bị biến mất.
        # Màu giữ theo file của bạn: viền quân xám, chấm gợi ý #F7DC6F viền xám.
        self.piece_images = {}
        if HAS_PIL:
            margin = 6
            diameter = CELL_SIZE - 2 * margin
            self.piece_images = {
                BLACK: make_disc_image(CELL_SIZE, diameter, (0, 0, 0), (128, 128, 128)),
                WHITE: make_disc_image(CELL_SIZE, diameter, (255, 255, 255), (128, 128, 128)),
                "hint": make_disc_image(CELL_SIZE, 10, (247, 220, 111), (128, 128, 128), shadow=False),
            }

        self.board = create_board()

        # BLACK đi trước
        self.player = BLACK
        self.mode = tk.StringVar(value="hvh")
        self.ai_job = None

        self.replay = None
        self.saved_game = None
        self.history = GameHistory()
        self.history.start(self.board)
        self.history_saved = False
        self.last_log_path = None

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

        self.mode_frame = tk.LabelFrame(root, text="Chế độ chơi")
        self.mode_frame.pack(pady=5)
        tk.Radiobutton(
            self.mode_frame, text="Human vs Human",
            variable=self.mode, value="hvh",
            command=self.change_mode
        ).pack(side="left", padx=10)
        tk.Radiobutton(
            self.mode_frame, text="Human vs Heuristic (bạn: Đen)",
            variable=self.mode, value="hva",
            command=self.change_mode
        ).pack(side="left", padx=10)
        tk.Radiobutton(
            self.mode_frame, text="Human vs Minimax (bạn: Đen)",
            variable=self.mode, value="hvm",
            command=self.change_mode
        ).pack(side="left", padx=10)
        tk.Radiobutton(
            self.mode_frame, text="Human vs CNN(Heuristic) (bạn: Đen)",
            variable=self.mode, value="hvcnn",
            command=self.change_mode
        ).pack(side="left", padx=10)
        tk.Radiobutton(
            self.mode_frame, text="Human vs CNN(Minimax) (bạn: Đen)",
            variable=self.mode, value="hvcnnminimax",
            command=self.change_mode
        ).pack(anchor="w")

        self.restart_button = tk.Button(
            root,
            text="Chơi lại",
            font=("Arial", 12),
            command=self.restart_game
        )

        self.restart_button.pack(pady=5)

        self.replay_frame = tk.Frame(root)
        self.replay_frame.pack(pady=5)

        tk.Button(
            self.replay_frame, text="Open History",
            command=self.open_history
        ).pack(side="left", padx=4)

        self.previous_button = tk.Button(
            self.replay_frame, text="Previous",
            command=self.previous_replay, state="disabled"
        )
        self.previous_button.pack(side="left", padx=4)

        self.next_button = tk.Button(
            self.replay_frame, text="Next",
            command=self.next_replay, state="disabled"
        )
        self.next_button.pack(side="left", padx=4)

        self.close_replay_button = tk.Button(
            self.replay_frame, text="Close Replay",
            command=self.close_replay, state="disabled"
        )
        self.close_replay_button.pack(side="left", padx=4)

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
                    fill="#009067",
                    outline="black"
                )

        # -------------------------
        # Hiển thị nước đi hợp lệ
        # -------------------------

        valid_moves = (
            [] if self.replay is not None or is_game_over(self.board)
            else get_valid_moves(self.board, self.player)
        )

        for row, col in valid_moves:

            x = col * CELL_SIZE + CELL_SIZE / 2
            y = row * CELL_SIZE + CELL_SIZE / 2

            if self.piece_images:
                self.canvas.create_image(x, y, image=self.piece_images["hint"])
                continue

            self.canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill="#F7DC6F",
                outline="gray"
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

                if self.piece_images:
                    cx = col * CELL_SIZE + CELL_SIZE / 2
                    cy = row * CELL_SIZE + CELL_SIZE / 2
                    self.canvas.create_image(cx, cy, image=self.piece_images[value])
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
                    outline="gray"
                )

        self.update_info()


    # =========================
    # CLICK
    # =========================


    
    def handle_click(self, event):
        if self.replay is not None or is_game_over(self.board):
            return
        if self.mode.get() in AI_AGENTS and self.player == WHITE:
            return  # Không cho người chơi đi thay AI.

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if not in_bounds(row, col):
            return
        self.apply_move((row, col))

    def apply_move(self, action):
        """Một đường xử lý chung cho người và AI, bảo đảm log nhất quán."""
        if self.replay is not None or is_game_over(self.board):
            return
        current = self.player
        if action not in get_valid_moves(self.board, current):
            return
        next_board = make_move(self.board, action[0], action[1], current)
        if next_board is None:
            return

        self.board = next_board
        self.history.record(current, action, self.board)
        self.player = -current
        self.after_move()

    def after_move(self):
        if is_game_over(self.board):
            self.cancel_ai_job()
            self.finish_and_save_game()
            self.draw_board()
            self.show_result()
            return

        # Một bên không thể đi thì ghi lại PASS; khi cả hai không
        # thể đi is_game_over đã xử lý ở trên.
        if not get_valid_moves(self.board, self.player):
            skipped = self.player
            self.history.record(skipped, None, self.board)
            self.player = -skipped
            messagebox.showinfo(
                "Bỏ lượt",
                f"{'Đen' if skipped == BLACK else 'Trắng'} không có nước đi hợp lệ."
            )

        self.draw_board()
        self.schedule_ai()

    def cancel_ai_job(self):
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

    def schedule_ai(self):
        self.cancel_ai_job()
        if (self.mode.get() in AI_AGENTS and self.replay is None
                and self.player == WHITE and not is_game_over(self.board)):
            self.ai_job = self.root.after(250, self.ai_move)

    def ai_move(self):
        self.ai_job = None
        if (self.mode.get() not in AI_AGENTS or self.replay is not None
                or self.player != WHITE or is_game_over(self.board)):
            return
        agent = AI_AGENTS[self.mode.get()]
        try:
            action = agent(self.board, WHITE)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            # A missing/incompatible CNN checkpoint must not crash the GUI.
            messagebox.showerror("Không chạy được AI", str(exc))
            self.mode.set("hvh")
            self.restart_game()
            return
        if action is None:
            # Đường phòng vệ cho trạng thái PASS, không dự kiến xảy ra
            # sau after_move() nhưng giúp code ổn định khi mở rộng.
            if not get_valid_moves(self.board, WHITE):
                self.history.record(WHITE, None, self.board)
                self.player = BLACK
                self.draw_board()
            return
        self.apply_move(action)

    def change_mode(self):
        # Đổi chế độ tạo ván mới để không làm sai lịch sử giữa chừng.
        self.restart_game()

    def finish_and_save_game(self):
        if self.history_saved:
            return
        self.history.finish(get_winner(self.board))
        try:
            path = self.history.save()
        except OSError as exc:
            messagebox.showerror("Lỗi lưu lịch sử", f"Không lưu được JSON: {exc}")
            return
        self.history_saved = True
        self.last_log_path = path

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

        if self.replay is not None:
            self.turn_label.config(
                text=f"Replay: {self.replay.index}/{len(self.replay.moves)}"
            )
        elif is_game_over(self.board):
            self.turn_label.config(text="Trận đấu kết thúc")
        else:
            self.turn_label.config(text=f"Lượt: {turn}")


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

        log_info = (f"\nĐã lưu: {self.last_log_path}"
                    if getattr(self, "history_saved", False) else "\nChưa lưu được lịch sử")

        messagebox.showinfo(
            "Kết thúc",
            f"""
Trò chơi kết thúc!

Đen: {black}
Trắng: {white}

{result}{log_info}
"""
        )


    # =========================
    # RESTART
    # =========================

    def restart_game(self):
        self.cancel_ai_job()
        if self.replay is not None:
            self.close_replay()

        self.board = create_board()

        self.player = BLACK
        self.history = GameHistory()
        self.history.start(self.board)
        self.history_saved = False
        self.last_log_path = None
        self.draw_board()


    # =========================
    # GAME REPLAY
    # =========================

    def open_history(self):
        filename = filedialog.askopenfilename(
            title="Chọn lịch sử Othello",
            initialdir="game_logs",
            filetypes=[("JSON files", "*.json")]
        )
        if not filename:
            return

        try:
            new_replay = GameReplay(filename)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            messagebox.showerror("Không mở được lịch sử", str(exc))
            return

        self.cancel_ai_job()
        if self.replay is None:
            self.saved_game = (deepcopy(self.board), self.player)

        self.replay = new_replay
        self.show_replay_position()

    def show_replay_position(self):
        if self.replay is None:
            return

        self.board = deepcopy(self.replay.boards[self.replay.index])
        if self.replay.index < len(self.replay.moves):
            self.player = self.replay.moves[self.replay.index]["player"]

        self.previous_button.config(
            state="normal" if self.replay.index > 0 else "disabled"
        )
        self.next_button.config(
            state="normal" if self.replay.index < len(self.replay.moves) else "disabled"
        )
        self.close_replay_button.config(state="normal")
        self.draw_board()

    def previous_replay(self):
        if self.replay is not None and self.replay.index > 0:
            self.replay.index -= 1
            self.show_replay_position()

    def next_replay(self):
        if self.replay is not None and self.replay.index < len(self.replay.moves):
            self.replay.index += 1
            self.show_replay_position()

    def close_replay(self):
        if self.replay is None:
            return

        self.replay = None
        self.board, self.player = self.saved_game
        self.saved_game = None
        self.previous_button.config(state="disabled")
        self.next_button.config(state="disabled")
        self.close_replay_button.config(state="disabled")
        self.draw_board()
        self.schedule_ai()


# =========================
# MAIN
# =========================

def enable_high_dpi():
    """Trên Windows có đặt Scale 125%/150%, Tk bị phóng to kiểu ảnh bitmap nên nhòe
    và răng cưa. Báo cho Windows biết app tự xử lý DPI để vẽ sắc nét."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass   # không phải Windows hoặc Windows quá cũ -> bỏ qua


def run_gui():
    enable_high_dpi()
    root = tk.Tk()
    OthelloGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()