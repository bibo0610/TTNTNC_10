import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from copy import deepcopy
from pathlib import Path

from core.othello_engine import (
    init_board,
    get_valid_moves,
    make_move,
    count_pieces,
    is_game_over,
    get_winner,
    in_bounds,
    black as BLACK,
    white as WHITE,
    empty as EMPTY,
    size as SIZE,
)

from agents.heuristic_agent import heuristic_agent
from agents.minimax_agent import minimax_agent
from agents.cnn_agent import cnn_agent, cnn_minimax_agent
from agents.mcts_agent import agent as mcts_agent

from history.game_history import GameHistory
from history.game_replay import GameReplay


# ============================================================
# THEME
# ============================================================

BG = "#101C19"
PANEL = "#1B3028"
PANEL_LIGHT = "#294739"

TEXT = "#F5F7F4"
MUTED = "#A3B8AA"
ACCENT = "#69D5A5"
GOLD = "#FFD071"

BOARD_LIGHT = "#2D805A"
BOARD_DARK = "#287451"
BOARD_LINE = "#174B35"

CELL_SIZE = 62

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "game_logs"

MODE_LABELS = {
    "hvh": "Human vs Human",
    "hva": "Human vs Heuristic",
    "hvm": "Human vs Minimax",
    "hvcnn": "Human vs CNN (Heuristic)",
    "hvcnnminimax": "Human vs CNN (Minimax)",
    "hvmcts": "Human vs MCTS/UCT",
}

AI_AGENTS = {
    "hva": heuristic_agent,
    "hvm": minimax_agent,
    "hvcnn": cnn_agent,
    "hvcnnminimax": cnn_minimax_agent,
    "hvmcts": mcts_agent,
}


# ============================================================
# OPTIONAL PILLOW
# ============================================================

try:
    from PIL import Image, ImageDraw, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def make_disc_image(size, diameter, color, outline):
    scale = 4
    big = size * scale

    image = Image.new(
        "RGBA",
        (big, big),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(image)

    margin = (big - diameter * scale) // 2

    coords = (
        margin,
        margin,
        margin + diameter * scale,
        margin + diameter * scale
    )

    draw.ellipse(
        coords,
        fill=color,
        outline=outline,
        width=scale
    )

    image = image.resize(
        (size, size),
        Image.Resampling.LANCZOS
    )

    return ImageTk.PhotoImage(image)


# ============================================================
# MAIN GUI
# ============================================================

class OthelloGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Othello AI | Strategy Arena")

        self.root.configure(bg=BG)
        self.root.geometry("1020x720")
        self.root.minsize(950, 680)

        self.board = init_board()
        self.player = BLACK

        self.mode = tk.StringVar(value="hvh")
        self.mode_label = tk.StringVar(
            value=MODE_LABELS["hvh"]
        )

        self.ai_job = None

        self.replay = None
        self.saved_game = None

        self.history = GameHistory()
        self.history.start(self.board)

        self.history_saved = False
        self.last_log_path = None

        self.piece_images = {}

        if HAS_PIL:
            self.piece_images[BLACK] = make_disc_image(
                CELL_SIZE,
                CELL_SIZE - 13,
                "#121B19",
                "#546B5F"
            )

            self.piece_images[WHITE] = make_disc_image(
                CELL_SIZE,
                CELL_SIZE - 13,
                "#F5F3E8",
                "#A9B8AD"
            )

        self.setup_styles()
        self.create_header()
        self.create_layout()
        self.draw_board()

    # ========================================================
    # STYLES
    # ========================================================

    def setup_styles(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Modern.TCombobox",
            fieldbackground=PANEL_LIGHT,
            background=PANEL_LIGHT,
            foreground=TEXT,
            arrowcolor=ACCENT,
            bordercolor=PANEL_LIGHT,
            lightcolor=PANEL_LIGHT,
            darkcolor=PANEL_LIGHT,
            padding=8,
            font=("Segoe UI", 10)
        )

        style.map(
            "Modern.TCombobox",
            fieldbackground=[
                ("readonly", PANEL_LIGHT)
            ],
            foreground=[
                ("readonly", TEXT)
            ],
            selectbackground=[
                ("readonly", PANEL_LIGHT)
            ],
            selectforeground=[
                ("readonly", TEXT)
            ]
        )

    # ========================================================
    # HEADER
    # ========================================================

    def create_header(self):

        header = tk.Frame(
            self.root,
            bg=BG
        )

        header.pack(
            fill="x",
            padx=32,
            pady=(22, 14)
        )

        title_frame = tk.Frame(
            header,
            bg=BG
        )

        title_frame.pack(side="left")

        tk.Label(
            title_frame,
            text="OTHELLO",
            font=("Segoe UI", 23, "bold"),
            bg=BG,
            fg=TEXT
        ).pack(side="left")

        tk.Label(
            title_frame,
            text="  AI",
            font=("Segoe UI", 23, "bold"),
            bg=BG,
            fg=ACCENT
        ).pack(side="left")

        tk.Label(
            header,
            text="8x8  |  STRATEGY ARENA",
            font=("Segoe UI", 10),
            bg=BG,
            fg=MUTED
        ).pack(side="right")

    # ========================================================
    # LAYOUT
    # ========================================================

    
    def create_layout(self):

        # Khung chính chứa bàn cờ và bảng điều khiển
        body = tk.Frame(
            self.root,
            bg=BG
        )

        
        body.pack(
            side="top",
            anchor="n",
            pady=(55, 0)
        )

        # Bàn cờ bên trái
        self.board_panel = tk.Frame(
            body,
            bg=PANEL,
            padx=20,
            pady=18
        )

        self.board_panel.pack(
            side="left",
            anchor="n",
            padx=(0, 20)
        )

        # Bảng điều khiển bên phải
        self.sidebar = tk.Frame(
            body,
            bg=BG,
            width=360
        )

        self.sidebar.pack(
            side="left",
            anchor="n",
            fill="y"
        )

        # Tạo các thành phần giao diện
        self.create_board_area()
        self.create_mode_card()
        self.create_replay_card()
        self.create_status_card()

    # ========================================================
    # REUSABLE UI
    # ========================================================

    
    def create_card(self):
        card = tk.Frame(
            self.sidebar,
            bg=PANEL,
            width=360,
            padx=18,
            pady=16
        )

        card.pack(
            fill="x",
            pady=(0, 14)
        )

        return card

    def create_button(
        self,
        parent,
        text,
        command,
        primary=False,
        state="normal"
    ):

        return tk.Button(
            parent,
            text=text,
            command=command,
            state=state,
            bg=ACCENT if primary else PANEL_LIGHT,
            fg="#092318" if primary else TEXT,
            activebackground=(
                "#7DE3B2" if primary else "#355346"
            ),
            activeforeground=(
                "#092318" if primary else TEXT
            ),
            disabledforeground="#82958B",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=10
        )

    # ========================================================
    # BOARD AREA
    # ========================================================

    def create_board_area(self):

        score_frame = tk.Frame(
            self.board_panel,
            bg=PANEL
        )

        score_frame.pack(
            fill="x",
            pady=(0, 12)
        )

        tk.Label(
            score_frame,
            text="BLACK",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=TEXT
        ).pack(side="left")

        self.score_label = tk.Label(
            score_frame,
            text="02  :  02",
            font=("Segoe UI", 19, "bold"),
            bg=PANEL,
            fg=TEXT
        )

        self.score_label.pack(
            side="left",
            expand=True
        )

        tk.Label(
            score_frame,
            text="WHITE",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=TEXT
        ).pack(side="right")

        board_border = tk.Frame(
            self.board_panel,
            bg="#123D2C",
            padx=3,
            pady=3
        )

        board_border.pack()

        self.canvas = tk.Canvas(
            board_border,
            width=SIZE * CELL_SIZE,
            height=SIZE * CELL_SIZE,
            bg=BOARD_LIGHT,
            highlightthickness=0,
            cursor="hand2"
        )

        self.canvas.pack()

        self.canvas.bind(
            "<Button-1>",
            self.handle_click
        )

        self.turn_label = tk.Label(
            self.board_panel,
            text="Lượt của Đen",
            font=("Segoe UI", 12, "bold"),
            bg=PANEL,
            fg=ACCENT
        )

        self.turn_label.pack(
            pady=(15, 0)
        )

    # ========================================================
    # MODE CARD
    # ========================================================

    def create_mode_card(self):

        card = self.create_card()

        tk.Label(
            card,
            text="CHẾ ĐỘ CHƠI",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w")

        self.mode_combo = ttk.Combobox(
            card,
            textvariable=self.mode_label,
            values=list(MODE_LABELS.values()),
            state="readonly",
            style="Modern.TCombobox",
            font=("Segoe UI", 10)
        )

        self.mode_combo.pack(
            fill="x",
            pady=(12, 12)
        )

        self.mode_combo.bind(
            "<<ComboboxSelected>>",
            self.change_mode
        )

        self.restart_button = self.create_button(
            card,
            "BẮT ĐẦU VÁN MỚI",
            self.restart_game,
            primary=True
        )

        self.restart_button.pack(
            fill="x",
            ipady=9
        )

        tk.Label(
            card,
            text="Bạn cầm Đen khi đấu với AI.",
            font=("Segoe UI", 9),
            bg=PANEL,
            fg=MUTED
        ).pack(
            anchor="w",
            pady=(12, 0)
        )

    # ========================================================
    # REPLAY CARD
    # ========================================================

    def create_replay_card(self):

        card = self.create_card()

        tk.Label(
            card,
            text="LỊCH SỬ TRẬN ĐẤU",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w")

        self.open_button = self.create_button(
            card,
            "Mở file lịch sử (JSON)",
            self.open_history
        )

        self.open_button.pack(
            fill="x",
            pady=(12, 12),
            ipady=7
        )

        controls = tk.Frame(
            card,
            bg=PANEL
        )

        controls.pack(fill="x")

        self.previous_button = self.create_button(
            controls,
            "◀",
            self.previous_replay,
            state="disabled"
        )

        self.previous_button.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=6
        )

        self.replay_count_label = tk.Label(
            controls,
            text="— / —",
            bg=PANEL,
            fg=TEXT,
            width=9,
            font=("Segoe UI", 10)
        )

        self.replay_count_label.pack(
            side="left"
        )

        self.next_button = self.create_button(
            controls,
            "▶",
            self.next_replay,
            state="disabled"
        )

        self.next_button.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=6
        )

        self.close_replay_button = self.create_button(
            card,
            "Đóng Replay",
            self.close_replay,
            state="disabled"
        )

        self.close_replay_button.pack(
            fill="x",
            pady=(10, 0),
            ipady=6
        )

    # ========================================================
    # STATUS CARD
    # ========================================================

    def create_status_card(self):

        card = self.create_card()

        tk.Label(
            card,
            text="TRẠNG THÁI",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=MUTED,
            anchor="w"
        ).pack(fill="x")

        # Khung cố định kích thước cho thông báo
        status_container = tk.Frame(
            card,
            bg=PANEL,
            width=280,
            height=48
        )

        status_container.pack(
            fill="x",
            pady=(10, 0)
        )

        # kích thước của khung chứa
        status_container.pack_propagate(False)

        self.status_label = tk.Label(
            status_container,
            text="Sẵn sàng chơi",
            font=("Segoe UI", 11),
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            justify="left",
            wraplength=270
        )

        self.status_label.pack(
            fill="both",
            expand=True
        )

        # Khung cố định cho thông báo lưu log
        log_container = tk.Frame(
            card,
            bg=PANEL,
            width=280,
            height=36
        )

        log_container.pack(
            fill="x",
            pady=(4, 0)
        )

        log_container.pack_propagate(False)

        self.log_label = tk.Label(
            log_container,
            text="Log: chưa lưu ván hiện tại",
            font=("Segoe UI", 9),
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=270
        )

        self.log_label.pack(
            fill="both",
            expand=True
        )

    # ========================================================
    # DRAW BOARD
    # ========================================================

    def draw_board(self):

        self.canvas.delete("all")

        for row in range(SIZE):
            for col in range(SIZE):

                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE

                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                color = (
                    BOARD_LIGHT
                    if (row + col) % 2 == 0
                    else BOARD_DARK
                )

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=BOARD_LINE
                )

        if (
            self.replay is None
            and not is_game_over(self.board)
        ):
            valid_moves = get_valid_moves(
                self.board,
                self.player
            )

            # Chỉ hiển thị gợi ý cho người chơi.
            if not (
                self.mode.get() in AI_AGENTS
                and self.player == WHITE
            ):
                for row, col in valid_moves:

                    cx = col * CELL_SIZE + CELL_SIZE / 2
                    cy = row * CELL_SIZE + CELL_SIZE / 2

                    self.canvas.create_oval(
                        cx - 5,
                        cy - 5,
                        cx + 5,
                        cy + 5,
                        fill=GOLD,
                        outline=""
                    )

        for row in range(SIZE):
            for col in range(SIZE):

                value = self.board[row][col]

                if value == EMPTY:
                    continue

                cx = col * CELL_SIZE + CELL_SIZE / 2
                cy = row * CELL_SIZE + CELL_SIZE / 2

                if value in self.piece_images:
                    self.canvas.create_image(
                        cx,
                        cy,
                        image=self.piece_images[value]
                    )
                else:
                    margin = 7

                    color = (
                        "#121B19"
                        if value == BLACK
                        else "#F5F3E8"
                    )

                    self.canvas.create_oval(
                        col * CELL_SIZE + margin,
                        row * CELL_SIZE + margin,
                        (col + 1) * CELL_SIZE - margin,
                        (row + 1) * CELL_SIZE - margin,
                        fill=color,
                        outline="#546B5F",
                        width=2
                    )

        self.update_info()

    # ========================================================
    # PLAYER INPUT
    # ========================================================

    def handle_click(self, event):

        if self.replay is not None:
            return

        if is_game_over(self.board):
            return

        if (
            self.mode.get() in AI_AGENTS
            and self.player == WHITE
        ):
            return

        row = event.y // CELL_SIZE
        col = event.x // CELL_SIZE

        if not in_bounds(row, col):
            return

        self.apply_move((row, col))

    # ========================================================
    # APPLY MOVE
    # ========================================================

    def apply_move(self, action):

        if self.replay is not None:
            return

        if is_game_over(self.board):
            return

        current = self.player

        legal_moves = get_valid_moves(
            self.board,
            current
        )

        if action not in legal_moves:
            return

        new_board = make_move(
            self.board,
            action[0],
            action[1],
            current
        )

        if new_board is None:
            return

        self.board = new_board

        self.history.record(
            current,
            action,
            self.board
        )

        self.player = -current

        self.after_move()

    def after_move(self):

        if is_game_over(self.board):

            self.cancel_ai_job()
            self.finish_and_save_game()

            self.draw_board()
            self.show_result()
            return

        # Xử lý PASS
        if not get_valid_moves(
            self.board,
            self.player
        ):

            skipped = self.player

            self.history.record(
                skipped,
                None,
                self.board
            )

            self.player = -skipped

            messagebox.showinfo(
                "Bỏ lượt",
                (
                    "Đen không có nước đi hợp lệ."
                    if skipped == BLACK
                    else "Trắng không có nước đi hợp lệ."
                )
            )

        self.draw_board()
        self.schedule_ai()

    # ========================================================
    # AI
    # ========================================================

    def cancel_ai_job(self):

        if self.ai_job is not None:

            self.root.after_cancel(self.ai_job)
            self.ai_job = None

    def schedule_ai(self):

        self.cancel_ai_job()

        if (
            self.mode.get() in AI_AGENTS
            and self.replay is None
            and self.player == WHITE
            and not is_game_over(self.board)
        ):

            self.ai_job = self.root.after(
                250,
                self.ai_move
            )

    def ai_move(self):

        self.ai_job = None

        if self.replay is not None:
            return

        if self.mode.get() not in AI_AGENTS:
            return

        if self.player != WHITE:
            return

        if is_game_over(self.board):
            return

        agent = AI_AGENTS[self.mode.get()]

        self.status_label.config(
            text="AI đang suy nghĩ..."
        )

        self.root.update_idletasks()

        try:
            action = agent(
                self.board,
                WHITE
            )

        except Exception as exc:

            messagebox.showerror(
                "Lỗi AI",
                str(exc)
            )

            self.mode.set("hvh")
            self.mode_label.set(MODE_LABELS["hvh"])

            self.restart_game()
            return

        if action is None:

            if not get_valid_moves(
                self.board,
                WHITE
            ):

                self.history.record(
                    WHITE,
                    None,
                    self.board
                )

                self.player = BLACK
                self.draw_board()

            else:
                messagebox.showerror(
                    "Lỗi AI",
                    "AI không trả về nước đi hợp lệ."
                )

            return

        if action not in get_valid_moves(
            self.board,
            WHITE
        ):
            messagebox.showerror(
                "Lỗi AI",
                f"AI trả về nước đi không hợp lệ: {action}"
            )
            return

        self.apply_move(action)

    # ========================================================
    # MODE
    # ========================================================

    def change_mode(self, event=None):

        selected_label = self.mode_label.get()

        selected_mode = next(
            (
                key for key, label
                in MODE_LABELS.items()
                if label == selected_label
            ),
            "hvh"
        )

        self.mode.set(selected_mode)
        self.restart_game()

    # ========================================================
    # GAME HISTORY
    # ========================================================

    def finish_and_save_game(self):

        if self.history_saved:
            return

        self.history.finish(
            get_winner(self.board)
        )

        try:
            LOG_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            # GameHistory.save() hiện lưu vào
            # game_logs/ khi chạy từ thư mục gốc.
            path = self.history.save()

        except OSError as exc:
            messagebox.showerror(
                "Lỗi lưu lịch sử",
                str(exc)
            )
            return

        self.history_saved = True
        self.last_log_path = path

    # ========================================================
    # UPDATE INFORMATION
    # ========================================================

    def update_info(self):

        black, white = count_pieces(
            self.board
        )

        self.score_label.config(
            text=f"{black:02d}  :  {white:02d}"
        )

        if self.replay is not None:

            position = (
                f"{self.replay.index}"
                f" / {len(self.replay.moves)}"
            )

            self.turn_label.config(
                text=f"Replay • {position}",
                fg=GOLD
            )

            self.replay_count_label.config(
                text=position
            )

            self.status_label.config(
                text="Đang xem lịch sử trận đấu"
            )

        
        elif is_game_over(self.board):

            if black > white:
                result_text = "Đen thắng!"
            elif white > black:
                result_text = "Trắng thắng!"
            else:
                result_text = "Trận đấu hòa!"

            self.turn_label.config(
                text=result_text,
                fg=GOLD
            )

            self.replay_count_label.config(
                text="— / —"
            )

            self.status_label.config(
                text="Đã kết thúc ván đấu"
            )

        else:

            turn = (
                "Đen"
                if self.player == BLACK
                else "Trắng"
            )

            self.turn_label.config(
                text=f"Lượt của {turn}",
                fg=ACCENT
            )

            self.replay_count_label.config(
                text="— / —"
            )

            if (
                self.mode.get() in AI_AGENTS
                and self.player == WHITE
            ):
                status = "AI đang chuẩn bị nước đi..."

            elif self.mode.get() in AI_AGENTS:
                status = "Đến lượt bạn • quân Đen"

            else:
                status = f"Đến lượt quân {turn}"

            self.status_label.config(
                text=status
            )

        
        if self.history_saved and self.last_log_path:

            self.log_label.config(
                text="Đã lưu lịch sử thành công"
            )

        else:
            self.log_label.config(
                text="Log: chưa lưu ván hiện tại"
            )

    # ========================================================
    # GAME OVER
    # ========================================================

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

        log_info = (
            f"\nĐã lưu: {self.last_log_path}"
            if self.history_saved
            else "\nChưa lưu được lịch sử"
        )

        messagebox.showinfo(
            "Kết thúc",
            (
                "Trò chơi kết thúc!\n\n"
                f"Đen: {black}\n"
                f"Trắng: {white}\n\n"
                f"{result}{log_info}"
            )
        )

    # ========================================================
    # RESTART
    # ========================================================

    def restart_game(self):

        self.cancel_ai_job()

        if self.replay is not None:
            self.close_replay()
            self.cancel_ai_job()

        self.board = init_board()
        self.player = BLACK

        self.history = GameHistory()
        self.history.start(self.board)

        self.history_saved = False
        self.last_log_path = None

        self.draw_board()

    # ========================================================
    # REPLAY
    # ========================================================

    def open_history(self):

        filename = filedialog.askopenfilename(
            title="Chọn lịch sử Othello",
            initialdir=str(LOG_DIR),
            filetypes=[
                ("JSON files", "*.json")
            ]
        )

        if not filename:
            return

        try:
            new_replay = GameReplay(
                filename
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError
        ) as exc:

            messagebox.showerror(
                "Không mở được lịch sử",
                str(exc)
            )
            return

        self.cancel_ai_job()

        if self.replay is None:
            self.saved_game = (
                deepcopy(self.board),
                self.player
            )

        self.replay = new_replay
        self.show_replay_position()

    def show_replay_position(self):

        if self.replay is None:
            return

        self.board = deepcopy(
            self.replay.boards[
                self.replay.index
            ]
        )

        if self.replay.index < len(
            self.replay.moves
        ):
            self.player = (
                self.replay.moves[
                    self.replay.index
                ]["player"]
            )

        self.previous_button.config(
            state=(
                "normal"
                if self.replay.index > 0
                else "disabled"
            )
        )

        self.next_button.config(
            state=(
                "normal"
                if self.replay.index < len(
                    self.replay.moves
                )
                else "disabled"
            )
        )

        self.close_replay_button.config(
            state="normal"
        )

        self.draw_board()

    def previous_replay(self):

        if (
            self.replay is not None
            and self.replay.index > 0
        ):

            self.replay.index -= 1
            self.show_replay_position()

    def next_replay(self):

        if (
            self.replay is not None
            and self.replay.index < len(
                self.replay.moves
            )
        ):

            self.replay.index += 1
            self.show_replay_position()

    def close_replay(self):

        if self.replay is None:
            return

        self.replay = None

        self.board, self.player = (
            self.saved_game
        )

        self.saved_game = None

        self.previous_button.config(
            state="disabled"
        )

        self.next_button.config(
            state="disabled"
        )

        self.close_replay_button.config(
            state="disabled"
        )

        self.draw_board()
        self.schedule_ai()


# ============================================================
# MAIN
# ============================================================

def enable_high_dpi():

    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)

    except Exception:
        pass


def run_gui():

    enable_high_dpi()

    root = tk.Tk()
    OthelloGUI(root)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
