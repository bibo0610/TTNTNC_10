import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from copy import deepcopy
from pathlib import Path
from functools import partial
from core.othello_engine import random_agent

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
from agents.cnn_agent import cnn_agent, cnn_minimax_agent, cnn_weighted_agent, cnn_residual_agent
from agents.mcts_agent import (
    agent as mcts_agent,
    mcts_agent as mcts_with_simulations
)


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

CELL_SIZE = 60

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "game_logs"

MODE_LABELS = {
    "hvh": "Human vs Human",
    "hva": "Human vs Heuristic",
    "hvm": "Human vs Minimax",
    "hvcnn": "Human vs CNN (Heuristic)",
    "hvcnnminimax": "Human vs CNN (Minimax)",
    "hvcnnweighted": "Human vs CNN (V4.1 Weighted)",
    "hvcnnresidual": "Human vs CNN (V5 Residual)",
    "hvmcts": "Human vs MCTS/UCT",
    "aiai": "AI vs AI",
}

AI_AGENTS = {
    "hva": heuristic_agent,
    "hvm": minimax_agent,
    "hvcnn": cnn_agent,
    "hvcnnminimax": cnn_minimax_agent,
    "hvcnnweighted": cnn_weighted_agent,
    "hvcnnresidual": cnn_residual_agent,
    "hvmcts": mcts_agent,
}

DEMO_AGENTS = {
    "Random": random_agent,
    "Heuristic": heuristic_agent,
    "Minimax D2": partial(minimax_agent, depth=2),
    "MCTS 64": partial(mcts_with_simulations, simulations=64),
    "CNN V1": cnn_agent,
    "CNN V2": cnn_minimax_agent,
    "CNN V4.1": cnn_weighted_agent,
    "CNN V5": cnn_residual_agent,
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
        self.root.geometry("1120x820")
        self.root.minsize(980, 720)

        self.board = init_board()
        self.player = BLACK

        self.mode = tk.StringVar(value="hvh")
        self.mode_label = tk.StringVar(
            value=MODE_LABELS["hvh"]
        )

        self.ai_job = None

        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False

        self.black_ai_name = tk.StringVar(
            value="Heuristic"
        )

        self.white_ai_name = tk.StringVar(
            value="Minimax D2"
        )

        # Thời gian nghỉ giữa hai nước đi (ms)
        self.demo_delay = 700

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
            padding=5,
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
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=30, pady=(12, 0))

        title_frame = tk.Frame(header, bg=BG)
        title_frame.pack(side="left")
        tk.Label(title_frame, text="OTHELLO", font=("Segoe UI", 22, "bold"),
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(title_frame, text="  AI", font=("Segoe UI", 22, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(header, text="STRATEGY ARENA  /  8 × 8",
                 font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(side="right", pady=10)

    # ========================================================
    # LAYOUT: compact cards fit above the pinned footer
    # ========================================================

    def create_layout(self):
        # Footer owns its height; controls are intentionally compact enough
        # to fit above it at the minimum supported window size.
        self.status_sidebar = tk.Frame(self.root, bg=BG)
        self.status_sidebar.pack(side="bottom", fill="x", padx=26, pady=(0, 9))
        self.create_status_card()

        self.workspace = tk.Frame(self.root, bg=BG)
        self.workspace.pack(side="top", fill="both", expand=True, padx=18, pady=(8, 5))

        content = tk.Frame(self.workspace, bg=BG)
        # Top-aligned rather than vertically centered: no huge blank band.
        content.pack(anchor="n")

        self.board_panel = tk.Frame(content, bg=PANEL, padx=17, pady=14)
        self.board_panel.grid(row=0, column=0, padx=(0, 13), sticky="n")

        self.sidebar = tk.Frame(content, bg=BG, width=322)
        self.sidebar.grid(row=0, column=1, sticky="n")
        self.sidebar.grid_propagate(False)

        self.create_board_area()
        self.create_mode_card()
        self.create_ai_vs_ai_card()
        self.create_replay_card()
        self.refresh_mode_panels()

    def refresh_mode_panels(self):
        """Only show the AI-versus-AI controller in AI-versus-AI mode."""
        demo_mode = self.mode.get() == "aiai"
        if demo_mode:
            if not self.ai_demo_card.winfo_manager():
                self.ai_demo_card.pack(fill="x", pady=(0, 7), before=self.replay_card)
            self.restart_button.config(text="ĐẶT LẠI BÀN CỜ")
            self.mode_hint.config(text="Chọn hai AI bên dưới để trình diễn.")
        else:
            if self.ai_demo_card.winfo_manager():
                self.ai_demo_card.pack_forget()
            self.restart_button.config(text="BẮT ĐẦU VÁN MỚI")
            if self.mode.get() == "hvh":
                self.mode_hint.config(text="Hai người chơi thay phiên nhau.")
            else:
                self.mode_hint.config(text="Bạn cầm Đen khi đấu với AI.")

    # ========================================================
    # REUSABLE UI
    # ========================================================

    def create_card(self, parent=None):
        if parent is None:
            parent = self.sidebar
        card = tk.Frame(parent, bg=PANEL, width=322, padx=14, pady=9)
        card.pack(fill="x", pady=(0, 7))
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
        score_frame = tk.Frame(self.board_panel, bg=PANEL)
        score_frame.pack(fill="x", pady=(0, 8))

        tk.Label(score_frame, text="BLACK", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")
        self.score_label = tk.Label(score_frame, text="02  :  02",
                                    font=("Segoe UI", 21, "bold"),
                                    bg=PANEL, fg=TEXT)
        self.score_label.pack(side="left", expand=True)
        tk.Label(score_frame, text="WHITE", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="right")

        board_border = tk.Frame(self.board_panel, bg="#123D2C", padx=3, pady=3)
        board_border.pack()
        self.canvas = tk.Canvas(board_border, width=SIZE * CELL_SIZE,
                                height=SIZE * CELL_SIZE,
                                bg=BOARD_LIGHT, highlightthickness=0,
                                cursor="hand2")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.turn_label = tk.Label(self.board_panel, text="Lượt của Đen",
                                   font=("Segoe UI", 12, "bold"),
                                   bg=PANEL, fg=ACCENT)
        self.turn_label.pack(pady=(10, 1))

    # ========================================================
    # MODE CARD
    # ========================================================

    def create_mode_card(self):
        self.mode_card = self.create_card()
        card = self.mode_card
        tk.Label(card, text="CHẾ ĐỘ CHƠI", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w")
        self.mode_combo = ttk.Combobox(card, textvariable=self.mode_label,
                                       values=list(MODE_LABELS.values()),
                                       state="readonly", style="Modern.TCombobox",
                                       width=29, font=("Segoe UI", 10))
        self.mode_combo.pack(fill="x", pady=(6, 7))
        self.mode_combo.bind("<<ComboboxSelected>>", self.change_mode)

        self.restart_button = self.create_button(card, "BẮT ĐẦU VÁN MỚI",
                                                  self.restart_game, primary=True)
        self.restart_button.pack(fill="x", ipady=5)
        self.mode_hint = tk.Label(card, text="Hai người chơi thay phiên nhau.",
                                  font=("Segoe UI", 9), bg=PANEL, fg=MUTED,
                                  anchor="w")
        self.mode_hint.pack(fill="x", pady=(6, 0))

    def create_ai_vs_ai_card(self):
        self.ai_demo_card = self.create_card()
        card = self.ai_demo_card
        tk.Label(card, text="DEMO AI VS AI", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", pady=(0, 2))
        tk.Label(card, text="AI QUÂN ĐEN", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w", pady=(4, 2))
        self.black_ai_combo = ttk.Combobox(card, textvariable=self.black_ai_name,
                                           values=list(DEMO_AGENTS.keys()),
                                           state="readonly", style="Modern.TCombobox", width=29)
        self.black_ai_combo.pack(fill="x")
        tk.Label(card, text="AI QUÂN TRẮNG", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w", pady=(5, 2))
        self.white_ai_combo = ttk.Combobox(card, textvariable=self.white_ai_name,
                                           values=list(DEMO_AGENTS.keys()),
                                           state="readonly", style="Modern.TCombobox", width=29)
        self.white_ai_combo.pack(fill="x")

        self.demo_start_button = self.create_button(card, "BẮT ĐẦU AI VS AI",
                                                     self.start_ai_vs_ai, primary=True)
        self.demo_start_button.pack(fill="x", ipady=4, pady=(7, 4))
        self.demo_pause_button = self.create_button(card, "TẠM DỪNG",
                                                     self.toggle_ai_vs_ai_pause,
                                                     state="disabled")
        self.demo_pause_button.pack(fill="x", ipady=3)

    # ========================================================
    # REPLAY CARD
    # ========================================================

    def create_replay_card(self):
        self.replay_card = self.create_card()
        card = self.replay_card
        tk.Label(card, text="LỊCH SỬ TRẬN ĐẤU", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w")
        self.open_button = self.create_button(card, "Mở file lịch sử (JSON)",
                                               self.open_history)
        self.open_button.pack(fill="x", pady=(7, 6), ipady=4)
        controls = tk.Frame(card, bg=PANEL)
        controls.pack(fill="x")
        self.previous_button = self.create_button(controls, "◀",
                                                   self.previous_replay,
                                                   state="disabled")
        self.previous_button.pack(side="left", fill="x", expand=True, ipady=4)
        self.replay_count_label = tk.Label(controls, text="— / —", bg=PANEL,
                                           fg=TEXT, width=9, font=("Segoe UI", 10))
        self.replay_count_label.pack(side="left")
        self.next_button = self.create_button(controls, "▶", self.next_replay,
                                               state="disabled")
        self.next_button.pack(side="left", fill="x", expand=True, ipady=4)
        self.close_replay_button = self.create_button(card, "Đóng Replay",
                                                      self.close_replay,
                                                      state="disabled")
        self.close_replay_button.pack(fill="x", pady=(6, 0), ipady=4)

    # ========================================================
    # STATUS CARD
    # ========================================================

    def create_status_card(self):
        card = tk.Frame(self.status_sidebar, bg=PANEL, padx=16, pady=7)
        card.pack(fill="x")
        tk.Label(card, text="TRẠNG THÁI", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 16))

        status_container = tk.Frame(card, bg=PANEL, height=27)
        status_container.pack(side="left", fill="x", expand=True)
        status_container.pack_propagate(False)
        self.status_label = tk.Label(status_container, text="Sẵn sàng chơi",
                                     font=("Segoe UI", 10), bg=PANEL, fg=TEXT,
                                     anchor="w", justify="left")
        self.status_label.pack(fill="both", expand=True)

        log_container = tk.Frame(card, bg=PANEL, width=245, height=27)
        log_container.pack(side="right", padx=(14, 0))
        log_container.pack_propagate(False)
        self.log_label = tk.Label(log_container, text="Log: chưa lưu ván hiện tại",
                                  font=("Segoe UI", 9), bg=PANEL, fg=MUTED,
                                  anchor="e", justify="right")
        self.log_label.pack(fill="both", expand=True)

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
                self.mode.get() == "aiai"
                or (
                    self.mode.get() in AI_AGENTS
                    and self.player == WHITE
                )
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
        
        if self.mode.get() == "aiai":
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
            self.ai_vs_ai_running = False
            self.black_ai_combo.config(state="readonly")
            self.white_ai_combo.config(state="readonly")

            self.demo_pause_button.config(
                state="disabled"
            )

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

            if self.mode.get() != "aiai":
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

        if self.replay is not None:
            return

        if is_game_over(self.board):
            return

        mode = self.mode.get()

        # AI vs AI: tự động chạy cho cả Đen và Trắng
        if mode == "aiai":
            if (
                not self.ai_vs_ai_running
                or self.ai_vs_ai_paused
            ):
                return

            self.ai_job = self.root.after(
                self.demo_delay,
                self.ai_move
            )
            return

        # Các chế độ Human vs AI hiện tại:
        # chỉ gọi AI khi đến lượt Trắng.
        if mode in AI_AGENTS and self.player == WHITE:
            self.ai_job = self.root.after(
                250,
                self.ai_move
            )

    
    def ai_move(self):
        self.ai_job = None

        if self.replay is not None:
            return

        if is_game_over(self.board):
            return

        mode = self.mode.get()

        if mode == "aiai":
            if (
                not self.ai_vs_ai_running
                or self.ai_vs_ai_paused
            ):
                return

            if self.player == BLACK:
                agent_name = self.black_ai_name.get()
            else:
                agent_name = self.white_ai_name.get()

            agent = DEMO_AGENTS[agent_name]
            current_player = self.player

        elif mode in AI_AGENTS:
            if self.player != WHITE:
                return

            agent_name = MODE_LABELS[mode]
            agent = AI_AGENTS[mode]
            current_player = WHITE

        else:
            return

        colour = (
            "Đen" if current_player == BLACK else "Trắng"
        )

        self.status_label.config(
            text=f"{colour}: {agent_name} đang suy nghĩ..."
        )
        self.root.update_idletasks()

        # Trường hợp hết nước hợp lệ: PASS
        legal_moves = get_valid_moves(
            self.board,
            current_player
        )

        if not legal_moves:
            self.history.record(
                current_player,
                None,
                self.board
            )

            self.player = -current_player
            self.draw_board()
            self.schedule_ai()
            return

        try:
            # Truyền bản sao để agent không làm thay đổi
            # trạng thái bàn cờ chính.
            action = agent(
                deepcopy(self.board),
                current_player
            )

        except Exception as exc:
            self.cancel_ai_job()
            self.ai_vs_ai_running = False
            self.black_ai_combo.config(state="readonly")
            self.white_ai_combo.config(state="readonly")

            self.demo_pause_button.config(
                state="disabled"
            )

            messagebox.showerror(
                "Lỗi AI",
                f"{agent_name}: {exc}"
            )
            return

        if action not in legal_moves:
            self.cancel_ai_job()
            self.ai_vs_ai_running = False
            self.black_ai_combo.config(state="readonly")
            self.white_ai_combo.config(state="readonly")

            self.demo_pause_button.config(
                state="disabled"
            )

            messagebox.showerror(
                "Lỗi AI",
                f"{agent_name} trả về nước đi "
                f"không hợp lệ: {action}"
            )
            return

        self.apply_move(action)

    # ========================================================
    # MODE
    # ========================================================

    def start_ai_vs_ai(self):
        # Chuyển sang chế độ AI vs AI
        self.mode.set("aiai")
        self.mode_label.set(MODE_LABELS["aiai"])
        self.refresh_mode_panels()

        # Khởi tạo lại bàn cờ và lịch sử
        self.restart_game()

        self.ai_vs_ai_running = True
        self.ai_vs_ai_paused = False
        self.black_ai_combo.config(state="disabled")
        self.white_ai_combo.config(state="disabled")

        self.demo_pause_button.config(
            text="TẠM DỪNG",
            state="normal"
        )

        self.draw_board()
        self.schedule_ai()


    def toggle_ai_vs_ai_pause(self):
        if self.mode.get() != "aiai":
            return

        if not self.ai_vs_ai_running:
            return

        self.ai_vs_ai_paused = not self.ai_vs_ai_paused

        if self.ai_vs_ai_paused:
            self.cancel_ai_job()

            self.demo_pause_button.config(
                text="TIẾP TỤC"
            )
            self.status_label.config(
                text="Demo đã tạm dừng"
            )
        else:
            self.demo_pause_button.config(
                text="TẠM DỪNG"
            )
            self.schedule_ai()
        
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
        self.refresh_mode_panels()
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

            if self.mode.get() == "aiai":
                if self.ai_vs_ai_paused:
                    status = "Demo đã tạm dừng"
                else:
                    ai_name = (
                        self.black_ai_name.get()
                        if self.player == BLACK
                        else self.white_ai_name.get()
                    )

                    status = f"Đang đến lượt {ai_name}"

            elif (
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

        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False
        self.black_ai_combo.config(state="readonly")
        self.white_ai_combo.config(state="readonly")

        self.demo_pause_button.config(
            text="TẠM DỪNG",
            state="disabled"
        )

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

        if self.mode.get() == "aiai":
            self.ai_vs_ai_paused = True

            self.demo_pause_button.config(
                text="TIẾP TỤC"
            )

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
