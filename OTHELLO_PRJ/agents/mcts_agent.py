import math
import random

from core.othello_engine import (
    black,
    white,
    get_valid_moves,
    is_game_over,
    make_move,
    get_winner
)


class MCTSNode:
    def __init__(
        self,
        board,
        player_to_move,
        parent=None,
        move=None
    ):
        # 1. Lưu trạng thái bàn cờ
        # Tạo bản sao để không làm thay đổi bàn cờ gốc
        self.board = [row[:] for row in board]

        # 2. Người chơi sẽ thực hiện lượt tiếp theo
        self.player_to_move = player_to_move

        # 3. Liên kết với nút cha
        self.parent = parent

        # 4. Danh sách các nút con
        self.children = []

        # 5. Nước đi tạo ra nút hiện tại
        # Root không có nước đi nên move = None
        self.move = move

        # 6. Người vừa thực hiện nước đi
        self.player_just_moved = -player_to_move

        # 7. Số lần nút được thăm
        self.visits = 0

        # 8. Tổng điểm kết quả mô phỏng
        self.wins = 0.0

        # 9. Các nước đi chưa được mở rộng
        self.untried_moves = self._get_untried_moves()

    def _get_untried_moves(self):
        """
        Lấy danh sách nước đi chưa được thử.

        Nếu người chơi hiện tại không có nước đi:
        - Đưa None vào danh sách nếu đối thủ còn nước đi.
        - Trả về danh sách rỗng nếu cả hai bên đều hết nước.
        """

        if is_game_over(self.board):
            return []

        moves = get_valid_moves(
            self.board,
            self.player_to_move
        )

        if moves:
            return moves

        # Không có nước đi: phải PASS
        return [None]

    def is_fully_expanded(self):
        """
        Kiểm tra tất cả nước đi đã được mở rộng chưa.
        """
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """
        Kiểm tra trạng thái bàn cờ đã kết thúc chưa.
        """
        return is_game_over(self.board)

    def __repr__(self):
        return (
            f"MCTSNode("
            f"player={self.player_to_move}, "
            f"move={self.move}, "
            f"visits={self.visits}, "
            f"wins={self.wins}, "
            f"children={len(self.children)}"
            f")"
        )

    
    def uct_score(self, exploration_constant=math.sqrt(2)):
        """
        Tính điểm UCT của nút hiện tại.

        exploration_constant:
            Hệ số cân bằng Exploration và Exploitation.
        """

        # Nút chưa được thăm: ưu tiên khám phá
        if self.visits == 0:
            return float("inf")

        # Nút gốc không có cha nên không tính UCT
        if self.parent is None:
            raise ValueError("Root node does not have a UCT score.")

        # Thành phần khai thác
        exploitation = self.wins / self.visits

        # Thành phần khám phá
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    
    def select_child(self, exploration_constant=math.sqrt(2)):
        """
        Chọn nút con có điểm UCT cao nhất.
        """

        if not self.children:
            raise ValueError("Node has no children.")

        return max(
            self.children,
            key=lambda child: child.uct_score(
                exploration_constant
            )
        )

    def expand(self):
        """
        Mở rộng cây bằng cách tạo một nút con mới.

        Trả về:
            Nút con vừa được tạo.
        """

        # Nếu không còn nước đi chưa mở rộng
        if not self.untried_moves:
            return None

        # 1. Chọn ngẫu nhiên một nước đi chưa được thử
        move = random.choice(self.untried_moves)

        # 2. Xóa nước đi khỏi danh sách chưa mở rộng
        self.untried_moves.remove(move)

        # 3. Tạo trạng thái bàn cờ tiếp theo
        if move is None:
            # Không có nước đi hợp lệ: PASS
            next_board = self.board
        else:
            row, col = move

            next_board = make_move(
                self.board,
                row,
                col,
                self.player_to_move
            )

            if next_board is None:
                raise ValueError("Invalid move during expansion.")

        # 4. Chuyển lượt cho người chơi tiếp theo
        next_player = -self.player_to_move

        # 5. Tạo nút con
        child = MCTSNode(
            board=next_board,
            player_to_move=next_player,
            parent=self,
            move=move
        )

        # 6. Thêm nút con vào danh sách children
        self.children.append(child)

        return child

    
    def selection(seft, exploration_constant=math.sqrt(2)):
        """
        Đi từ Root xuống cây bằng UCT.

        Trả về nút được chọn để tiếp tục Expansion.
        """

        node = seft

        while not node.is_terminal():

            # Nếu còn nước đi chưa mở rộng thì dừng
            if not node.is_fully_expanded():
                break

            # Nếu không có nút con thì dừng
            if not node.children:
                break

            # Chọn nút con có UCT cao nhất
            node = node.select_child(exploration_constant)

        return node

def simulation(node):
    """
    Mô phỏng một ván Othello bằng cách chọn nước đi ngẫu nhiên.

    Tham số:
        node: Nút bắt đầu mô phỏng.

    Trả về:
        black: Đen thắng.
        white: Trắng thắng.
        0: Hai bên hòa.
    """

    # 1. Sao chép bàn cờ từ nút hiện tại
    board = [row[:] for row in node.board]

    # 2. Xác định người chơi sẽ đi tiếp
    player = node.player_to_move

    # 3. Mô phỏng cho đến khi ván cờ kết thúc
    while not is_game_over(board):

        # Lấy các nước đi hợp lệ
        moves = get_valid_moves(board, player)

        if moves:
            # 4. Chọn ngẫu nhiên một nước đi
            move = random.choice(moves)

            row, col = move

            # 5. Thực hiện nước đi
            board = make_move(
                board,
                row,
                col,
                player
            )

        # 6. Chuyển lượt cho người chơi tiếp theo
        # Kể cả khi không có nước đi (PASS)
        player = -player

    # 7. Xác định người chiến thắng
    winner = get_winner(board)

    return winner

def backpropagation(node, winner):
    """
    Cập nhật kết quả mô phỏng ngược từ nút hiện tại về Root.

    Tham số:
        node: Nút bắt đầu cập nhật.
        winner: Người chiến thắng (black, white hoặc 0).
    """

    # Bắt đầu từ nút vừa thực hiện Simulation
    current = node

    # Đi ngược từ nút hiện tại về Root
    while current is not None:

        # Tăng số lần nút được thăm
        current.visits += 1

        # Nếu người vừa đi thắng
        if winner == current.player_just_moved:
            current.wins += 1.0

        # Nếu ván đấu hòa
        elif winner == 0:
            current.wins += 0.5

        # Nếu người vừa đi thua thì không cộng điểm
        # current.wins giữ nguyên

        # Chuyển lên nút cha
        current = current.parent

def mcts_agent(board, player, simulations=1000):
    """
    Agent sử dụng Monte Carlo Tree Search (MCTS)
    để lựa chọn nước đi.

    Tham số:
        board: Bàn cờ hiện tại.
        player: Người chơi hiện tại (black hoặc white).
        simulations: Số lần mô phỏng.

    Trả về:
        (row, col): Nước đi được lựa chọn.
        None: Nếu người chơi không có nước đi hợp lệ.
    """

    if simulations <= 0:
        raise ValueError("simulations must be greater than 0")
    
    # 1. Lấy các nước đi hợp lệ
    valid_moves = get_valid_moves(board, player)

    # Không có nước đi thì PASS
    if not valid_moves:
        return None

    # 2. Tạo nút gốc
    root = MCTSNode(
        board=board,
        player_to_move=player
    )

    # 3. Thực hiện MCTS nhiều lần
    for _ in range(simulations):

        # BƯỚC 1: SELECTION
        node = root.selection()

        # BƯỚC 2: EXPANSION
        if not node.is_terminal():
            if not node.is_fully_expanded():
                node = node.expand()

        # BƯỚC 3: SIMULATION
        winner = simulation(node)

        # BƯỚC 4: BACKPROPAGATION
        backpropagation(node, winner)

    # 4. Chọn nút con được thăm nhiều nhất
    best_child = max(
        root.children,
        key=lambda child: child.visits
    )

    # 5. Trả về nước đi tốt nhất
    return best_child.move

def agent(board, player):
    return mcts_agent(board, player, simulations=1000)