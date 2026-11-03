import math

from core.othello_engine import (
    black,
    white,
    get_valid_moves,
    is_game_over
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

    
def selection(root, exploration_constant=math.sqrt(2)):
    """
    Đi từ Root xuống cây bằng UCT.

    Trả về nút được chọn để tiếp tục Expansion.
    """

    node = root

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

