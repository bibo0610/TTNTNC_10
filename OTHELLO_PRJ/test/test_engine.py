import numpy as np

from core.othello_engine import (
    init_board,
    get_valid_moves,
    make_move,
    is_game_over,
    get_winner,
)


def test_initial_board():
    board = init_board()

    assert len(board) == 8
    assert all(len(row) == 8 for row in board)

    black = sum(row.count(1) for row in board)
    white = sum(row.count(-1) for row in board)
    empty = sum(row.count(0) for row in board)

    assert black == 2
    assert white == 2
    assert empty == 60

    print("PASS: Khởi tạo bàn cờ")


def test_initial_valid_moves():
    board = init_board()

    moves = get_valid_moves(board, 1)

    assert len(moves) == 4
    assert set(moves) == {
        (2, 3), (3, 2), (4, 5), (5, 4)
    }

    print("PASS: Nước đi hợp lệ ban đầu")



def test_make_move():
    board = init_board()

    new_board = make_move(board, 2, 3, 1)

    assert new_board is not None

    black = sum(row.count(1) for row in new_board)
    white = sum(row.count(-1) for row in new_board)

    assert black == 4
    assert white == 1

    print("PASS: Đặt quân và lật quân")


def test_invalid_move():
    board = init_board()

    result = make_move(board, 0, 0, 1)

    assert result is None

    print("PASS: Từ chối nước đi không hợp lệ")


def test_game_not_over():
    board = init_board()

    assert not is_game_over(board)

    print("PASS: Kiểm tra trạng thái chưa kết thúc")


def test_pass():
    # Đen không có nước hợp lệ, Trắng vẫn có thể đi.
    board = [[-1] * 8 for _ in range(8)]
    board[0][0] = 0
    board[0][1] = 1

    assert get_valid_moves(board, 1) == []
    assert (0, 0) in get_valid_moves(board, -1)
    assert not is_game_over(board)

    print("PASS: Xử lý trường hợp bỏ lượt")


def test_multi_direction_flipping():
    # Đặt quân đen ở (3,3) để lật 8 hướng.
    board = [[0] * 8 for _ in range(8)]

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for dr, dc in directions:
        board[3 + dr][3 + dc] = -1
        board[3 + 2*dr][3 + 2*dc] = 1

    new_board = make_move(board, 3, 3, 1)

    assert new_board is not None
    assert new_board[3][3] == 1

    for dr, dc in directions:
        assert new_board[3 + dr][3 + dc] == 1

    print("PASS: Lật quân theo 8 hướng")


def test_game_over():
    board = [[1] * 8 for _ in range(8)]

    assert is_game_over(board)
    assert get_winner(board) == 1

    print("PASS: Kết thúc và xác định người thắng")



if __name__ == "__main__":
    test_initial_board()
    test_initial_valid_moves()
    test_make_move()
    test_invalid_move()
    test_game_not_over()

    test_pass()
    test_multi_direction_flipping()
    test_game_over()

    print("\nTẤT CẢ 8 BÀI KIỂM THỬ ĐỀU PASS!")