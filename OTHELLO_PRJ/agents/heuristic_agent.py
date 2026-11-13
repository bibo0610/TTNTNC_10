from core.othello_engine import get_valid_moves, make_move

POSITION_WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


def evaluate_board(board, player):
    opponent = -player

    # 1. Corner Control
    corner_score = sum(
        (board[r][c] == player) -
        (board[r][c] == opponent)
        for r, c in CORNERS
    )

    # 2. Mobility
    my_moves = len(get_valid_moves(board, player))
    opp_moves = len(get_valid_moves(board, opponent))
    mobility_score = my_moves - opp_moves

    # 3. Positional Weight
    position_score = sum(
        POSITION_WEIGHTS[r][c] * board[r][c] * player
        for r in range(8)
        for c in range(8)
    )

    # 4. Disc Difference
    disc_score = sum(
        cell * player
        for row in board
        for cell in row
    )

    return (
        30 * corner_score
        + 5 * mobility_score
        + position_score
        + disc_score
    )


def heuristic_agent(board, player):
    moves = get_valid_moves(board, player)

    if not moves:
        return None

    best_move = None
    best_score = float("-inf")

    for row, col in moves:
        next_board = make_move(board, row, col, player)

        score = evaluate_board(next_board, player)

        if score > best_score:
            best_score = score
            best_move = (row, col)

    return best_move
