from core.othello_engine import (
    init_board,
    get_valid_moves,
    make_move,
)

from agents.heuristic_agent import heuristic_agent, evaluate_board


def main():
    board = init_board()
    player = 1

    legal_moves = get_valid_moves(board, player)
    selected_move = heuristic_agent(board, player)

    print("Nuoc di hop le:", legal_moves)
    print("Heuristic chon:", selected_move)

    assert selected_move in legal_moves

    next_board = make_move(
        board,
        selected_move[0],
        selected_move[1],
        player
    )

    assert next_board is not None

    print("Diem heuristic:", evaluate_board(next_board, player))
    print("\nPASS: Heuristic Agent hoat dong!")


if __name__ == "__main__":
    main()
