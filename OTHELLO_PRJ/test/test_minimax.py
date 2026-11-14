from core.othello_engine import (
    init_board,
    get_valid_moves,
    make_move,
)

from agents.minimax_agent import minimax_agent, stats


def main():
    board = init_board()
    player = 1  # Quan Den

    legal_moves = get_valid_moves(board, player)

    # Reset bo dem so node
    stats["nodes"] = 0

    # Chay Minimax voi Alpha-Beta Pruning
    selected_move = minimax_agent(
        board,
        player,
        depth=3,
        use_pruning=True
    )

    print("Nuoc di hop le:", legal_moves)
    print("Minimax chon:", selected_move)
    print("So node da duyet:", stats["nodes"])

    # Kiem tra nuoc di duoc chon la hop le
    assert selected_move in legal_moves

    # Kiem tra trang thai ban co sau nuoc di
    next_board = make_move(
        board,
        selected_move[0],
        selected_move[1],
        player
    )

    assert next_board is not None

    print("\nPASS: Minimax Agent hoat dong!")


if __name__ == "__main__":
    main()
