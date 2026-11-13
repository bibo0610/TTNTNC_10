from collections import Counter
from scripts.evaluate_cnn import make_opening, play_from, evaluate
from core.othello_engine import init_board, get_valid_moves, black, white


def first_legal(board, player):
    return get_valid_moves(board, player)[0]


def test_openings_reproduce():
    assert make_opening(42, 6) == make_opening(42, 6)


def test_legal_moves_and_paired_colours():
    by_colour, overall, n = evaluate(first_legal, first_legal, 4, 42, 6)
    assert sum(by_colour[black].values()) == 2
    assert sum(by_colour[white].values()) == 2
    assert sum(overall.values()) == 4
    assert 1 <= n <= 2


def test_illegal_move_raises():
    import pytest
    with pytest.raises(ValueError, match='illegal move'):
        play_from(init_board(), black, lambda b,p: (-1,-1), first_legal)
