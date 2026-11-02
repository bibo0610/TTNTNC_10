import math
 
from core.othello_engine import (
    get_valid_moves,
    make_move,
    count_pieces,
    is_game_over,
    black,
)
from agents.heuristic_agent import evaluate_board, POSITION_WEIGHTS
 
DEFAULT_DEPTH = 4
WIN_SCORE = 100000      

# Đếm số thế cờ đã xét 
stats = {"nodes": 0}

def best_moves(board, player):
    key = lambda m: POSITION_WEIGHTS[m[0]] [m[1]]
    
    moves = get_valid_moves(board, player)
    moves.sort(key = key, reverse=True)
    return moves

def terminal_score(board, player):
    b, w = count_pieces(board)
    
    diff = b - w if player == black else w - b
    
    if is_game_over(board):
        if diff > 0:
            return WIN_SCORE + diff
        elif diff < 0:
            return -WIN_SCORE + diff
        else:
            return 0
    
    
def minimax(board, depth, alpha, beta, maximizing, ai_player, use_pruning = True):
    stats["nodes"] += 1
    #maximizing=True  -> lượt AI, chọn điểm LỚN nhất
    #maximizing=False -> lượt đối thủ, chọn điểm NHỎ nhất
    if is_game_over(board):
        return terminal_score(board, ai_player)
    elif depth == 0:
        return evaluate_board(board, ai_player)
    
    current= ai_player if maximizing else -ai_player
    moves = get_valid_moves(board, current)
    
    if not moves:
        return minimax(board, depth - 1, alpha, beta, not maximizing,
                       ai_player, use_pruning)
 
    if maximizing:
        best = -math.inf
        for row, col in moves:
            child = make_move(board, row, col, current)
            best = max(best, minimax(child, depth - 1, alpha, beta, False,
                                     ai_player, use_pruning))
            alpha = max(alpha, best)
            if use_pruning and alpha >= beta:
                break           # cắt tỉa beta
        return best
    else:
        best = math.inf
        for row, col in moves:
            child = make_move(board, row, col, current)
            best = min(best, minimax(child, depth - 1, alpha, beta, True,
                                     ai_player, use_pruning))
            beta = min(beta, best)
            if use_pruning and alpha >= beta:
                break           # cắt tỉa alpha
        return best
    
def minimax_agent(board, player, depth=DEFAULT_DEPTH, use_pruning=True):
    """Chọn nước đi tốt nhất cho `player`. Trả về (row, col), hoặc None nếu phải bỏ lượt.
    Cùng dạng với heuristic_agent(board, player) nên GUI gọi giống hệt."""
    moves = best_moves(board, player)
    if not moves:
        return None
 
    best_move = moves[0]
    best_score = -math.inf
    alpha, beta = -math.inf, math.inf
 
    for row, col in moves:
        child = make_move(board, row, col, player)
        score = minimax(child, depth - 1, alpha, beta, False, player, use_pruning)
        if score > best_score:
            best_score = score
            best_move = (row, col)
        alpha = max(alpha, best_score)
 
    return best_move