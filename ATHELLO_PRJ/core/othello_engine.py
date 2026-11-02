import random

size = 8

empty = 0
black = 1
white = -1

symbols = {empty: ".", black: "●", white: "○"}

def init_board():
    board = [[empty] * size for _ in range(size)]

    mid = size // 2 # chia kích thước bàn cờ thành 4x4
    board[mid-1][mid-1] = white #d4
    board[mid][mid] = white # e5
    board[mid-1][mid] = black #e4
    board[mid][mid-1] = black # d5
    
    return board


def print_board(board):
    print("  "+ " ".join("abcdefgh"[:size]))
    for r in range(size):
        row = " ".join(symbols[board[r][c]] for c in range(size))
        print(f"{r+1} {row}")
        
#hàm đếm số lượng quân cờ trên bàn cờ
def count_pieces(board):
    count_black = sum(row.count(black) for row in board)
    count_white = sum(row.count(white) for row in board)
    
    return count_black, count_white

#Hướng di chuyển của cờ tại ô hiện tại (row, col) theo 8 hướng
directions= [(-1, -1), (-1, 0), (-1, 1), 
             (0, -1),            (0, 1), 
             (1, -1),   (1, 0),  (1, 1)]

#hàm này dùng để kiểm tra xem một ô có nằm trong phạm vi bàn cờ hay không
def in_bounds(row, col):
    return 0 <= row < size and 0 <= col < size

def get_flips(board, row, col, player):
    if board[row][col] != empty:
        return []  # nếu ô này đã có cờ, không thể đặt
    
    if player == black:
        opponent = white
    else:
        opponent = black
        
    flips=[]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        temp_flips=[]
        while in_bounds(r, c) and board[r][c] == opponent:
            temp_flips.append((r, c))
            r += dr
            c += dc
        if in_bounds(r, c) and board[r][c] == player:
            flips.extend(temp_flips)
    return flips

def get_valid_moves(board, player):
    moves = []
    for r in range(size):
        for c in range(size):
            if get_flips(board, r, c, player):
                moves.append((r, c))
    return moves

def to_notation(row, col):
    """Đổi (r, c) sang ký hiệu cờ, ví dụ (2, 3) -> 'd3'."""
    return f"{'abcdefgh'[col]}{row + 1}"

def print_valid_moves(board, player):
    moves = get_valid_moves(board, player)
    move_set = set(moves)
    print("  "+ " ".join("abcdefgh"[:size]))
    for row in range(size):
        cell = []
        for col in range(size):
            if(row, col) in move_set:
                cell.append("*")
            else:
                cell.append(symbols[board[row][col]])
        print(f"{row+1} " + " ".join(cell))
        
    if moves:
        print("Moves:", ", ".join(to_notation(r, c) for r, c in moves))
    else:
        print("No valid moves. Next")
        
def make_move(board, row, col, player):
    flips = get_flips(board, row, col, player)
    if not flips:
        return None  # move không hợp lệ
    
    #tạo bàn cờ mới để lưu các nước đi 
    new_board=[line[:] for line in board]
    new_board[row][col] =player
    for r, c in flips:
        new_board[r][c] = player
    return new_board    

def is_game_over(board):
    return not get_valid_moves(board, black) and not get_valid_moves(board, white)

def get_winner(board):
    b, w = count_pieces(board)
    if b > w:
        return black
    elif w > b:
        return white
    else: return empty # hòa
    
def random_agent(board, player):
    return random.choice(get_valid_moves(board, player))

def play_game(black_agent, white_agent, verbose= True):
    board = init_board()
    player = black
    agents = {black: black_agent, white: white_agent}
    names = {black: "Black", white: "White"}
    
    while not is_game_over(board):
        moves = get_valid_moves(board, player)
        
        if not moves: 
            if verbose: #hiển thị chi tiết quá trình chạy khi verbose = True
                print(f"{names[player]} has no valid moves. Skipping turn.")
            player = -player
            continue
        
        if verbose:
            print_valid_moves(board, player)
            
        r, c = agents[player](board, player)
        board = make_move(board, r, c, player)
        
        if(verbose):
            print(f"Black: %d, White: %d" % count_pieces(board))
            
        player = -player
        
    winner = get_winner(board)
    if verbose: 
        print("Game finished!")
        print_board(board)
        b, w = count_pieces(board)
        print(f"Black: {b} , White: {w}")
        print("Tie!" if winner == empty else f"{names[winner]} wins!")
    return winner, board
            
            
def parse_move(text):
    """Đổi chuỗi người chơi nhập (vd 'd3') thành (r, c).
    Báo lỗi ValueError nếu sai định dạng hoặc ngoài bàn 8x8."""
    text = text.strip().lower()
    if len(text) < 2:
        raise ValueError("Nhập theo dạng cột + hàng, ví dụ: d3")
 
    col_char, row_text = text[0], text[1:]
    if col_char not in "abcdefgh"[:size]:
        raise ValueError(f"Cột '{col_char}' không hợp lệ, chỉ từ a đến {'abcdefgh'[size - 1]}")
    if not row_text.isdigit():
        raise ValueError(f"Hàng '{row_text}' phải là số")
 
    r = int(row_text) - 1
    c = ord(col_char) - ord("a")
    if not in_bounds(r, c):
        raise ValueError(f"Hàng {row_text} nằm ngoài bàn cờ, chỉ từ 1 đến {size}")
    return r, c
 
 
def human_agent(board, player):
    """Người chơi nhập nước đi từ terminal. Hỏi lại cho đến khi nhập đúng."""
    valid = get_valid_moves(board, player)
    while True:
        text = input("Nhập nước đi (vd d3, 'q' để thoát): ")
        if text.strip().lower() == "q":
            raise SystemExit("Đã thoát game.")
        try:
            r, c = parse_move(text)
        except ValueError as err:
            print("Lỗi:", err)
            continue
        if (r, c) not in valid:
            print(f"Lỗi: {to_notation(r, c)} không phải nước hợp lệ. "
                  f"Chọn một trong: {', '.join(to_notation(*m) for m in valid)}")
            continue
        return r, c