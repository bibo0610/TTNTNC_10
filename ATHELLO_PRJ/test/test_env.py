from core.othello_env import OthelloEnv

env = OthelloEnv()

print("=== BAN CO BAN DAU ===")
env.render()

print("\nNguoi choi:", env.current_player)
print("Nuoc di hop le:", env.legal_actions())

assert len(env.legal_actions()) == 4

# Thử nước đi đầu tiên
move = env.legal_actions()[0]
board, reward, done = env.step(move)

print("\n=== SAU NUOC DI DAU TIEN ===")
env.render()

assert env.current_player == -1
assert not done

# Kiểm tra clone độc lập
copy_env = env.clone()
assert copy_env.board == env.board

copy_env.board[0][0] = 99
assert env.board[0][0] != 99

print("\nPASS: OthelloEnv hoạt động!")
