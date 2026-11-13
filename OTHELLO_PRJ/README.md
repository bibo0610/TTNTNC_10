# Othello 8×8 – Multi-level Game AI

Đồ án môn Trí tuệ nhân tạo nâng cao.

## 1. Giới thiệu

Xây dựng và đánh giá hệ thống Game AI đa cấp độ cho trò chơi Othello 8×8, tích hợp các phương pháp Heuristic, Minimax/Alpha-Beta, Monte Carlo Tree Search (MCTS) và Neural Network.

Hệ thống hỗ trợ chơi với AI, lưu lịch sử trận đấu, phát lại và hướng đến phân tích các yếu tố chiến thuật dẫn đến thắng hoặc thua.

## 2. Chức năng đã triển khai

- Game Engine Othello 8×8.
- Random Agent và Heuristic Agent.
- Human vs Human và Human vs Heuristic.
- Tự động lưu lịch sử trận đấu dưới dạng JSON.
- Game Replay với các thao tác Previous và Next.
- Kiểm thử Game Engine bằng pytest.

## 3. Công nghệ

- Python 3
- Tkinter
- pytest (dùng cho kiểm thử tự động)

## 4. Cấu trúc project

```text
ATHELLO_PRJ/
├── agents/          # Các tác tử AI
├── core/            # Game Engine và Environment
├── gui/             # Giao diện
├── history/         # Game History và Replay
├── test/            # Kiểm thử
├── notebooks/       # Notebook thử nghiệm
├── game_logs/       # Lịch sử trận đấu
├── main.py          # File chạy chính
└── README.md
```

## 5. Hướng dẫn chạy

Mở terminal tại thư mục `ATHELLO_PRJ`.

Chạy ứng dụng:

```bash
python main.py
```

Chạy kiểm thử tự động:

```bash
python -m pytest test/
```

Chạy kiểm thử Replay thủ công:

```bash
python -m test.test_replay
```

## 6. Kế hoạch phát triển

- AI cấp 3: Minimax/Alpha-Beta.
- AI cấp 4: MCTS/UCT.
- AI cấp 5: Neural Network.
- Heuristic thích nghi theo giai đoạn trận đấu.
- Post-game Analysis.
- Automated Tournament và đánh giá hiệu năng.

## 7. Lưu ý

Các file JSON trong `game_logs` được tạo tự động và không được đưa lên GitHub theo mặc định.

Các thành viên cần thống nhất giao diện Game Engine API khi triển khai tác tử AI mới.