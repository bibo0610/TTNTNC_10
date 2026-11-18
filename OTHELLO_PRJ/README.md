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
- AI sử dụng thuật toán Minimax kết hợp Alpha-Beta Pruning.
- AI sử dụng Monte Carlo Tree Search (MCTS/UCT).
- Neural Network (CNN) với các phiên bản huấn luyện từ V1 đến V5.
- Hỗ trợ nhiều chế độ Human vs AI, bao gồm Heuristic, Minimax, MCTS và CNN.
- Huấn luyện, đánh giá và so sánh các mô hình CNN bằng nhiều đối thủ và bộ seed khác nhau.

## 3. Công nghệ

- Python 3
- Tkinter
- pytest (dùng cho kiểm thử tự động)
- NumPy (xử lý dữ liệu trạng thái bàn cờ và dataset).
- PyTorch (xây dựng, huấn luyện và suy luận mô hình CNN).

## 4. Cấu trúc project

```text
OTHELLO_PRJ/
├── agents/          # Các tác tử AI
├── core/            # Game Engine và Environment
├── gui/             # Giao diện
├── history/         # Game History và Replay
├── test/            # Kiểm thử
├── notebooks/       # Notebook thử nghiệm
├── game_logs/       # Lịch sử trận đấu
├── data/            # Dataset huấn luyện CNN
├── checkpoints/     # Các mô hình CNN đã huấn luyện
├── scripts/         # Script sinh dữ liệu, huấn luyện và đánh giá
├── results/         # Kết quả thực nghiệm
├── main.py          # File chạy chính
├── README.md
├── CNN_GUIDE.md
├── EXPERIMENT_RESULTS_CNN.md  # Kết quả thực nghiệm CNN
└── requirements-cnn.txt
```

## 5. Hướng dẫn chạy

Mở terminal tại thư mục `OTHELLO_PRJ`.

Cài đặt các thư viện cần thiết:

```powershell
python -m pip install -r requirements-cnn.txt
```

Chạy ứng dụng:

```powershell
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

- Heuristic thích nghi theo giai đoạn trận đấu.
- Post-game Analysis.
- Automated Tournament và đánh giá hiệu năng.

## 7. Lưu ý

Các file JSON trong `game_logs` được tạo tự động và không được đưa lên GitHub theo mặc định.

Các thành viên cần thống nhất giao diện Game Engine API khi triển khai tác tử AI mới.
