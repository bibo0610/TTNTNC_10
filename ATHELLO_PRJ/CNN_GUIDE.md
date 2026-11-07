# Othello CNN

Mở terminal tại thư mục gốc `ATHELLO_PRJ` (nơi chứa `main.py`). Cần Python 3.10+.

```powershell
python -m pip install -r requirements-cnn.txt
python -m scripts.generate_cnn_data --games 200 --epsilon 0.15
python -m scripts.train_cnn --epochs 20 --batch-size 128
python -m scripts.evaluate_cnn --games 40
python -m pytest test/ -q
python main.py
```

Chọn **Human vs CNN** trong giao diện để chơi với model đã train. Checkpoint được nạp lần đầu khi CNN đến lượt. Nếu muốn thay checkpoint, đóng và mở lại GUI.

## Cấu trúc

- `agents/cnn_agent.py`: mạng CNN, encode, legal mask, nạp checkpoint, agent `__call__(board,player)`.
- `scripts/generate_cnn_data.py`: sinh dữ liệu giáo viên Heuristic (mẫu gồm `states`, `labels`, `masks`, `game_ids`).
- `scripts/train_cnn.py`: Supervised Cross Entropy + legal mask; chia train/validation **theo trận** tránh chia cùng trận vào hai nhóm; lưu checkpoint tốt nhất.
- `scripts/evaluate_cnn.py`: đấu với Random và Heuristic, luân phiên đen/trắng; W/D/L, win rate và score rate.
- `test/test_cnn.py`: kiểm thử encoding, legal mask, pass, tạo dữ liệu.

## Ghi chú phương pháp

CNN chỉ bắt chước chính sách chọn nước của Heuristic từ các trạng thái đã quan sát, **không được bảo đảm mạnh hơn Heuristic**. Epsilon exploration khiến nhiều trạng thái khác biệt xuất hiện, nhưng nhãn vẫn là nước Heuristic chọn tại trạng thái đó. Mẫu PASS không có nhãn trong tập 64 hành động; ở thời điểm suy luận agent trả `None` và engine tự bỏ lượt. CNN đầu ra 64 logits; masked cross entropy dùng legal move mask. `argmax` sau mask tương đương chọn nước có xác suất softmax cao nhất. Tỷ lệ thắng đo trên các ván đấu, không chỉ validation accuracy.

Trong bản project ZIP cung cấp mới chỉ có Heuristic, chưa có Minimax/Alpha-Beta. Khi nhóm cấp 3 hoàn thành, có thể tích hợp agent đó làm teacher mới, nhưng cần tạo thêm dữ liệu và huấn luyện lại. Không nên dùng cùng trận trong validation và train. Để ước lượng hiệu năng tin cậy hơn, tăng số trận và thử nhiều seed.

`data/*.npz` và `checkpoints/*.pt` được .gitignore để tránh commit dữ liệu/model nặng. Nếu muốn demo ngay trên máy khác, copy `checkpoints/cnn_policy.pt` thủ công.
