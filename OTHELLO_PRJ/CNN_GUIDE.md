# CNN — Hướng dẫn sinh dữ liệu, huấn luyện và đánh giá V1/V2

## 1. Các file và quy trình chính

| File                           | Vai trò                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `scripts/generate_cnn_data.py` | Sinh dữ liệu có giám sát từ Teacher                                                                     |
| `scripts/train_cnn.py`         | Huấn luyện CNN, chia validation theo trận, Early Stopping, lưu checkpoint tốt nhất theo Validation Loss |
| `scripts/evaluate_cnn.py`      | Cho CNN đấu Random, Heuristic, Minimax, MCTS hoặc CNN khác                                              |
| `agents/cnn_agent.py`          | `CNNPolicy`, `CNNAgent`, Legal-action Mask và các hàm agent cho GUI                                     |
| `gui/othello_gui.py`           | Giao diện lựa chọn chế độ chơi                                                                          |
| `test/`                        | Các bài kiểm thử tự động                                                                                |

Quy trình: **Cài thư viện → Kiểm thử → Sinh dataset V1/V2 → Kiểm tra dataset → Huấn luyện → Kiểm tra checkpoint → Đánh giá → (Tùy chọn) dùng trong GUI.**

Đầu vào CNN có kích thước `(3, 8, 8)`; nhãn là vị trí nước đi từ `0` đến `63` (`row * 8 + col`). Legal-action Mask loại các vị trí không hợp lệ. V1 và V2 dùng **cùng kiến trúc CNN** nhưng khác nguồn nhãn Teacher.

## 2. Chuẩn bị môi trường (Windows)

Mở **CMD hoặc PowerShell tại thư mục gốc `ATHELLO_PRJ`** (thư mục chứa `main.py`). Tất cả lệnh bên dưới chạy tại đây.

```powershell
python --version
python -m pip install -r requirements-cnn.txt
python -m pytest test/ -v
```

Nếu muốn tạo môi trường ảo (không bắt buộc):

```powershell
python -m venv .venv
```

- CMD: `.venv\Scripts\activate.bat`
- PowerShell: `.\.venv\Scripts\Activate.ps1`

Sau khi kích hoạt môi trường ảo, chạy lại lệnh cài thư viện ở trên. Nếu không kích hoạt, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` thay cho `python` ở các lệnh tiếp theo.

**Kiểm tra tham số script**, đặc biệt khi một thành viên dùng bản source khác:

```powershell
python -m scripts.generate_cnn_data --help
python -m scripts.train_cnn --help
python -m scripts.evaluate_cnn --help
```

## 3. Sinh dữ liệu huấn luyện

### 3.1. Chạy thử trước (tùy chọn)

Không dùng dữ liệu thử này làm bộ dữ liệu chính thức:

```powershell
python -m scripts.generate_cnn_data --teacher heuristic --games 4 --seed 42 --out data/cnn_v1_smoke.npz
python -m scripts.generate_cnn_data --teacher minimax --minimax-depth 1 --games 4 --seed 42 --out data/cnn_v2_smoke.npz
```

### 3.2. CNN V1 — Heuristic Teacher

```powershell
python -m scripts.generate_cnn_data --teacher heuristic --games 200 --seed 42 --epsilon 0.15 --out data/cnn_v1_heuristic.npz
```

### 3.3. CNN V2 — Minimax/Alpha-Beta Teacher (Depth 3)

```powershell
python -m scripts.generate_cnn_data --teacher minimax --minimax-depth 3 --games 100 --seed 42 --epsilon 0.15 --out data/cnn_v2_minimax.npz
```

**Tham số sinh dữ liệu:**

- `--teacher`: `heuristic`, `minimax` hoặc `mcts`; tài liệu này dùng hai lựa chọn đầu.
- `--games`: số trận sinh dữ liệu.
- `--minimax-depth`: độ sâu tìm kiếm, chỉ áp dụng cho Minimax Teacher.
- `--seed`: seed cho quá trình tạo dữ liệu.
- `--epsilon 0.15`: xác suất 15% chọn **một nước ngẫu nhiên hợp lệ để tiếp tục diễn biến ván đấu**; nhãn vẫn là nước được Teacher đề xuất.
- `--out`: đường dẫn lưu dataset `.npz`.

**Chú ý:** lệnh sinh dữ liệu có thể **ghi đè dataset cùng tên**. Muốn lưu nhiều lần thử, hãy đổi tên `--out`. Sinh dữ liệu Minimax Depth 3 thường tốn nhiều CPU hơn Heuristic.

Mỗi file `.npz` chứa `states`, `labels`, `masks`, `game_ids`. Số mẫu không bắt buộc bằng `games × 60`, vì Othello có thể có lượt PASS hoặc kết thúc khi không còn nước đi.

## 4. Kiểm tra dataset vừa tạo

**V1:**

```powershell
python -c "import numpy as np; d=np.load('data/cnn_v1_heuristic.npz'); y=d['labels']; print('V1 | samples:',len(y),'| games:',len(np.unique(d['game_ids'])),'| states:',d['states'].shape,'| legal:',bool(d['masks'][np.arange(len(y)),y].all()))"
```

**V2:**

```powershell
python -c "import numpy as np; d=np.load('data/cnn_v2_minimax.npz'); y=d['labels']; print('V2 | samples:',len(y),'| games:',len(np.unique(d['game_ids'])),'| states:',d['states'].shape,'| legal:',bool(d['masks'][np.arange(len(y)),y].all()))"
```

Kết quả hợp lệ phải có `states: (N, 3, 8, 8)`, số trận phù hợp lệnh tạo dữ liệu và `legal: True`. Đây là **kiểm tra tính hợp lệ**, không đo độ mạnh của Teacher.

## 5. Huấn luyện CNN V1 và V2

Hai mô hình được **huấn luyện từ đầu**, lưu checkpoint mới.

### 5.1. Huấn luyện V1

```powershell
python -m scripts.train_cnn --data data/cnn_v1_heuristic.npz --out checkpoints/cnn_v1_heuristic.pt --epochs 30 --patience 6 --batch-size 128 --val-split 0.2 --lr 0.0005 --weight-decay 0.0001 --seed 42
```

### 5.2. Huấn luyện V2

```powershell
python -m scripts.train_cnn --data data/cnn_v2_minimax.npz --out checkpoints/cnn_v2_minimax.pt --epochs 30 --patience 6 --batch-size 128 --val-split 0.2 --lr 0.0005 --weight-decay 0.0001 --seed 42
```

**Tham số huấn luyện:**

- `--data`: dataset đầu vào.
- `--out`: checkpoint `.pt` đầu ra.
- `--epochs 30`: số epoch tối đa.
- `--patience 6`: dừng sớm nếu Validation Loss không giảm qua 6 epoch liên tiếp; `0` để tắt.
- `--batch-size 128`: kích thước mini-batch.
- `--val-split 0.2`: dùng khoảng 20% số **trận** cho validation; không chia ngẫu nhiên từng nước đi.
- `--lr 0.0005`: learning rate của Adam.
- `--weight-decay 0.0001`: regularization thông qua Adam weight decay.
- `--seed 42`: seed chia dữ liệu và khởi tạo huấn luyện.
- `--device cpu`: mặc định. Chỉ dùng `--device cuda` khi PyTorch nhận GPU CUDA.

Huấn luyện in ra `train_loss`, `val_loss`, `val_accuracy` mỗi epoch. **Checkpoint được chọn theo Validation Loss thấp nhất**, nên epoch lưu checkpoint không nhất thiết là epoch có Accuracy cao nhất hoặc epoch cuối.

Nếu checkpoint tại `--out` **đã tồn tại**, script sẽ báo lỗi để tránh ghi đè. Đổi thành tên mới (ví dụ `cnn_v1_run02.pt`) để giữ lịch sử. Chỉ dùng `--overwrite` khi chủ động muốn thay thế checkpoint đó.

Các giá trị trên là cấu hình thực nghiệm khởi đầu, không bảo đảm tối ưu cho mọi máy hoặc dataset.

## 6. Xác minh checkpoint sau huấn luyện

### 6.1. Kiểm tra thông tin checkpoint

**V1:**

```powershell
python -c "import torch; c=torch.load('checkpoints/cnn_v1_heuristic.pt',map_location='cpu',weights_only=True); print('V1 | best epoch:',c['epoch'],'| val_loss:',c['val_loss'],'| val_accuracy:',c['val_accuracy'])"
```

**V2:**

```powershell
python -c "import torch; c=torch.load('checkpoints/cnn_v2_minimax.pt',map_location='cpu',weights_only=True); print('V2 | best epoch:',c['epoch'],'| val_loss:',c['val_loss'],'| val_accuracy:',c['val_accuracy'])"
```

Checkpoint mới lưu trọng số, epoch được chọn, validation metrics, cấu hình huấn luyện và SHA-256 của dataset. Chỉ nạp checkpoint từ nguồn tin cậy.

### 6.2. Kiểm tra CNN trả về nước đi hợp lệ

**V1:**

```powershell
python -c "from agents.cnn_agent import CNNAgent; from core.othello_engine import init_board,get_valid_moves; b=init_board(); m=CNNAgent('checkpoints/cnn_v1_heuristic.pt')(b,1); print('V1 move:',m,'| legal:',m in get_valid_moves(b,1))"
```

**V2:**

```powershell
python -c "from agents.cnn_agent import CNNAgent; from core.othello_engine import init_board,get_valid_moves; b=init_board(); m=CNNAgent('checkpoints/cnn_v2_minimax.pt')(b,1); print('V2 move:',m,'| legal:',m in get_valid_moves(b,1))"
```

Cả hai cần trả về `legal: True` trên bàn cờ ban đầu. Đây là bài kiểm tra **suy luận và nước đi hợp lệ**, chưa phải đánh giá Win Rate.

## 7. Đánh giá các checkpoint (Evaluation)

Để so sánh công bằng, dùng cùng `--games`, `--seed`, `--opening-plies` và cấu hình đối thủ. `--games 40` nghĩa là **40 trận với mỗi đối thủ**, chia đều 20 trận CNN cầm đen và 20 trận CNN cầm trắng. Script tạo cùng thế cờ mở đầu cho hai trận trong một cặp, rồi hoán đổi màu quân.

### 7.1. V1 trước Random, Heuristic, Minimax Depth 2

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v1_heuristic.pt --opponent all --depth 2 --games 40 --seed 42 --opening-plies 8
```

### 7.2. V2 trước Random, Heuristic, Minimax Depth 2

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v2_minimax.pt --opponent all --depth 2 --games 40 --seed 42 --opening-plies 8
```

`--opponent all` gồm **Random, Heuristic, Minimax**;

### 7.3. Đánh giá trước Minimax Depth 3

**V1:**

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v1_heuristic.pt --opponent minimax --depth 3 --games 40 --seed 42 --opening-plies 8
```

**V2:**

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v2_minimax.pt --opponent minimax --depth 3 --games 40 --seed 42 --opening-plies 8
```

### 7.4. CNN V2 đấu trực tiếp CNN V1

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v2_minimax.pt --opponent cnn --opponent-checkpoint checkpoints/cnn_v1_heuristic.pt --games 40 --seed 42 --opening-plies 8
```

W/D/L được tính theo **CNN V2** là checkpoint sau `--checkpoint`.

### 7.5. Đánh giá với MCTS (tùy chọn)

**V1:**

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v1_heuristic.pt --opponent mcts --mcts-simulations 64 --games 40 --seed 42 --opening-plies 8
```

**V2:**

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v2_minimax.pt --opponent mcts --mcts-simulations 64 --games 40 --seed 42 --opening-plies 8
```

### 7.6. Thử nghiệm seed khác

```powershell
python -m scripts.evaluate_cnn --checkpoint checkpoints/cnn_v2_minimax.pt --opponent heuristic --games 40 --seed 123 --opening-plies 8
```

Seed khác giúp kiểm tra khả năng chơi ở tập thế cờ mở đầu khác; không nên chỉ dựa vào một seed để kết luận sức mạnh mô hình.

### 7.7. Cách đọc đầu ra

- `CNN BLACK` / `CNN WHITE`: kết quả theo màu quân.
- `TOTAL`: tổng hợp hai màu.
- `W`, `D`, `L`: thắng, hòa, thua.
- `win_rate = W / games`.
- `score_rate = (W + 0.5 × D) / games`.
- `unique_openings`: số bàn cờ mở đầu khác nhau thực tế.

**Lưu ý:** Validation Accuracy đo khả năng dự đoán nước Teacher trên validation; Score Rate đo kết quả chơi thực tế trước đối thủ. Hai thước đo này không đồng nghĩa.

## 8. Sử dụng checkpoint mới trong GUI

Khởi chạy:

```powershell
python main.py
```

GUI hiện dùng các đường dẫn mặc định trong `agents/cnn_agent.py`:

- `DEFAULT_CHECKPOINT`: CNN V1 cũ (`cnn_policy.pt`).
- `MINIMAX_CHECKPOINT`: CNN V2 cũ (`cnn_v2_minimax.pt`).

Nếu muốn GUI dùng checkpoint mới, đổi **chỉ tên file** trong hai hằng số trên thành:

```python
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cnn_v1_heuristic_new.pt"
MINIMAX_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cnn_v2_minimax_new.pt"
```

Giữ nguyên các hàm `cnn_agent(board, player)`, `cnn_minimax_agent(board, player)` và logic GUI. **Khởi động lại `python main.py`** sau khi thay checkpoint để cache mô hình được nạp lại. Nếu chỉ muốn đánh giá từ dòng lệnh, không cần sửa đường dẫn trong GUI.

## 9. Các lỗi thường gặp

| Hiện tượng                                            | Cách xử lý                                                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: scripts` hoặc `agents`          | Mở terminal ở thư mục chứa `main.py` rồi chạy bằng `python -m ...`                                        |
| `No module named torch / numpy / pytest`              | Chạy `python -m pip install -r requirements-cnn.txt` trong đúng môi trường Python                         |
| `unrecognized arguments: --patience / --weight-decay` | Source chưa cập nhật bản `V1_V2_TRAIN_PATCH`/`TRAIN_READY`; kiểm tra `python -m scripts.train_cnn --help` |
| `Dataset not found`                                   | Chạy bước sinh dữ liệu và kiểm tra đúng đường dẫn `.npz`                                                  |
| `Checkpoint already exists`                           | Dùng tên file `--out` khác; không tùy tiện dùng `--overwrite`                                             |
| `FileNotFoundError` khi đánh giá hoặc mở GUI          | Kiểm tra checkpoint tồn tại và đường dẫn đã trỏ đúng tên                                                  |
| `CUDA is not available`                               | Bỏ `--device cuda` hoặc dùng `--device cpu`                                                               |
| Validation Loss tăng dù Train Loss giảm               | Có thể overfitting; kiểm tra epoch được chọn theo Validation Loss, không mặc định dùng epoch cuối         |

## 10. Quy tắc lưu phiên bản và so sánh

- Không đặt trùng đường dẫn dataset/checkpoint giữa V1, V2, V3 hoặc các lần huấn luyện.
- Khi đánh giá hai checkpoint, giữ giống nhau số trận, seed, opening plies và cấu hình đối thủ.
- Ghi rõ nguồn Teacher, độ sâu Minimax, số trận dữ liệu, seed và cấu hình train trong báo cáo sau này.
- Dataset và checkpoint là các đầu ra của quy trình huấn luyện; không cần sửa Game Engine hay thuật toán Heuristic/Minimax để chạy lại V1/V2.
- Hướng dẫn này không bao gồm kết quả thực nghiệm có sẵn; mỗi thành viên tự chạy các lệnh đánh giá để thu thập số liệu.
