# Othello 8×8 – Multi-level Game AI

**Đồ án học phần:** Trí tuệ nhân tạo nâng cao  
**Tên đề tài:** Xây dựng và đánh giá hệ thống Game AI đa cấp độ cho trò chơi Othello 8×8  
**Trường:** Đại học Sài Gòn – Khoa Công nghệ Thông tin

## 1. Giới thiệu

Dự án xây dựng và đánh giá hệ thống Game AI cho trò chơi Othello 8×8, tích hợp năm nhóm tác tử AI với những phương pháp ra quyết định khác nhau:

1. Random Agent.
2. Heuristic Agent.
3. Minimax kết hợp Alpha-Beta Pruning.
4. Monte Carlo Tree Search (MCTS/UCT).
5. Neural Network Agent sử dụng Convolutional Neural Network (CNN).

Hệ thống cung cấp giao diện trò chơi bằng Tkinter, hỗ trợ người chơi đối đầu với các tác tử AI hoặc quan sát hai AI tự động thi đấu.

Bên cạnh chức năng chơi Othello, dự án triển khai Game History, Game Replay, quy trình huấn luyện và đánh giá CNN, thí nghiệm Ablation Study và giải đấu Round-robin nhằm so sánh hiệu quả của các phương pháp AI.

## 2. Chức năng đã triển khai

### 2.1. Game Engine

- Xây dựng bàn cờ Othello 8×8 và khởi tạo trạng thái ban đầu.
- Xác định và kiểm tra nước đi hợp lệ.
- Thực hiện nước đi và lật quân theo tám hướng.
- Xử lý trường hợp PASS khi người chơi không có nước đi hợp lệ.
- Kiểm tra điều kiện kết thúc và xác định kết quả thắng, thua hoặc hòa.
- Cung cấp môi trường `OthelloEnv` để phục vụ việc tích hợp và kiểm thử các tác tử AI.

### 2.2. Các tác tử AI

| Tác tử | Phương pháp |
|---|---|
| Random | Lựa chọn ngẫu nhiên một nước đi hợp lệ |
| Heuristic | Đánh giá trạng thái dựa trên Corner Control, Mobility, Positional Weight và Disc Difference |
| Minimax | Tìm kiếm đối kháng kết hợp Alpha-Beta Pruning |
| MCTS/UCT | Tìm kiếm dựa trên mô phỏng với cơ chế lựa chọn nút UCT |
| CNN | Học chính sách lựa chọn nước đi từ dữ liệu Teacher Agents |

Neural Network Agent gồm các hướng phát triển chính:

- **CNN V1:** Heuristic Teacher.
- **CNN V2:** Minimax Teacher.
- **CNN V3.0/V3.1:** MCTS Teacher.
- **CNN V3.2:** D4 Data Augmentation.
- **CNN V4.0:** Multi-Teacher Learning.
- **CNN V4.1:** Weighted Multi-Teacher Learning.
- **CNN V5:** Residual CNN.

Các mô hình sử dụng Legal-action Mask để giới hạn hành động được lựa chọn trong tập nước đi hợp lệ.


### 2.3. Giao diện trò chơi

Giao diện được xây dựng bằng **Tkinter**, hỗ trợ **9 chế độ chơi**:

1. Human vs Human.
2. Human vs Heuristic.
3. Human vs Minimax.
4. Human vs CNN (Heuristic).
5. Human vs CNN (Minimax).
6. Human vs CNN (V4.1 Weighted).
7. Human vs CNN (V5 Residual).
8. Human vs MCTS/UCT.
9. AI vs AI.

Trong chế độ **AI vs AI**, người dùng có thể:

- Lựa chọn độc lập tác tử điều khiển quân Đen và quân Trắng.
- Quan sát hai AI tự động luân phiên thực hiện nước đi.
- Tạm dừng hoặc tiếp tục trận đấu.
- Theo dõi lượt chơi, tỷ số và kết quả.
- Tự động lưu lịch sử JSON khi trận đấu kết thúc.

Chế độ AI vs AI cung cấp tám lựa chọn tác tử: **Random, Heuristic, Minimax D2, MCTS 64, CNN V1, CNN V2, CNN V4.1 và CNN V5**.

#### Xử lý bất đồng bộ và khả năng phản hồi của giao diện

Để hạn chế tình trạng giao diện không phản hồi khi AI thực hiện các tác vụ tính toán dài, hệ thống sử dụng **worker thread kết hợp Queue**:

- Thực hiện quá trình suy luận AI trên worker thread, tách khỏi luồng xử lý giao diện Tkinter.
- Chuyển kết quả qua `Queue` và sử dụng `root.after()` để kiểm tra, cập nhật bàn cờ trên luồng chính.
- Sử dụng mã định danh lượt tính toán để bỏ qua các kết quả không còn hợp lệ khi trạng thái trận đấu thay đổi.
- Hỗ trợ tạm dừng và tiếp tục chế độ AI vs AI mà không áp dụng nước đi trong thời gian tạm dừng.

Cơ chế này cải thiện khả năng phản hồi của giao diện nhưng không trực tiếp giảm thời gian suy luận. Những worker thread đã khởi chạy vẫn có thể tiếp tục tính toán sau khi tác vụ bị hủy ở cấp giao diện.

### 2.4. Game History và Game Replay

- Ghi nhận trạng thái bàn cờ ban đầu, từng hành động và lượt PASS.
- Lưu lịch sử trận đấu dưới dạng JSON.
- Lưu kết quả thắng, thua hoặc hòa.
- Mở lại tệp lịch sử bằng Game Replay.
- Di chuyển đến lượt trước hoặc lượt tiếp theo.
- Phục dựng và kiểm tra tính nhất quán của các trạng thái bàn cờ từ lịch sử đã lưu.

### 2.5. Thực nghiệm và đánh giá

Hệ thống đã triển khai:

- Kiểm thử tự động bằng pytest.
- Thực nghiệm Heuristic đối đầu Random.
- Đánh giá MCTS với các ngân sách mô phỏng khác nhau.
- Huấn luyện và đánh giá các phiên bản CNN V1–V5.
- So sánh các mô hình CNN qua nhiều đối thủ và seed.
- Phân tích dữ liệu Multi-Teacher.
- Ablation Study với hai phương án Keep Conflicts và Drop Conflicts.
- Giải đấu Round-robin giữa năm tác tử AI.

**Kết quả kiểm thử:** 56/56 bài kiểm thử thành công trong lần đánh giá được ghi nhận.

**Giải đấu Round-robin:** 1.000 trận, gồm 10 cặp đối kháng với 100 trận mỗi cặp.

Cấu hình giải đấu:

- Seed: `2026`, `3026`.
- Opening plies: `8`.
- Minimax: Depth `2`, có Alpha-Beta Pruning.
- MCTS: `64 simulations`.
- Neural Network: CNN V4.1 Weighted Multi-Teacher.

Bảng xếp hạng theo Score Rate:

| Hạng | Tác tử | Score Rate |
|---|---|---|
| 1 | MCTS 64 | 76,88% |
| 2 | Minimax D2 | 75,25% |
| 3 | Heuristic | 46,38% |
| 4 | CNN V4.1 | 41,63% |
| 5 | Random | 9,88% |

Kết quả chỉ phản ánh hiệu quả của các tác tử trong những cấu hình và điều kiện thực nghiệm đã sử dụng.

## 3. Công nghệ sử dụng

- **Python 3:** Ngôn ngữ lập trình chính.
- **Tkinter:** Giao diện người dùng.
- **NumPy:** Xử lý dữ liệu và trạng thái bàn cờ.
- **PyTorch:** Xây dựng, huấn luyện và suy luận CNN.
- **pytest:** Kiểm thử tự động.
- **JSON:** Lưu trữ lịch sử trận đấu.
- **CSV:** Lưu trữ và tổng hợp kết quả thực nghiệm.
- **Git/GitHub:** Quản lý phiên bản mã nguồn và cộng tác.

## 4. Cấu trúc project

```text
OTHELLO_PRJ/
├── agents/                  # Random/Heuristic/Minimax/MCTS/CNN Agents
├── core/                    # Game Engine và Othello Environment
├── gui/                     # Giao diện Tkinter
├── history/                 # Game History và Game Replay
├── test/                    # Bộ kiểm thử tự động
├── notebooks/               # Notebook khởi tạo và thử nghiệm ban đầu
├── scripts/                 # Sinh dữ liệu, huấn luyện, đánh giá, Round-robin
├── data/                    # Dataset huấn luyện CNN
├── checkpoints/             # Checkpoint mô hình CNN
├── results/                 # Kết quả thực nghiệm và giải đấu
├── game_logs/               # Lịch sử các trận đấu JSON
├── main.py                  # Điểm khởi chạy ứng dụng
├── README.md                # Tài liệu giới thiệu và hướng dẫn
├── CNN_GUIDE.md             # Hướng dẫn liên quan đến CNN
├── EXPERIMENT_RESULTS_CNN.md
└── requirements-cnn.txt
```

## 5. Hướng dẫn cài đặt và chạy chương trình

### 5.1. Chuẩn bị môi trường

Khuyến nghị Python 3.13 và một môi trường ảo Python riêng.

Clone repository:

```powershell
git clone https://github.com/bibo0610/TTNTNC_10.git
cd TTNTNC_10/OTHELLO_PRJ
```

Cài đặt các thư viện:

```powershell
python -m pip install -r requirements-cnn.txt
python -m pip install pytest
```

Lưu ý: Các chế độ CNN cần checkpoint tương ứng trong thư mục `checkpoints/`.

### 5.2. Khởi chạy GUI

```powershell
python main.py
```

Trong giao diện, lựa chọn chế độ chơi và bắt đầu trận mới.

Để trình diễn AI vs AI:

1. Chọn `AI vs AI (Demo)`.
2. Lựa chọn AI quân Đen và AI quân Trắng.
3. Nhấn `BẮT ĐẦU AI VS AI`.
4. Quan sát trận đấu hoặc sử dụng `TẠM DỪNG` / `TIẾP TỤC`.
5. Sau khi kết thúc, kiểm tra lịch sử trận đấu trong `game_logs/`.

### 5.3. Chạy kiểm thử

Chạy toàn bộ bài kiểm thử:

```powershell
python -m pytest test/ -v
```

Chạy riêng một nhóm kiểm thử, ví dụ Game Engine:

```powershell
python -m pytest test/test_engine.py -v
```

### 5.4. Game Replay

Trong giao diện, chọn `Mở file lịch sử (JSON)` và mở tệp lịch sử cần xem.

Sử dụng các nút Previous và Next để di chuyển qua những trạng thái đã lưu.

Tệp JSON có thể được tạo từ các trận chơi trên GUI hoặc từ chương trình thực nghiệm có hỗ trợ lưu Replay.

## 6. Quản lý thực nghiệm và dữ liệu

Các thực nghiệm được tổ chức theo cấu hình thuật toán, seed, đối thủ và số trận đánh giá.

Những thông tin cần ghi nhận khi thực hiện một thí nghiệm gồm:

- Tên và mã thí nghiệm.
- Người thực hiện và thời gian thực hiện.
- Phiên bản mã nguồn hoặc commit SHA.
- Thuật toán và checkpoint được sử dụng.
- Seed và các tham số.
- Lệnh thực thi.
- Kết quả W/D/L, Win Rate, Score Rate hoặc chỉ số huấn luyện.
- Đường dẫn đến CSV, log và các tài liệu liên quan.

Mã nguồn được quản lý bằng GitHub. Các tệp kết quả, checkpoint, hình ảnh và tài liệu có thể được chia sẻ thông qua Google Drive.

Nhật ký thực nghiệm và phân công công việc được quản lý trong hồ sơ cộng tác của nhóm.

## 7. Quản lý cộng tác

Nhóm tổ chức công việc theo hai hướng:

- **Trục dọc:** Phân công trách nhiệm chính theo nhóm công việc, bao gồm triển khai thuật toán, thí nghiệm, phân tích kết quả, báo cáo và slide.
- **Trục ngang:** Chia từng nhóm công việc thành những nhiệm vụ cụ thể, có người thực hiện, người kiểm tra và minh chứng kết quả.

Các thay đổi mã nguồn được theo dõi bằng Git. Mỗi nhiệm vụ nên được liên kết với commit, kết quả thực nghiệm hoặc tài liệu tương ứng.

## 8. Hạn chế và hướng phát triển

Các hướng phát triển tiếp theo bao gồm:

- Hoàn thiện Post-game Analysis để đánh giá chất lượng nước đi và phân tích diễn biến chiến thuật.
- Mở rộng thực nghiệm với nhiều seed, checkpoint và cấu hình thuật toán.
- Nghiên cứu Iterative Deepening, Move Ordering và Transposition Table nhằm tối ưu tìm kiếm.
- Khảo sát thêm các ngân sách MCTS và chiến lược mô phỏng.
- Nghiên cứu kết hợp CNN với MCTS, Self-play và Reinforcement Learning.
- Bổ sung khả năng điều chỉnh thông số thuật toán trực tiếp trên giao diện.
- Tối ưu cơ chế quản lý và hủy các tác vụ AI đang thực thi.

## 9. Lưu ý

- Các tác tử AI sử dụng chung Game Engine và cần trả về nước đi hợp lệ hoặc `None` khi PASS.
- Cần duy trì tính nhất quán giữa checkpoint CNN, kiến trúc mạng và dữ liệu đánh giá.
- Các kết quả thực nghiệm phụ thuộc vào seed, đối thủ và cấu hình được sử dụng.
- Những tệp dữ liệu và checkpoint có dung lượng lớn nên được quản lý phù hợp, tránh làm repository tăng kích thước không cần thiết.
- Các tệp JSON đã được Git theo dõi sẽ không tự động bị loại bỏ khi cập nhật `.gitignore`.
- Không nên chỉnh sửa lịch sử commit đã chia sẻ chỉ để thay đổi số lượng commit của từng thành viên.

## 10. Thành viên thực hiện

**Giảng viên hướng dẫn:** TS. Đỗ Như Tài

- Nguyễn Dương Bảo Trân – 3123410386
- Giang Hào Tường – 3123410421
- Lê Thị Thủy Tiên – 3123410371
- Trương Thị Huyền Trang – 3123410385

**Repository:** https://github.com/bibo0610/TTNTNC_10