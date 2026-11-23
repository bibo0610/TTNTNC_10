# Kết quả thực nghiệm CNN — Othello 8×8

**Nguồn:** log chạy huấn luyện/đánh giá mà nhóm đã cung cấp trong quá trình phát triển; checkpoint và dataset tương ứng được lưu trong các thư mục checkpoints/ và data/ của project hiện tại. Các số Score Rate dưới đây dựa trên log, **không phải kết quả chạy lại toàn bộ giải đấu trong bản rà soát**.

## Thiết lập đánh giá

- **Đánh giá CNN V1–V5 trước Heuristic và các CNN khác:** mỗi seed (42, 123) có 40 trận, 20 random openings, hoán đổi quân Đen/Trắng; `opening_plies=8`. Các phép thử bổ sung với MCTS và Round-robin có số trận/seed riêng, được ghi cụ thể ở các mục dưới.
- `W/D/L` luôn tính theo **tác tử/mô hình ghi ở trước** trong tên phép thử; `Win Rate = W/N × 100%`; `Score Rate = (W + 0,5 × D)/N × 100%`.
- Đối đầu hai model CNN không hoàn toàn thay thế được đánh giá trước một baseline chung.
- Các bộ seed và cặp trận hoán đổi màu chỉ hỗ trợ kiểm soát điều kiện đánh giá; chưa đủ để khẳng định ưu thế tổng quát hoặc ý nghĩa thống kê của mọi chênh lệch.

## So sánh với Heuristic

| CNN      | Seed 42 W/D/L | Score Rate | Seed 123 W/D/L | Score Rate | Tổng 80 trận Score Rate |
| -------- | ------------: | ---------: | -------------: | ---------: | ----------------------: |
| V1       |       12/1/27 |     31,25% |        10/0/30 |     25,00% |                  28,13% |
| V2       |       16/2/22 |     42,50% |        20/0/20 |     50,00% |                  46,25% |
| V3.0     |        8/0/32 |     20,00% |        11/2/27 |     30,00% |                  25,00% |
| V3.1     |        6/1/33 |     16,25% |         7/3/30 |     21,25% |                  18,75% |
| V3.2     |        4/1/35 |     11,25% |         7/0/33 |     17,50% |                  14,38% |
| V4.0     |       11/1/28 |     28,75% |        12/1/27 |     31,25% |                  30,00% |
| **V4.1** |   **21/1/18** | **53,75%** |    **15/1/24** | **38,75%** |              **46,25%** |
| V5       |       14/1/25 |     36,25% |        20/3/17 |     53,75% |                  45,00% |

## Các kết quả chung khác

| CNN  | Đối thủ         | Seed 42 W/D/L; Score Rate | Seed 123 W/D/L; Score Rate | Tổng 80 trận Score Rate |
| ---- | --------------- | ------------------------: | -------------------------: | ----------------------: |
| V3.2 | CNN V3.1        |           25/1/14; 63,75% |            28/0/12; 70,00% |                  66,88% |
| V3.2 | CNN V2          |           16/2/22; 42,50% |            14/1/25; 36,25% |                  39,38% |
| V4.0 | CNN V2          |           15/4/21; 42,50% |            25/0/15; 62,50% |                  52,50% |
| V4.0 | CNN V3.2        |            30/2/8; 77,50% |             29/2/9; 75,00% |                  76,25% |
| V4.1 | CNN V4.0        |           20/1/19; 51,25% |            28/0/12; 70,00% |                  60,63% |
| V4.1 | CNN V2          |           23/0/17; 57,50% |            22/0/18; 55,00% |                  56,25% |
| V4.1 | Minimax depth 2 |            5/2/33; 15,00% |             6/0/34; 15,00% |                  15,00% |
| V5   | CNN V4.1        |           11/3/26; 31,25% |            16/1/23; 41,25% |                  36,25% |
| V5   | CNN V2          |           14/4/22; 40,00% |            12/1/27; 31,25% |                  35,63% |
| V5   | Minimax depth 2 |            6/0/34; 15,00% |              2/0/38; 5,00% |                  10,00% |
| V5   | Random          |            34/0/6; 85,00% |             38/1/1; 96,25% |                  90,63% |

## Đối chiếu checkpoint và Validation

| CNN  | Checkpoint                      | Epoch được lưu | Validation Loss ở epoch lưu |
| ---- | ------------------------------- | -------------: | --------------------------: |
| V3.0 | `cnn_v3_0_mcts_128_final.pt`    |              2 |                      1,9662 |
| V3.1 | `cnn_v3_1_mcts_128_final.pt`    |              5 |                      1,9869 |
| V3.2 | `cnn_v3_2_mcts_d4_final.pt`     |             21 |                      1,9746 |
| V4.0 | `cnn_v4_multi_teacher_final.pt` |              5 |                      1,7898 |
| V4.1 | `cnn_v4_1_weighted_final.pt`    |              6 |                      1,7944 |
| V5   | `cnn_v5_residual_final.pt`      |              3 |                      1,7911 |

## CNN V4.1 đối đầu MCTS/UCT (64 simulations)

Phép thử bổ sung độc lập với Round-robin: checkpoint `checkpoints/cnn_v4_1_weighted_final.pt`, đối thủ MCTS 64 simulations, `opening_plies=8`, hoán đổi quân Đen/Trắng; **20 trận/seed**, tổng **40 trận**. Không gộp với các phép thử Round-robin sử dụng seed khác.

| Tác tử đứng trước | Đối thủ | Seed 42 W/D/L; Score Rate | Seed 123 W/D/L; Score Rate | Tổng 40 trận W/D/L | Tổng Score Rate |
| ----------------- | ------- | ------------------------: | -------------------------: | -----------------: | --------------: |
| CNN V4.1          | MCTS 64 |            8/1/11; 42,50% |             5/1/14; 27,50% |        **13/2/25** |      **35,00%** |

Tổng hợp: **Win Rate = 32,50%** (13/40); Score Rate = (13 + 0,5 × 2)/40 = **35,00%**. Kết quả này **không thay thế** số liệu CNN V4.1 vs MCTS trong Round-robin 100 trận/cặp.

## Thống kê dữ liệu CNN V4.0/V4.1 Multi-Teacher

**File dữ liệu:** `data/cnn_v4_multi_teacher_final.npz`; metadata: `data/cnn_v4_multi_teacher_final.json`; tạo bởi `scripts/build_cnn_v4_data.py` với `method=balanced_without_replacement`, `conflict_policy=keep`, seed 42.

| Chỉ số trên dataset cân bằng dùng để huấn luyện                |                         Giá trị |
| -------------------------------------------------------------- | ------------------------------: |
| Tổng số mẫu                                                    |                          17.844 |
| Số mẫu từ Heuristic                                            |                           5.948 |
| Số mẫu từ Minimax                                              |                           5.948 |
| Số mẫu từ MCTS                                                 |                           5.948 |
| Tổng ván nguồn được chọn (`total_games`)                       |                             400 |
| Trạng thái mã hóa khác nhau                                    |                          15.439 |
| Mẫu lặp ngoài lần xuất hiện đầu tiên                           |         2.405 (13,48% tổng mẫu) |
| Trạng thái có từ hai nhãn hành động khác nhau                  | 82 (0,53% trạng thái khác nhau) |
| Trong đó: trạng thái xung đột liên quan từ hai Teacher trở lên |                              51 |

**Phân bố theo số quân hiện diện trên bàn cờ**, đếm trên cả 17.844 mẫu (phân nhóm mô tả, không phải đo độ đa dạng chiến thuật):

| Nhóm trạng thái | Điều kiện  |     Số mẫu |       Tỷ lệ |
| --------------- | ---------- | ---------: | ----------: |
| Đầu trận        | ≤ 20 quân  |      5.027 |      28,17% |
| Giữa trận       | 21–44 quân |      7.154 |      40,09% |
| Cuối trận       | ≥ 45 quân  |      5.663 |      31,74% |
| **Tổng**        |            | **17.844** | **100,00%** |

**Phân biệt phạm vi thống kê:** Metadata của **các dataset nguồn trước khi lấy cân bằng** ghi `source_unique_encoded_states=20100`, `source_conflicting_states=84`, `cross_teacher_conflicting_states=55`. Các giá trị **15.439 / 82 / 51** ở bảng trên được tính trực tiếp trên dataset đã cân bằng và **không mâu thuẫn** với các giá trị trước khi lấy mẫu. Có trạng thái lặp lại và nhãn không nhất quán, nhưng các thống kê này **chưa chứng minh** chúng là nguyên nhân làm giảm hiệu quả CNN.

## Ablation Study — Keep Conflicts so với Drop Conflicts

**Mục đích:** khảo sát chính sách xử lý trạng thái có nhiều nhãn từ các nguồn Teacher. Hai nhánh có cùng kích thước **15.000 mẫu**, gồm **5.000 mẫu/Teacher** (Heuristic, Minimax, MCTS), cùng kiến trúc CNN cơ sở và Weighted Cross-Entropy với trọng số **[0,6; 1,5; 0,9]**. Seed tạo dataset = 42; seed huấn luyện = 42; Learning Rate = 0,0005; Weight Decay = 0,0001; Batch Size = 128; tối đa 30 epoch; Patience = 6; Validation split = 0,2; bật `--drop-shared-val-states`; không dùng D4 augmentation.

- **Keep:** `data/cnn_ablation_keep.npz` → `checkpoints/cnn_ablation_keep.pt` (`--conflicts keep`).
- **Drop:** `data/cnn_ablation_drop.npz` → `checkpoints/cnn_ablation_drop.pt` (`--conflicts drop`).

### Kết quả huấn luyện Ablation

| Chỉ số                                               |           Keep Conflicts |           Drop Conflicts |
| ---------------------------------------------------- | -----------------------: | -----------------------: |
| Số mẫu trong dataset                                 |                   15.000 |                   15.000 |
| Mẫu Validation trùng hẳn trạng thái Training bị loại |                      477 |                      284 |
| Mẫu Training theo Teacher [Heuristic, Minimax, MCTS] |    [3.989, 3.960, 4.008] |    [3.999, 3.991, 3.981] |
| Tỷ trọng đóng góp có trọng số trong Training         | [0,2004; 0,4975; 0,3021] | [0,2005; 0,5002; 0,2994] |
| Epoch tốt nhất, lưu checkpoint                       |                    **7** |                    **7** |
| Epoch dừng (Early Stopping)                          |                       13 |                       13 |
| **Best Validation Loss**                             |               **1,7789** |               **1,7888** |
| Validation Accuracy tại epoch được lưu               |               **32,58%** |               **31,88%** |

**Chú thích:** Validation Loss không được so sánh như phép đo sức mạnh thi đấu, đặc biệt khi hai tập dữ liệu và tập Validation khác nhau.

### Kết quả thi đấu Ablation

Quy tắc: `W/D/L` thuộc mô hình ở bên trái. Mỗi seed 40 trận; `opening_plies=8`, hoán đổi màu quân. Hai seed = 42 và 123.

| Cặp đấu           | Seed 42 W/D/L; Score Rate | Seed 123 W/D/L; Score Rate | Tổng 80 trận W/D/L | Score Rate tổng |
| ----------------- | ------------------------: | -------------------------: | -----------------: | --------------: |
| Keep vs Heuristic |           20/1/19; 51,25% |            16/3/21; 43,75% |        **36/4/40** |      **47,50%** |
| Drop vs Heuristic |           16/1/23; 41,25% |            16/0/24; 40,00% |        **32/1/47** |      **40,63%** |
| Drop vs Keep      |           19/1/20; 48,75% |            23/3/14; 61,25% |        **42/4/34** |      **55,00%** |

- Keep hơn Drop **6,875 điểm phần trăm** (≈ 6,88 điểm) về Score Rate trước Heuristic, nhưng **Drop thắng trong đối đầu trực tiếp** với Keep (55,00%).
- Chính sách Drop **chưa tạo cải thiện nhất quán** qua các phép đánh giá. So sánh chưa cô lập hoàn toàn tác động của nhãn xung đột vì mẫu dữ liệu và tập Validation khác nhau; mỗi mô hình chỉ được huấn luyện một seed.
- Hai checkpoint Ablation chỉ dùng để nghiên cứu, **không thay thế** checkpoint CNN V4.1 đại diện trong GUI/Round-robin.

## Round-robin chính thức — 5 tác tử, 1.000 trận

**Đây là giải đấu giữa 5 _nhóm tác tử_, không phải giải đấu 8 phiên bản CNN.** Tác tử đại diện: Random, Heuristic, Minimax/Alpha-Beta Depth 2, MCTS/UCT 64 simulations, CNN V4.1 Weighted (`checkpoints/cnn_v4_1_weighted_final.pt`).

- Script: `scripts/evaluate_round_robin.py`; **10 cặp × 100 trận/cặp = 1.000 trận**.
- Mỗi cặp có **50 trận/seed**, seed **2026 và 3026**; `opening_plies=8`; hoán đổi quân Đen/Trắng theo cặp từ trạng thái mở đầu tương ứng.
- Mỗi tác tử tham gia **400 trận**, gồm 200 trận cầm quân Đen và 200 trận cầm quân Trắng.
- CSV kết quả: `results/round_robin/round_robin_100.csv`.
- CSV xếp hạng: `results/round_robin/round_robin_100_ranking.csv`.
- Replay: chương trình ghi nhận đã lưu **1.000 file JSON** trong `game_logs/round_robin_100/` khi bật `--save-replays`. Việc lưu đủ file theo log **không đồng nghĩa** đã kiểm tra phát lại độc lập toàn bộ 1.000 file.

**Lệnh đã sử dụng:**

```powershell
python -m scripts.evaluate_round_robin --checkpoint checkpoints/cnn_v4_1_weighted_final.pt --depth 2 --mcts-simulations 64 --games-per-seed 50 --seeds 2026 3026 --opening-plies 8 --out results/round_robin/round_robin_100.csv --save-replays
```

### Kết quả cả 10 cặp theo từng seed

`W/D/L` và Score Rate đều tính theo **tác tử ghi trước** trong cặp đấu. Mỗi cột seed gồm **50 trận**, cột tổng gồm **100 trận**.

| Cặp đối kháng (A vs B)  | Seed 2026 W/D/L; Score Rate A | Seed 3026 W/D/L; Score Rate A | Tổng 100 trận W/D/L của A | Score Rate A |
| ----------------------- | ----------------------------: | ----------------------------: | ------------------------: | -----------: |
| Random vs Heuristic     |                6/0/44; 12,00% |                8/1/41; 17,00% |                   14/1/85 |       14,50% |
| Random vs Minimax D2    |                5/1/44; 11,00% |                 3/0/47; 6,00% |                    8/1/91 |        8,50% |
| Random vs MCTS 64       |                 0/0/50; 0,00% |                 3/1/46; 7,00% |                    3/1/96 |        3,50% |
| Random vs CNN V4.1      |                4/2/44; 10,00% |                8/0/42; 16,00% |                   12/2/86 |       13,00% |
| Heuristic vs Minimax D2 |                8/2/40; 18,00% |                7/1/42; 15,00% |                   15/3/82 |       16,50% |
| Heuristic vs MCTS 64    |               15/2/33; 32,00% |               10/2/38; 22,00% |                   25/4/71 |       27,00% |
| Heuristic vs CNN V4.1   |               29/2/19; 60,00% |               25/3/22; 53,00% |                   54/5/41 |       56,50% |
| Minimax D2 vs MCTS 64   |               19/0/31; 38,00% |               21/0/29; 42,00% |                   40/0/60 |       40,00% |
| Minimax D2 vs CNN V4.1  |                43/0/7; 86,00% |                43/0/7; 86,00% |                   86/0/14 |       86,00% |
| MCTS 64 vs CNN V4.1     |               39/1/10; 79,00% |               38/1/11; 77,00% |                   77/2/21 |       78,00% |

### Bảng xếp hạng cuối giải đấu

|  Hạng | Tác tử         |   W |   D |   L | Tổng trận | Win Rate | Score Rate theo log |
| ----: | -------------- | --: | --: | --: | --------: | -------: | ------------------: |
| **1** | **MCTS 64**    | 304 |   7 |  89 |       400 |   76,00% |          **76,88%** |
| **2** | **Minimax D2** | 299 |   4 |  97 |       400 |   74,75% |          **75,25%** |
| **3** | **Heuristic**  | 179 |  13 | 208 |       400 |   44,75% |          **46,38%** |
| **4** | **CNN V4.1**   | 162 |   9 | 229 |       400 |   40,50% |          **41,62%** |
| **5** | **Random**     |  37 |   5 | 358 |       400 |    9,25% |           **9,88%** |

**Kiểm tra tổng:** 1.000 trận = **981 trận có bên thắng + 19 trận hòa**; cộng tất cả các lượt tham gia cho năm tác tử bằng **2.000** (5 × 400).

**Riêng CNN V4.1 trong Round-robin** (mỗi đối thủ 100 trận; ghi W/D/L và Score Rate theo V4.1):

| Đối thủ    | W/D/L của CNN V4.1 |          Score Rate |
| ---------- | -----------------: | ------------------: |
| Random     |            86/2/12 |              87,00% |
| Heuristic  |            41/5/54 |              43,50% |
| Minimax D2 |            14/0/86 |              14,00% |
| MCTS 64    |            21/2/77 |              22,00% |
| **Tổng**   |      **162/9/229** | **41,62% theo log** |

**Lưu ý về số làm tròn:** Score Rate chính xác của CNN V4.1 là `(162 + 0,5 × 9)/400 × 100 = 41,625%`. Terminal ghi **41,62%**; nếu làm tròn hai chữ số bằng `ROUND_HALF_UP`, giá trị sẽ là **41,63%**. Tài liệu này **giữ 41,62% theo log**, kèm giá trị chính xác để tránh nhầm lẫn; tuyệt đối không làm thay đổi W/D/L.

**Giới hạn diễn giải:** MCTS đứng đầu với 76,88%, Minimax đứng thứ hai với 75,25%; chênh lệch tổng hợp chỉ khoảng 1,63 điểm phần trăm. MCTS thắng Minimax **60/100 trận đối đầu trực tiếp**, nhưng hai phương pháp có ngân sách tính toán khác nhau. Kết quả chỉ phản ánh cấu hình nêu trên, không xác lập thứ hạng tuyệt đối trong mọi điều kiện. CNN V4.1 xếp thứ tư trên năm tác tử, không phải AI mạnh nhất toàn hệ thống.

## Kết luận sau khi bổ sung thực nghiệm

- **CNN V4.1** tiếp tục là phiên bản CNN đại diện được chọn cho GUI và giải đấu, do có kết quả tốt hơn V4.0 trong các phép thử nội bộ (60,63% khi đối đầu trực tiếp) và hơn V5 khi đối đầu trực tiếp (V5 đạt 36,25% trước V4.1). Trước Heuristic trên 80 trận, **V4.1 và V2 cùng đạt 46,25%**, còn V5 đạt 45,00%; không kết luận V4.1 đứng đầu tất cả phép thử CNN.
- **Ablation Keep/Drop** xác nhận phương án lọc xung đột nhãn không tạo cải thiện nhất quán. Sự hiện diện của nhãn xung đột đã đo được, nhưng chưa chứng minh đó là nguyên nhân trực tiếp khiến CNN yếu.
- **Round-robin** xếp MCTS 64 thứ nhất, Minimax D2 thứ hai, Heuristic thứ ba, CNN V4.1 thứ tư và Random thứ năm trong đúng cấu hình đã kiểm tra. Không suy ra kết quả này áp dụng cho mọi seed, ngân sách tính toán hoặc phiên bản CNN.
- Validation Loss, Accuracy và mức độ lặp/xung đột của dữ liệu là các chỉ số mô tả quá trình học, **không được dùng riêng lẻ để suy ra hiệu quả chiến thuật**. Tất cả số liệu ở file này được tổng hợp từ log người thực hiện đã cung cấp; không có thí nghiệm mới nào được chạy trong lúc cập nhật file.
