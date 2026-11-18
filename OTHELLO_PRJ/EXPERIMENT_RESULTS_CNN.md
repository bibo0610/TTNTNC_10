# Kết quả thực nghiệm CNN — Othello 8×8

**Nguồn:** log chạy huấn luyện/đánh giá mà nhóm đã cung cấp trong quá trình phát triển; checkpoint và dataset tương ứng được lưu trong các thư mục checkpoints/ và data/ của project hiện tại. Các số Score Rate dưới đây dựa trên log, **không phải kết quả chạy lại toàn bộ giải đấu trong bản rà soát**.

## Thiết lập đánh giá

- Mỗi seed (42, 123) có 40 trận, 20 random openings, hoán đổi màu đen/trắng; `opening_plies=8`.
- `W/D/L` tính theo CNN đứng trước; `Score Rate = (W + 0,5 × D) / N`.
- Đối đầu hai model CNN không hoàn toàn thay thế được đánh giá trước một baseline chung.
- Hai seed và các cặp màu cùng opening chưa đủ để khẳng định sức mạnh tổng quát hay ý nghĩa thống kê.

## So sánh với Heuristic

| CNN      | Seed 42 W/D/L | Score Rate | Seed 123 W/D/L | Score Rate | Tổng 80 trận Score Rate |
| -------- | ------------: | ---------: | -------------: | ---------: | ----------------------: |
| V1       |       12/1/27 |     31,25% |        chưa có |          — |                       — |
| V2       |       16/2/22 |     42,50% |        chưa có |          — |                       — |
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

**Kết luận có điều kiện:** V4.1 là lựa chọn trình diễn CNN được ưu tiên vì vượt V4.0 trong đối đầu, có hiệu suất trước Heuristic tốt hơn V4.0 và V5 trong các phép thử hiện tại. V5 giảm số tham số nhưng không chứng tỏ sức mạnh vượt V4.1. V2 vẫn là baseline đáng giữ. Mô hình CNN còn yếu khi so với Minimax depth 2. Không diễn giải cải thiện loss là cải thiện chiến thuật hay suy ra khả năng tổng quát cho mọi trạng thái.
