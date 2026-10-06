# Kết quả Tuần 3

## Cấu hình chạy

- Kaggle job: `nguyentrann0703/facetrace-20261001-180511-91bbbb`
- Dataset: bản mirror private `nguyentrann0703/facetrace-who-is-ai`
- Thiết bị: CUDA/T4
- Split: in-domain stratified cố định 80/20, seed 42
- Số epoch: 20 cho mỗi nhánh
- Artifact: `outputs/week03_kaggle_20261001/`

## Metric Validation

| Nhánh | Epoch tốt nhất | AUROC | Balanced Accuracy | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Residual-only TinyCNN | 19 | 0.798700 | 0.682500 | 0.682500 | 0.739220 |
| RGB-control TinyCNN | 20 | 0.715525 | 0.647500 | 0.647500 | 0.687361 |

Trên split này, Residual-only cao hơn RGB-control tương ứng `+0.083175`
AUROC và `+0.035000` Balanced Accuracy.

## Fusion Sạch Với Kết Quả Tuần 2

Sau khi nhận artifact Phase 5, tất cả các nhánh được kiểm tra có cùng 400 ID
validation và cùng nhãn. Fusion xác suất với trọng số bằng nhau cho kết quả:

| Biểu diễn | AUROC | Balanced Accuracy |
|---|---:|---:|
| Global-only | 0.930025 | 0.8400 |
| Local-only | 0.952550 | 0.8825 |
| Global + Local (nối vector) | 0.945550 | 0.8725 |
| Global + Local (trung bình xác suất) | 0.951350 | 0.8825 |
| Global + Residual (trung bình xác suất) | 0.927200 | 0.8450 |
| Local + Residual (trung bình xác suất) | **0.953300** | **0.8925** |
| Global + Local + Residual (trung bình xác suất) | 0.954225 | 0.8825 |
| Local + RGB-control (trung bình xác suất) | 0.948500 | 0.8800 |

Trên split sạch duy nhất này, `Local + Residual` cao hơn Local-only
`+0.000750` AUROC và `+0.0100` Balanced Accuracy. Fusion ba nhánh có AUROC
cao nhất (`0.954225`) nhưng không cải thiện Balanced Accuracy tại threshold cố
định `0.5`.

## Robustness Của Các Nhánh Tuần 3

Checkpoint đã lưu được chạy lại trên cùng 400 ảnh validation. Corruption được
áp dụng trước bước tiền xử lý residual/RGB:

| Nhánh / điều kiện | AUROC | Balanced Accuracy |
|---|---:|---:|
| Residual sạch | 0.798700 | 0.6825 |
| Residual JPEG70 | 0.793250 | 0.7125 |
| Residual resize112 | 0.617863 | 0.5000 |
| Residual blur1 | 0.782500 | 0.5000 |
| RGB-control sạch | 0.715525 | 0.6475 |
| RGB-control JPEG70 | 0.715050 | 0.6475 |
| RGB-control resize112 | 0.701700 | 0.6075 |
| RGB-control blur1 | 0.709600 | 0.6400 |

Ở các mức corruption đã thử, resize gây suy giảm mạnh nhất cho nhánh Residual
(delta AUROC `-0.1808375`). JPEG70 chỉ làm AUROC thay đổi nhẹ. Balanced
Accuracy tại threshold cố định có thể thay đổi độc lập với AUROC. Đây là các
sanity check corruption trong-domain, không phải unseen-generator test.

Artifact Phase 6 của notebook 04 được tải từ `dinhbaobao/04-error-robustness`
vào `outputs/week02_phase6_artifacts/`. Kết quả Global/Local cũng cho thấy
resize là corruption gây suy giảm lớn nhất: AUROC Local-only giảm từ `0.952550`
xuống `0.925675`; JPEG70 giảm xuống `0.950525` và blur1 xuống `0.948275`.

Metric fusion cố định ở các điều kiện được lưu tại
`outputs/week03_kaggle_20261001/analysis/robustness_seed42/all_branch_fusion_metrics.csv`.
Ví dụ, Local + Residual đạt AUROC/BAcc `0.951625/0.8950` trên JPEG70, nhưng
giảm còn `0.923675/0.5825` trên resize112 và `0.950300/0.7775` trên blur1.
Các kết quả này dùng checkpoint được chọn trên dữ liệu sạch và threshold `0.5`;
không tối ưu riêng theo từng corruption.

## Error Rescue/Harm Của Residual

Notebook `08_residual_error_analysis.ipynb` và script
`scripts/run_residual_error_analysis.py` đã chạy trên đúng 400 ảnh validation,
join theo `image_id`, kiểm tra nhãn nhất quán và dùng threshold cố định `0.5`.
Artifact chính nằm trong
`outputs/week03_kaggle_20261001/analysis/clean_seed42/error_rescue_harm/`.

| Comparison | Rescue | Harm | Net gain | Baseline AUROC | Candidate AUROC |
|---|---:|---:|---:|---:|---:|
| Global → Residual | 35 | 98 | -63 | 0.930025 | 0.798700 |
| Local → Residual | 24 | 104 | -80 | 0.952550 | 0.798700 |
| Local-only → Local + Residual | 9 | 5 | **+4** | 0.952550 | 0.953300 |
| Global + Local → Global + Local + Residual | 5 | 5 | 0 | 0.951350 | 0.954225 |

Định nghĩa: `rescue` là baseline sai nhưng candidate đúng; `harm` là baseline
đúng nhưng candidate sai. Kết quả cho thấy residual-only chưa phải nhánh thay
thế cho Global/Local, nhưng khi fusion với Local thì có bổ sung nhỏ trên split
in-domain này. Các group đầy đủ (`both_correct`, `rescue`, `harm`, `both_wrong`)
được lưu trong `error_rescue_harm.csv`; từng image và xác suất được lưu trong
`error_cases.csv`; montage kiểm tra định tính nằm trong `montages/`.

Comparison cuối cho thấy thêm Residual vào fusion Global + Local cải thiện AUROC
nhưng không đổi Balanced Accuracy tại threshold `0.5`: có 5 rescue và 5 harm,
nên net gain bằng `0`.

## Giới Hạn Diễn Giải

Kết quả cho thấy phép biến đổi residual kết hợp TinyCNN có giá trị trên split
in-domain hiện tại. Tuy nhiên, đây chưa phải bằng chứng về khả năng tổng quát
sang generator chưa thấy: dataset không có nhãn source/generator đáng tin cậy,
chỉ chạy một seed cho thí nghiệm residual và chưa có test external hoặc
source-disjoint.

RGB-control là control có cùng kiểu kiến trúc với TinyCNN, không phải baseline
DINOv3 Global/Local của Tuần 2. Vì vậy kết quả residual trả lời câu hỏi
Residual branch so với control tương ứng, không thay thế được so sánh model của
Tuần 2.
