# Tuần 2: Audit dữ liệu, baseline và đặc trưng cục bộ

## Mục tiêu

1. Khảo sát nguồn ảnh, phân bố real/fake, resolution, JPEG, crop và các shortcut tiềm năng.
2. Chốt split source-disjoint; giữ một hoặc nhiều nguồn/generator hoàn toàn ngoài train.
3. Xây dựng mốc Frozen DINOv3 + global feature + linear/MLP classifier.
4. So sánh Global-only với Global + Local (patch feature từ cùng DINOv3), giữ các yếu tố khác cố định.

## Thứ tự làm việc

- Ghi lại manifest và metadata cần thiết trong `00_dataset_audit.ipynb`.
- Trích xuất/kiểm tra global và patch representation trong `01_dinov3_feature_extraction.ipynb`.
- Chạy baseline, chọn cấu hình chỉ bằng train/validation trong `02_global_baseline.ipynb`.
- Đánh giá local pooling đã chốt trong `03_local_experiment.ipynb`.

## Đánh giá và đầu ra

AUROC và Balanced Accuracy là chỉ số chính; Accuracy và F1 là bổ sung. Báo cáo kết quả trên validation cùng nguồn và test nguồn chưa thấy riêng biệt. Lưu seed, nguồn trong mỗi split, checkpoint/model ID, config và lỗi dự đoán. Không điều chỉnh threshold dựa trên tập test.

Đầu ra tuần: báo cáo audit, manifest/split có thể tái lập, baseline global hoàn chỉnh và so sánh Global vs Global + Local.
