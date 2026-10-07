# Phase E — Công việc của teammate: Forensic Spectrum Branch

## 1. Mục tiêu

Implement và đánh giá một forensic branch độc lập dựa trên phổ tần số. Nhánh này
nhằm kiểm tra liệu thông tin phổ có bổ sung cho representation Global/Local của
DINOv3 trên generator chưa xuất hiện trong train hay không.

Đây là baseline frequency-domain có kiểm soát, không phải reproduction đầy đủ của
SPAI, AIDE hoặc một paper cụ thể.

## 2. Pipeline bắt buộc

```text
RGB
→ fixed resize/crop
→ Hann window
→ 2D FFT
→ log magnitude
→ radial averaging
→ MLP nhỏ
→ probability
```

Hann window được dùng để giảm spectral leakage do biên crop. Radial averaging chuyển
phổ 2D thành vector năng lượng theo bán kính, giúp giảm số tham số và tránh để MLP
học trực tiếp vị trí giả trong ma trận phổ.

Không dùng TinyCNN residual của pilot Who Is AI làm forensic branch chính cho AI-Face.

## 3. Dataset và protocol dùng chung

Dùng đúng dataset và fixed split:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

| Split | Fake generators |
| --- | --- |
| Train | `taming_transformer_VQGAN`, `stylegan3`, `StableDiffusion1.5`, `latent_diffusion` |
| Validation | `AttGAN`, `Palette` |
| Test unseen | `STARGAN`, `StableDiffusion_Inpainting` |

Phải dùng chung với baseline:

- image IDs;
- resize/crop;
- label mapping;
- seed policy;
- threshold chọn trên validation;
- format prediction.

## 4. File cần tạo

Tạo các file riêng, không sửa trực tiếp code baseline Global/Local:

```text
src/models/spectral.py
src/input_data/spectral_dataset.py
src/training/spectral_trainer.py
scripts/train_spectral.py
scripts/predict_spectral.py
configs/spectral.yaml
tests/test_spectral.py
```

Các thành phần tối thiểu:

- transform tính FFT ổn định trên batch;
- Hann window được tạo theo kích thước input cố định;
- radial bins xác định và ghi trong config;
- xử lý `log1p` hoặc epsilon để tránh `log(0)`;
- MLP nhỏ với parameter budget được ghi rõ;
- checkpoint selection dựa trên validation AUROC;
- prediction giữ nguyên `image_id` và thứ tự manifest.

Interface cần thống nhất:

```text
predict(image) -> probability
predict_manifest(manifest) -> predictions.csv
```

## 5. Test và smoke test

### 5.1. Unit test

Kiểm tra:

- output FFT không có NaN/Inf;
- radial profile có kích thước cố định;
- Hann window có đúng shape và range hợp lệ;
- batch size khác nhau cho cùng output dimension;
- model forward trả probability hợp lệ;
- dataset giữ đúng label và `image_id`.

### 5.2. Smoke test

Chạy trên subset nhỏ trước khi gửi Kaggle:

- đọc được train/val/test manifest;
- forward một batch thành công;
- loss giảm hoặc training loop hoàn tất vài bước;
- prediction CSV có đủ image IDs;
- không có duplicate hoặc missing IDs.

## 6. Các cấu hình cần chạy

1. **Forensic-only**

   Đo khả năng độc lập của phổ tần số.

2. **Local + Forensic**

   Ghép prediction của Local baseline với Forensic prediction bằng late probability
   fusion. Dùng cùng test images và cùng threshold policy.

3. **Global + Local + Forensic**

   Ghép ba nhánh để kiểm tra forensic có bổ sung thêm ngoài pipeline DINOv3 hay không.

Fusion mặc định:

```text
p_fusion = mean(p_branch_1, p_branch_2, ...)
```

Không chọn trọng số bằng test set. Weighted fusion chỉ được thử nếu trọng số được chọn
trên validation và ghi vào resolved config.

## 7. Artifact phải bàn giao

Mỗi run lưu:

```text
outputs/aiface_phase_e/<model_name>/
├── config_resolved.yaml
├── train_log.csv
├── val_metrics.json
├── test_metrics_overall.json
├── test_metrics_by_generator.csv
├── predictions_test.csv
├── confusion_matrix.csv
├── checkpoint.pt
├── environment.json
└── run_metadata.json
```

`predictions_test.csv` phải có schema:

```text
image_id,label,probability,generator_id,split
```

Bàn giao tối thiểu:

- source code;
- `configs/spectral.yaml` đã resolve;
- unit-test output;
- smoke-test log;
- checkpoint tốt nhất;
- predictions của Forensic-only;
- prediction của các fusion;
- metric theo `STARGAN` và `StableDiffusion_Inpainting`;
- macro AUROC/BAcc;
- environment và dataset/manifest checksum.

## 8. Kiểm tra rescue/harm

Sau khi nhận prediction baseline từ người phụ trách, tính trên đúng cùng test images:

- `Local-only → Local + Forensic`;
- `Global + Local → Global + Local + Forensic`.

Với mỗi cặp, lưu:

- rescue: baseline sai, candidate đúng;
- harm: baseline đúng, candidate sai;
- net gain = rescue − harm;
- kết quả tổng thể;
- kết quả tách theo từng unseen generator.

Không đổi threshold riêng cho test và không loại bỏ các ảnh khó khỏi phân tích.

## 9. Điều kiện hoàn thành phần việc

Phần việc được xem là hoàn tất khi:

- Radial FFT + MLP forward/training smoke test pass;
- unit tests pass;
- Forensic-only đã chạy trên fixed split;
- predictions có đủ và đúng thứ tự test IDs;
- metrics có theo từng unseen generator và macro-average;
- checkpoint/config/environment đã lưu;
- fusion predictions có thể ghép trực tiếp với baseline;
- không thay đổi protocol hoặc code baseline Global/Local.
