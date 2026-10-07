# Phase E — Công việc của người phụ trách Dataset và Global/Local

## 1. Mục tiêu

Phụ trách đưa subset `aiface_pilot_32k` lên Kaggle, xác nhận fixed
generator-disjoint split và chạy các baseline DINOv3 làm mốc so sánh cho forensic
branch.

Phần việc này trả lời ba câu hỏi:

1. DINOv3 Global-only hoạt động thế nào trên hai generator chưa thấy?
2. Local-only có cải thiện so với Global-only hay không?
3. Global + Local có bổ sung thực sự trên unseen generators hay chỉ tốt hơn trong
   in-domain setting?

## 2. Dataset và split bắt buộc

Dataset dùng chung:

```text
aiface_pilot_32k
```

Generator mapping:

| Split | Fake generators | Real |
| --- | --- | --- |
| Train | `taming_transformer_VQGAN`, `stylegan3`, `StableDiffusion1.5`, `latent_diffusion` | `imdb_wiki` |
| Validation | `AttGAN`, `Palette` | `imdb_wiki` |
| Test unseen | `STARGAN`, `StableDiffusion_Inpainting` | `imdb_wiki` |

Không được dùng split image-level mặc định của AI-Face. Không được để test unseen
tham gia chọn checkpoint, threshold hoặc hyperparameter.

## 3. Việc cần thực hiện

### 3.1. Upload và xác nhận Kaggle

Upload archive:

```text
kaggle_upload/aiface_pilot_32k_archive/aiface_pilot_32k.tar
```

Sau khi chạy KRun, kiểm tra log có:

```text
Extracted dataset archive
Split mode: generator_disjoint_fixed
train generators: ...
val generators: ...
test generators: ...
test manifest: ...
```

Nếu log có `in_domain_random_fallback`, run đó chỉ là smoke test và không được dùng
cho báo cáo unseen-generator.

Kiểm tra thêm:

- manifest train/val/test đều tồn tại;
- mọi image path trong manifest đều đọc được;
- test generator không overlap với train hoặc validation;
- `subset_metadata.json` được đọc thành công;
- test manifest được giữ riêng.

### 3.2. Chạy baseline

Chạy ba cấu hình trên cùng fixed split:

1. `Global-only`: frozen DINOv3 global feature + classifier nhỏ.
2. `Local-only`: frozen DINOv3 patch feature + pooling cố định + classifier nhỏ.
3. `Global + Local`: late probability fusion của hai nhánh độc lập.

Giữ cố định:

- checkpoint DINOv3;
- image processor, resize/crop và normalization;
- seed;
- classifier và training budget;
- checkpoint selection rule;
- threshold selection rule.

Threshold chỉ được chọn trên validation generators.

## 4. Artifact phải lưu

Mỗi cấu hình lưu riêng:

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

`run_metadata.json` phải ghi dataset slug/version, manifest checksum, split mode,
generator mapping, seed, device, preprocessing, checkpoint rule, threshold rule,
fusion rule và Kaggle URL/run ID.

## 5. Metric và bàn giao

Lưu riêng cho:

- `STARGAN`;
- `StableDiffusion_Inpainting`;
- macro-average của hai generator.

Metric chính:

- AUROC;
- Balanced Accuracy.

Metric bổ sung:

- F1;
- Accuracy;
- confusion matrix;
- số lượng ảnh real/fake.

Bàn giao cho teammate:

- fixed test manifest hoặc checksum manifest;
- ba file prediction có cùng `image_id`;
- config preprocessing;
- threshold validation;
- checkpoint và resolved config;
- bảng metric theo generator.

## 6. Shortcut diagnostic

Nếu script đã sẵn sàng, chạy Logistic Regression bằng các feature đơn giản:

- file size;
- luminance;
- saturation;
- gray-pixel fraction;
- edge statistics;
- resolution/format.

Train shortcut trên train split và đánh giá trên validation. Nếu AUROC cao bất thường,
ghi rõ trong artifact và không diễn giải kết quả chính như bằng chứng forensic trước
khi phân tích shortcut.

## 7. Điều kiện hoàn thành phần việc

Phần việc được xem là hoàn tất khi:

- Kaggle run dùng `generator_disjoint_fixed`;
- ba baseline đã chạy thành công;
- có prediction cho toàn bộ test images;
- có metric riêng cho hai unseen generators và macro-average;
- artifact có config, checkpoint, metadata và Kaggle URL;
- prediction format sẵn sàng để ghép với forensic branch.

Sau khi teammate bàn giao forensic predictions, dùng cùng test IDs để chạy fusion,
rescue/harm và cập nhật bảng Phase E.
