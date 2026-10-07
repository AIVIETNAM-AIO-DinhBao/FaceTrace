# Báo cáo Phase E — AI-Face Global/Local Baseline

**Ngày ghi nhận:** 2026-10-07
**Trạng thái:** Baseline Global/Local đã chạy full; Phase E tổng thể còn thiếu nhánh Forensic và các fusion có Forensic.

## 1. Mục tiêu và câu hỏi nghiên cứu

Phase E là pilot để kiểm tra khả năng tổng quát hóa sang generator chưa xuất hiện trong quá trình huấn luyện. Câu hỏi đang được kiểm định là:

> Local representation có cải thiện so với Global DINOv3 hay không, và việc kết hợp các biểu diễn bổ sung có giúp mô hình nhận diện ảnh từ unseen generators tốt hơn không?

Trong run này mới có ba cấu hình:

1. `Global-only` — đặc trưng toàn ảnh từ DINOv3 đóng băng.
2. `Local-only` — trung bình các patch features của DINOv3 đóng băng.
3. `Global + Local` — late probability fusion, lấy trung bình hai xác suất dự đoán.

Nhánh Forensic phổ tần số chưa được đưa vào run này; vì vậy báo cáo chưa kết luận được đóng góp của Forensic.

## 2. Dataset và thiết kế split

Dataset Kaggle dùng trong run:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

Kaggle notebook:

<https://www.kaggle.com/code/nguyentrann0703/ai-face-phase-e-global-local-baseline?scriptVersionId=356092097>

Run metadata xác nhận:

| Thuộc tính | Giá trị |
|---|---|
| Split mode | `generator_disjoint_fixed` |
| Seed | `42` |
| Smoke | `false` |
| Test rows | `8,000` |
| Image size | `224 x 224` |
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Backbone training | Frozen; chỉ train classifier probe |
| Checkpoint selection | Validation AUROC, sau đó validation Balanced Accuracy |
| Threshold selection | Validation Balanced Accuracy → F1 → gần 0.5 nhất |
| Fusion | Mean của xác suất Global và Local |
| Device | CUDA |

### 2.1. Generator mapping

| Split | Real | Fake generators | Tổng ảnh |
|---|---:|---|---:|
| Train | 8,000 | `StableDiffusion1.5` (2,000), `latent_diffusion` (2,000), `stylegan3` (2,000), `taming_transformer_VQGAN` (2,000) | 16,000 |
| Validation | 4,000 | `AttGAN` (2,000), `Palette` (2,000) | 8,000 |
| Test unseen | 4,000 | `STARGAN` (2,000), `StableDiffusion_Inpainting` (2,000) | 8,000 |

Test generators không giao với train hoặc validation. Test manifest không được dùng để chọn checkpoint, threshold hoặc siêu tham số. Random fallback bị tắt.

Lưu ý: bảng metric theo từng fake generator dùng chung 4,000 ảnh real của test cho mỗi generator, nên mỗi dòng có `n_images = 6,000`; đây không phải là 12,000 ảnh test độc lập.

## 3. Phương pháp

### 3.1. Global-only

Ảnh được đưa qua image processor cố định của DINOv3. Global representation được đưa vào MLP classifier nhỏ. DINOv3 không được cập nhật trong quá trình train.

### 3.2. Local-only

Patch representations từ cùng backbone DINOv3 được mean-pool cố định theo patch, sau đó đưa vào classifier riêng. Cấu hình preprocessing và training budget giữ nguyên như Global-only.

### 3.3. Global + Local

Hai classifier được huấn luyện độc lập. Sau đó xác suất fake trên test được kết hợp theo late probability fusion:

```text
p_fusion = (p_global + p_local) / 2
```

Threshold của fusion được chọn trên validation, không chọn trên test.

## 4. Kết quả trên unseen generators

### 4.1. Metric macro theo hai unseen generators

Các giá trị AUROC và Balanced Accuracy dưới đây là trung bình của metric trên `STARGAN` và `StableDiffusion_Inpainting`.

| Model | Macro AUROC | Macro Balanced Accuracy | Macro Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Global-only | 0.939942 | 0.857625 | 0.868833 | 0.801033 |
| Local-only | **0.959689** | **0.883625** | 0.879333 | **0.829744** |
| Global + Local | 0.958677 | 0.877500 | **0.886917** | 0.828201 |

### 4.2. AUROC theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.893704 | 0.986179 |
| Local-only | 0.925259 | 0.994119 |
| Global + Local | 0.923204 | 0.994150 |

### 4.3. Balanced Accuracy theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.784125 | 0.931125 |
| Local-only | **0.836125** | 0.931125 |
| Global + Local | 0.808875 | **0.946125** |

### 4.4. Threshold validation

| Model | Validation AUROC | Validation BAcc | Chosen threshold |
|---|---:|---:|---:|
| Global-only | 0.955553 | 0.886750 | 0.000002 |
| Local-only | 0.955251 | 0.883125 | 0.000889 |
| Global + Local | 0.959663 | 0.891125 | 0.001555 |

Threshold rất thấp vì xác suất đầu ra của probe chưa được calibration về mức 0–1 đồng đều; threshold vẫn được chọn theo validation đúng protocol. Không được so sánh trực tiếp các threshold này như một thước đo độ tin cậy giữa các model.

## 5. Diễn giải kết quả

1. `Local-only` là baseline mạnh nhất theo macro AUROC (0.959689) và macro BAcc (0.883625) trong ba cấu hình hiện có.
2. `Global + Local` cải thiện so với `Global-only` về macro AUROC: `+0.018736`, và macro BAcc: `+0.019875`.
3. Tuy nhiên `Global + Local` vẫn thấp hơn `Local-only` khoảng `0.001011` AUROC và `0.006125` BAcc. Vì vậy chưa có bằng chứng rằng việc thêm Global vào Local tạo ra cải thiện trên pilot này.
4. `STARGAN` khó hơn `StableDiffusion_Inpainting` đối với cả ba cấu hình. Chênh lệch này cho thấy không nên chỉ báo cáo một metric gộp mà bỏ qua từng generator.
5. Kết quả hiện tại chỉ chứng minh mốc Global/Local trên hai generator được giữ ngoài train. Chưa thể kết luận Forensic có bổ sung thông tin hay không.

## 6. Shortcut diagnostic

Shortcut diagnostic được chạy trên train/validation bằng Logistic Regression. Feature được fit trên train và đánh giá trên validation gồm:

- `log_file_size_bytes`;
- mean và standard deviation của luminance;
- mean saturation;
- gray-pixel fraction;
- edge mean.

| Shortcut model | Train AUROC | Validation AUROC | Validation BAcc |
|---|---:|---:|---:|
| File size only | 0.780115 | 0.537581 | 0.747625 |
| Simple features | 0.860288 | 0.600553 | 0.638750 |

Validation AUROC của shortcut không cao đến mức thay thế được model chính, nhưng vẫn cho thấy các đặc trưng đơn giản có tín hiệu phân biệt nhất định. Vì vậy kết quả Global/Local cần được diễn giải thận trọng; chưa thể xem toàn bộ hiệu năng là bằng chứng thuần túy của semantic hoặc local forensic representation.

## 7. Artifact và khả năng tái lập

Artifact full run được tải về tại thư mục local:

```text
/tmp/aiface_phase_e_full.M74oba/FaceTrace/outputs/aiface_phase_e
```

Các artifact chính:

```text
run_metadata.json
config_resolved.yaml
environment.json
manifests/{train,val,test}.csv
{global_only,local_only,global_local}/
  checkpoint.pt
  config_resolved.yaml
  val_metrics.json
  test_metrics_overall.json
  test_metrics_by_generator.csv
  predictions_test.csv
  confusion_matrix.csv
shortcut/
  metrics.csv
  predictions.csv
  validation_predictions.csv
  feature_table.csv
  coefficients.csv
  scaler_and_protocol.json
```

Mỗi prediction test có các trường `image_id`, `label`, `generator_id`, `split`, `probability` và `predicted_label`, nên có thể dùng chung `image_id` để ghép với prediction của teammate.

Các entry point trong repository:

- `configs/aiface_phase_e.yaml`;
- `scripts/run_aiface_phase_e.py`;
- `scripts/run_aiface_phase_e_kaggle.py`;
- `notebooks/10_aiface_phase_e_baseline.ipynb`;
- `src/evaluation/aiface_phase_e.py`.

`run_metadata.json` của artifact chưa tự ghi Kaggle URL/run ID; provenance Kaggle được ghi thủ công trong mục 2 bằng URL notebook và script version ở trên.

## 8. Giới hạn

- Đây là pilot, mỗi fake generator có 2,000 ảnh; không đại diện cho toàn bộ AI-Face.
- Real images trong pilot chỉ dùng `imdb_wiki`, nên có nguy cơ real-source shortcut.
- Generator-disjointness được đảm bảo theo manifest; identity-disjointness chưa được chứng minh đầy đủ từ metadata.
- Chỉ có hai unseen generators, nên chưa thể khái quát cho mọi GAN hoặc diffusion generator.
- Chưa chạy seed 43/44, robustness hoặc confidence interval.
- Chưa có `Forensic-only`, `Local + Forensic` và `Global + Local + Forensic`; do đó câu hỏi về giá trị bổ sung của nhánh forensic vẫn đang bỏ ngỏ.

## 9. Trạng thái hoàn thành

### Đã hoàn tất

- Fixed generator-disjoint split và held-out test manifest.
- Full run trên Kaggle với `smoke: false`.
- Global-only, Local-only và Global + Local.
- Metric overall, metric theo từng unseen generator, prediction và checkpoint.
- Shortcut diagnostic trên train/validation.

### Chưa hoàn tất

- Teammate bàn giao spectral Forensic predictions.
- Fusion `Local + Forensic` và `Global + Local + Forensic`.
- Rescue/harm/net gain giữa các cấu hình.
- Seed 43/44 và robustness.

Vì vậy, kết luận đúng ở thời điểm báo cáo là: **Owner Phase E baseline đã hoàn tất; Phase E đầy đủ vẫn ở trạng thái partially complete cho đến khi nhánh Forensic và các fusion tương ứng được chạy trên cùng test manifest.**
