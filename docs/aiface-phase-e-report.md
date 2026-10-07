# Báo cáo Phase E — AI-Face Global/Local và Spectral Forensic

**Ngày ghi nhận:** 2026-10-07
**Trạng thái:** Global-only, Local-only, Global + Local và Forensic-only đã chạy full; Phase E tổng thể còn thiếu các fusion có Forensic và rescue/harm.

**Cập nhật:** Bổ sung kết quả Forensic-only từ ZIP artifact notebook 11 đã kiểm tra trực tiếp. Các giá trị Global/Local giữ nguyên từ báo cáo baseline; chưa nhận artifact baseline để đối chiếu checksum/ID giữa hai run hoặc chạy fusion.

## 1. Mục tiêu và câu hỏi nghiên cứu

Phase E là pilot để kiểm tra khả năng tổng quát hóa sang generator chưa xuất hiện trong quá trình huấn luyện. Câu hỏi đang được kiểm định là:

> Local representation có cải thiện so với Global DINOv3 hay không, và việc kết hợp các biểu diễn bổ sung có giúp mô hình nhận diện ảnh từ unseen generators tốt hơn không?

Các full run đã có kết quả cho bốn cấu hình:

1. `Global-only` — đặc trưng toàn ảnh từ DINOv3 đóng băng.
2. `Local-only` — trung bình các patch features của DINOv3 đóng băng.
3. `Global + Local` — late probability fusion, lấy trung bình hai xác suất dự đoán.
4. `Forensic-only` — radial FFT trên RGB và MLP nhỏ, không dùng DINOv3.

Global/Local chạy ở notebook 10; Forensic chạy độc lập ở notebook 11. Chưa có kết quả `Local + Forensic` hoặc `Global + Local + Forensic`, nên chưa kết luận được giá trị bổ sung của Forensic khi fusion.

## 2. Dataset và thiết kế split

Dataset Kaggle dùng trong run:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

Kaggle notebook:

<https://www.kaggle.com/code/nguyentrann0703/ai-face-phase-e-global-local-baseline?scriptVersionId=356092097>

Metadata baseline được ghi nhận trong báo cáo ban đầu:

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

Artifact Forensic xác nhận dataset version `1`, seed `42`, `smoke: false`, kích thước `224 x 224`, train/val/test `16,000/8,000/8,000` và cùng generator mapping bên dưới. Đây là sự phù hợp theo cấu hình đã báo cáo, chưa thay thế việc đối chiếu trực tiếp manifest checksum/ID với artifact baseline trước fusion. URL saved version của notebook 11 chưa được cung cấp; `kaggle_url` và `kaggle_run_id` trong artifact đang là `null`.

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

### 3.4. Forensic-only: radial FFT + MLP

Ảnh được EXIF-transpose, chuyển RGB, rescale về `[0,1]` và resize square bilinear antialias `224 x 224`; không crop, augmentation hay ImageNet normalization. Trên mỗi kênh RGB: Hann đối xứng → FFT2 → fftshift → `log1p(abs(spectrum))` → 64 radial bins từ tâm đến góc phổ. Ghép ba kênh thành vector 192 chiều. Binning dùng quy tắc số nguyên `integer_squared_edges_v1` để tránh khác biệt làm tròn giữa môi trường.

Mean/std feature chỉ được fit trên train và lưu trong checkpoint. Classifier là MLP `192 → 128 → 2`, GELU, dropout `0`, có `24,962` tham số trainable; không dùng DINOv3, DCT, DWT hay DTCWT trong run này.

Run full dùng seed `42`, 20 epoch, batch MLP `32`, learning rate `0.001`, weight decay `0.0001`, device CUDA. Checkpoint được chọn bằng validation AUROC, rồi BAcc tại threshold `0.5`; checkpoint tốt nhất ở epoch **13**. Sau đó chọn threshold trên validation theo BAcc → F1 → gần `0.5` nhất, được **`0.4730623960494995`**. Test chỉ được đánh giá sau khi checkpoint và threshold đã khóa.

## 4. Kết quả trên unseen generators

### 4.1. Metric macro theo hai unseen generators

Các giá trị AUROC và Balanced Accuracy dưới đây là trung bình của metric trên `STARGAN` và `StableDiffusion_Inpainting`.

| Model | Macro AUROC | Macro Balanced Accuracy | Macro Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Global-only | 0.939942 | 0.857625 | 0.868833 | 0.801033 |
| Local-only | **0.959689** | 0.883625 | 0.879333 | 0.829744 |
| Global + Local | 0.958677 | 0.877500 | 0.886917 | 0.828201 |
| Forensic-only | 0.958515 | **0.898750** | **0.902333** | **0.858245** |

Hàng Forensic lấy từ `forensic_only/test_metrics_overall.json` trong ZIP. Các so sánh này là mô tả hai full run; việc đồng nhất manifest giữa hai artifact chưa được kiểm tra trực tiếp và chưa có confidence interval.

### 4.2. AUROC theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.893704 | 0.986179 |
| Local-only | 0.925259 | 0.994119 |
| Global + Local | 0.923204 | 0.994150 |
| Forensic-only | **0.965205** | 0.951825 |

### 4.3. Balanced Accuracy theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.784125 | 0.931125 |
| Local-only | 0.836125 | 0.931125 |
| Global + Local | 0.808875 | **0.946125** |
| Forensic-only | **0.910250** | 0.887250 |

### 4.4. Threshold validation

| Model | Validation AUROC | Validation BAcc | Chosen threshold |
|---|---:|---:|---:|
| Global-only | 0.955553 | 0.886750 | 0.000002 |
| Local-only | 0.955251 | 0.883125 | 0.000889 |
| Global + Local | 0.959663 | 0.891125 | 0.001555 |
| Forensic-only | **0.961039** | **0.901125** | 0.473062 |

Threshold Global/Local rất thấp so với Forensic, nhưng đều được chọn theo validation đúng protocol. Không được so sánh trực tiếp threshold như một thước đo độ tin cậy giữa các model. Chưa có kiểm định calibration trong báo cáo; khác biệt thang xác suất cần được lưu ý khi diễn giải equal-probability fusion, không tự suy ra fusion sẽ tốt hơn.

Với Forensic, `train_log.csv` ghi BAcc tại threshold `0.5` (epoch 13: `0.900250`), còn `val_metrics.json` ghi BAcc sau chọn threshold `0.473062...` (`0.901125`); hai giá trị khác nhau là do threshold, không phải đổi split.

### 4.5. Metric overall của Forensic trên 8,000 ảnh test

| AUROC | Balanced Accuracy | Accuracy | F1 |
|---:|---:|---:|---:|
| 0.958515 | 0.898750 | 0.898750 | 0.897650 |

Overall F1 khác macro F1 (`0.858245`) vì macro là trung bình theo hai nhóm generator có dùng chung real pool. Không dùng overall F1 của Forensic để so trực tiếp với macro F1 của baseline.

## 5. Diễn giải kết quả

1. `Local-only` là baseline mạnh nhất theo macro AUROC (0.959689) và macro BAcc (0.883625) trong ba cấu hình Global/Local; trong cả bốn cấu hình, Local vẫn có AUROC cao nhất nhưng Forensic có BAcc cao nhất.
2. `Global + Local` cải thiện so với `Global-only` về macro AUROC: `+0.018736`, và macro BAcc: `+0.019875`.
3. Tuy nhiên `Global + Local` vẫn thấp hơn `Local-only` khoảng `0.001011` AUROC và `0.006125` BAcc. Vì vậy chưa có bằng chứng rằng việc thêm Global vào Local tạo ra cải thiện trên pilot này.
4. `STARGAN` khó hơn `StableDiffusion_Inpainting` đối với cả ba baseline Global/Local. Forensic có xu hướng ngược lại: AUROC/BAcc trên STARGAN cao hơn trên StableDiffusion_Inpainting. Không nên chỉ báo cáo metric gộp mà bỏ qua từng generator.
5. So với Local-only, Forensic cao hơn `0.015125` macro BAcc (**1.5125 điểm phần trăm**), nhưng thấp hơn khoảng `0.001174` macro AUROC. Đây không phải bằng chứng một model tốt hơn ở mọi tiêu chí, cũng chưa phải cải thiện có ý nghĩa thống kê.
6. Forensic mạnh hơn Local trên STARGAN nhưng yếu hơn trên StableDiffusion_Inpainting. Điều này gợi ý các nhánh có thế mạnh khác nhau theo generator; chưa chứng minh lỗi của chúng bổ sung cho nhau ở mức từng ảnh. Cần fusion và rescue/harm theo ID trên cùng manifest để kiểm tra giá trị bổ sung thực tế.

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

### 7.1. Baseline Global/Local

Artifact baseline full run được ghi nhận ở môi trường của owner tại:

```text
/tmp/aiface_phase_e_full.M74oba/FaceTrace/outputs/aiface_phase_e
```

Các artifact chính:

```text
run_metadata.json
config_resolved.yaml
environment.json
feature_cache.pt
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

### 7.2. Artifact Forensic đã kiểm tra

Nguồn kết quả: `D:/UNI_STUDY/AIO/AI_is_AI/facetrace_spectral_artifacts.zip` trên máy teammate; ZIP không được commit vào Git.

- SHA256 ZIP: `62d02ff5b202b5679a60bbf7fb068e5d01a198a62d1cb5dd83fc4527b726f93b`.
- Source commit của run: `187374ce70037afaa72eb7289f056d53c2117d8a`, `git_dirty: false`.
- Timestamp metadata: `2026-10-07T16:05:08.878650+00:00`.
- Dataset version: `1`; archive dataset SHA256: `58e37a0af568476724aad2357cc162f9f93f259f5cbc24a694ac4d1a4003bfd1`.
- `forensic_only/predictions_val.csv`: `8,000` dòng; `predictions_test.csv`: `8,000` dòng.
- Đã tính lại AUROC/BAcc/Accuracy/F1 validation và test (overall/macro test) từ CSV: khớp JSON artifact; ID trong từng split không trùng và validation/test không giao nhau.
- `unit_tests.log`: 10/10 test pass. Smoke riêng có `40/24/24` ảnh train/val/test và `passed: true`; metric smoke không dùng trong các bảng nghiên cứu.

Manifest SHA256 gốc ghi trong metadata Forensic:

```text
train: 691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88
val:   65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac
test:  b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf
```

Thư mục `forensic_only/` có `checkpoint.pt`, `config_resolved.yaml`, `run_metadata.json`, `environment.json`, `train_log.csv`, prediction validation/test, metric validation/test, confusion matrix, feature caches và bản manifest đã dùng. ZIP còn giữ log và snapshot source. Thư mục `forensic_only_smoke/` không phải kết quả full.

Các entry point Forensic/fusion: `notebooks/11_aiface_phase_e_spectral.ipynb`, `notebooks/12_aiface_phase_e_fusion.ipynb`, `scripts/train_spectral.py`, `scripts/run_aiface_fusion_kaggle.py`, `scripts/fuse_aiface_spectral.py`. Xem [hướng dẫn Kaggle](aiface-spectral-kaggle-guide.md).

Notebook 12 đã được chuẩn bị nhưng **chưa có kết quả fusion trên artifact thật**. Cần full baseline Output có `feature_cache.pt` và checkpoint Global/Local để khôi phục prediction validation, hoặc CSV validation tương ứng; runner kiểm tra manifest checksum, seed, geometry và ID/label/generator/split trước fusion. Hai cấu hình dự kiến là `(p_local + p_forensic)/2` và `(p_global + p_local + p_forensic)/3`, threshold chọn trên validation; không có số liệu fusion được suy diễn từ bảng metric hiện tại.

## 8. Giới hạn

- Đây là pilot, mỗi fake generator có 2,000 ảnh; không đại diện cho toàn bộ AI-Face.
- Real images trong pilot chỉ dùng `imdb_wiki`, nên có nguy cơ real-source shortcut.
- Generator-disjointness được đảm bảo theo manifest; identity-disjointness chưa được chứng minh đầy đủ từ metadata.
- Chỉ có hai unseen generators, nên chưa thể khái quát cho mọi GAN hoặc diffusion generator.
- Chưa chạy seed 43/44, robustness hoặc confidence interval.
- Forensic có hiệu năng standalone tốt nhưng chưa có `Local + Forensic`, `Global + Local + Forensic` hoặc rescue/harm, nên câu hỏi về giá trị bổ sung khi kết hợp vẫn đang bỏ ngỏ.
- Single-bin Forensic không có `subset_metadata.json`; metadata ghi rõ không biết original sampling seed và chưa đọc quality gate gốc. Manifest/class counts/generator mapping được xác minh, không được suy ra đã kiểm định toàn bộ quality gate gốc.
- So sánh hai run hiện dựa trên baseline report và ZIP Forensic; đối chiếu checksum/ID với artifact baseline và URL saved version Forensic vẫn cần hoàn tất.

## 9. Trạng thái hoàn thành

### Đã hoàn tất

- Fixed generator-disjoint split và held-out test manifest.
- Full run trên Kaggle với `smoke: false`.
- Global-only, Local-only và Global + Local.
- Forensic-only full run; đã kiểm tra artifact, prediction validation/test và checkpoint.
- Metric overall, metric theo từng unseen generator, prediction và checkpoint.
- Shortcut diagnostic trên train/validation.

### Chưa hoàn tất

- Nhận full baseline artifact để đối chiếu manifest checksum/ID với Forensic.
- Bổ sung URL Saved Version của notebook 11 để hoàn thiện provenance.
- Fusion `Local + Forensic` và `Global + Local + Forensic`.
- Rescue/harm/net gain giữa các cấu hình.
- Seed 43/44 và robustness.

Vì vậy, kết luận đúng ở thời điểm cập nhật là: **Global/Local baseline và Spectral Forensic standalone đã hoàn tất; Phase E đầy đủ vẫn partially complete, chờ đối chiếu artifact, hai fusion và rescue/harm trên cùng test manifest.**
