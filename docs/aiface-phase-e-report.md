# Báo cáo Phase E — AI-Face Global/Local và Spectral Forensic

**Ngày ghi nhận:** 2026-10-07
**Ngày cập nhật:** 2026-10-08
**Trạng thái:** Pilot Phase E seed 42 đã có đủ sáu cấu hình full, hai fusion có Forensic và rescue/harm. Seed 43/44, robustness, confidence interval và bổ sung provenance Saved Version vẫn là các bước tiếp theo.

**Cập nhật:** Đã kiểm tra trực tiếp cả `facetrace_spectral_artifacts.zip` và `facetrace_fusion_artifacts.zip`. ZIP fusion chứa prediction/metric baseline đầu vào; metric test của cả sáu cấu hình, threshold validation của hai fusion và rescue/harm đã được tính lại, khớp artifact. Manifest checksum, generator mapping, seed, image size và test ID/label/split giữa các nhánh đã được đối chiếu.

## 1. Mục tiêu và câu hỏi nghiên cứu

Phase E là pilot để kiểm tra khả năng tổng quát hóa sang generator chưa xuất hiện trong quá trình huấn luyện. Câu hỏi đang được kiểm định là:

> Local representation có cải thiện so với Global DINOv3 hay không, và việc kết hợp các biểu diễn bổ sung có giúp mô hình nhận diện ảnh từ unseen generators tốt hơn không?

Các full run đã có kết quả cho sáu cấu hình:

1. `Global-only` — đặc trưng toàn ảnh từ DINOv3 đóng băng.
2. `Local-only` — trung bình các patch features của DINOv3 đóng băng.
3. `Global + Local` — late probability fusion, lấy trung bình hai xác suất dự đoán.
4. `Forensic-only` — radial FFT trên RGB và MLP nhỏ, không dùng DINOv3.
5. `Local + Forensic` — trung bình xác suất Local và Forensic.
6. `Global + Local + Forensic` — trung bình xác suất ba nhánh.

Global/Local chạy ở notebook 10; Forensic chạy độc lập ở notebook 11; notebook 12 thực hiện hai fusion và rescue/harm từ artifact đã lưu, không train lại. Báo cáo kiểm định đóng góp của Forensic trong pilot seed 42, không suy rộng thành kết luận cho toàn bộ AI-Face.

## 2. Dataset và thiết kế split

Dataset Kaggle dùng trong run:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

Kaggle notebook:

<https://www.kaggle.com/code/nguyentrann0703/ai-face-phase-e-global-local-baseline?scriptVersionId=356092097>

Metadata baseline đã được đối chiếu với `input_evidence/baseline_run_metadata.json` trong ZIP fusion:

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

Artifact Forensic xác nhận dataset version `1`, seed `42`, `smoke: false`, kích thước `224 x 224`, train/val/test `16,000/8,000/8,000` và cùng generator mapping bên dưới. Đã xác minh metadata baseline và Forensic có cùng checksum cả ba manifest, seed, generator mapping và image size; prediction test của cả sáu cấu hình có cùng `8,000` ID, label, generator và split. Validation/test có ID không giao nhau. URL Saved Version notebook 11/12 chưa được cung cấp; `kaggle_url` và `kaggle_run_id` trong metadata đang là `null`.

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

### 3.5. Fusion có Forensic và protocol đánh giá

Notebook 12 chạy CPU, đọc các prediction theo `image_id` và kiểm tra ID/label/generator/split cùng metadata. Hai cấu hình dùng trọng số bằng nhau, không học trọng số fusion:

```text
p_local_forensic = (p_local + p_forensic) / 2
p_global_local_forensic = (p_global + p_local + p_forensic) / 3
```

Validation Global/Local được runner khôi phục từ frozen `feature_cache.pt` và classifier checkpoint vì baseline chưa xuất CSV validation. Threshold fusion được chọn trên `8,000` validation predictions theo BAcc → F1 → gần `0.5` nhất, rồi khóa trước đánh giá `8,000` test predictions. Không đọc lại ảnh, trích DINOv3/FFT hay train classifier trong notebook 12.

Rescue/harm chính so `Local + Forensic` với `Local-only`, và `Global + Local + Forensic` với `Global + Local`. Mỗi model dùng threshold riêng đã chọn trên validation, đúng với metric test chính. Một bảng phụ dùng threshold chung `0.5` để chẩn đoán; không trộn bảng phụ với kết quả threshold validation.

## 4. Kết quả trên unseen generators

### 4.1. Metric macro theo hai unseen generators

Các giá trị AUROC và Balanced Accuracy dưới đây là trung bình của metric trên `STARGAN` và `StableDiffusion_Inpainting`.

| Model | Macro AUROC | Macro Balanced Accuracy | Macro Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Global-only | 0.939942 | 0.857625 | 0.868833 | 0.801033 |
| Local-only | 0.959689 | 0.883625 | 0.879333 | 0.829744 |
| Global + Local | 0.958677 | 0.877500 | 0.886917 | 0.828201 |
| Forensic-only | 0.958515 | 0.898750 | 0.902333 | 0.858245 |
| Local + Forensic | **0.981896** | 0.935625 | 0.938000 | 0.908781 |
| Global + Local + Forensic | 0.981608 | **0.940375** | **0.942417** | **0.915116** |

Các hàng lấy từ `test_metrics_overall.json` của từng cấu hình trong ZIP fusion, đã tính lại từ prediction test. Manifest/ID đã đồng nhất; so sánh vẫn chỉ thuộc một seed và chưa có confidence interval. `comparison.csv` dùng macro AUROC/BAcc nhưng **overall** Accuracy/F1; bảng này dùng đủ bốn metric **macro** nên Accuracy/F1 không lấy trực tiếp từ hai cột overall của file đó.

### 4.2. AUROC theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.893704 | 0.986179 |
| Local-only | 0.925259 | 0.994119 |
| Global + Local | 0.923204 | **0.994150** |
| Forensic-only | 0.965205 | 0.951825 |
| Local + Forensic | **0.974467** | 0.989325 |
| Global + Local + Forensic | 0.972129 | 0.991087 |

### 4.3. Balanced Accuracy theo generator

| Model | STARGAN | StableDiffusion_Inpainting |
|---|---:|---:|
| Global-only | 0.784125 | 0.931125 |
| Local-only | 0.836125 | 0.931125 |
| Global + Local | 0.808875 | 0.946125 |
| Forensic-only | 0.910250 | 0.887250 |
| Local + Forensic | 0.922625 | 0.948625 |
| Global + Local + Forensic | **0.924750** | **0.956000** |

### 4.4. Threshold validation

| Model | Validation AUROC | Validation BAcc | Validation Accuracy | Validation F1 | Chosen threshold |
|---|---:|---:|---:|---:|---:|
| Global-only | 0.955553 | 0.886750 | 0.886750 | 0.886750 | 0.000002 |
| Local-only | 0.955251 | 0.883125 | 0.883125 | 0.885459 | 0.000889 |
| Global + Local | 0.959663 | 0.891125 | 0.891125 | 0.889928 | 0.001555 |
| Forensic-only | 0.961039 | 0.901125 | 0.901125 | 0.900365 | 0.473062 |
| Local + Forensic | 0.981139 | 0.935875 | 0.935875 | 0.935528 | 0.313337 |
| Global + Local + Forensic | **0.982985** | **0.941125** | **0.941125** | **0.940940** | 0.220576 |

Threshold Global/Local rất thấp so với Forensic, nhưng đều được chọn theo validation đúng protocol. Không được so sánh trực tiếp threshold như một thước đo độ tin cậy giữa các model. Chưa có kiểm định calibration trong báo cáo; hiệu quả equal-probability fusion được đo từ artifact thật ở đây, không suy ra từ riêng threshold. Threshold chính xác của hai fusion là `0.3133365660905838` và `0.2205758609731371`; chọn lại từ CSV validation cho kết quả khớp.

Với Forensic, `train_log.csv` ghi BAcc tại threshold `0.5` (epoch 13: `0.900250`), còn `val_metrics.json` ghi BAcc sau chọn threshold `0.473062...` (`0.901125`); hai giá trị khác nhau là do threshold, không phải đổi split.

### 4.5. Metric overall trên 8,000 ảnh test

| Model | AUROC | Balanced Accuracy | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Global-only | 0.939942 | 0.857625 | 0.857625 | 0.852671 |
| Local-only | 0.959689 | 0.883625 | 0.883625 | 0.885104 |
| Global + Local | 0.958677 | 0.877500 | 0.877500 | 0.873939 |
| Forensic-only | 0.958515 | 0.898750 | 0.898750 | 0.897650 |
| Local + Forensic | **0.981896** | 0.935625 | 0.935625 | 0.935163 |
| Global + Local + Forensic | 0.981608 | **0.940375** | **0.940375** | **0.940008** |

Ví dụ Forensic có overall F1 `0.897650` khác macro F1 `0.858245`, vì macro là trung bình theo hai nhóm generator có dùng chung real pool. Không dùng overall F1 của một cấu hình để so trực tiếp với macro F1 của cấu hình khác.

### 4.6. Accuracy/F1 theo generator

| Model | STARGAN Accuracy | STARGAN F1 | StableDiffusion_Inpainting Accuracy | StableDiffusion_Inpainting F1 |
|---|---:|---:|---:|---:|
| Global-only | 0.819833 | 0.714700 | 0.917833 | 0.887366 |
| Local-only | 0.847667 | 0.778155 | 0.911000 | 0.881333 |
| Global + Local | 0.841167 | 0.749277 | 0.932667 | 0.907126 |
| Forensic-only | 0.910000 | 0.870937 | 0.894667 | 0.845552 |
| Local + Forensic | 0.929333 | 0.894893 | 0.946667 | 0.922668 |
| Global + Local + Forensic | **0.932000** | **0.898507** | **0.952833** | **0.931725** |

### 4.7. Rescue/harm chính: threshold validation riêng của từng model

`rescue` = baseline sai nhưng fusion đúng; `harm` = baseline đúng nhưng fusion sai; `net_gain = rescue - harm`. Cột `both_correct`/`both_wrong` là hai model cùng đúng/cùng sai. Baseline so sánh là Local-only cho fusion hai nhánh, Global + Local cho fusion ba nhánh.

| Fusion | Nhóm | N | Rescue | Harm | Net gain | Both correct | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| Local + Forensic | Overall | 8,000 | 798 | 382 | **+416** | 6,687 | 133 |
| Local + Forensic | STARGAN | 6,000 | 787 | 297 | +490 | 4,789 | 127 |
| Local + Forensic | StableDiffusion_Inpainting | 6,000 | 452 | 238 | +214 | 5,228 | 82 |
| Global + Local + Forensic | Overall | 8,000 | 806 | 303 | **+503** | 6,717 | 174 |
| Global + Local + Forensic | STARGAN | 6,000 | 794 | 249 | +545 | 4,798 | 159 |
| Global + Local + Forensic | StableDiffusion_Inpainting | 6,000 | 314 | 193 | +121 | 5,403 | 90 |

Net gain overall tương ứng tăng Accuracy/BAcc `416/8,000 = 0.052000` so với Local-only và `503/8,000 = 0.062875` so với Global + Local; Accuracy bằng BAcc ở overall vì test cân bằng real/fake. Không cộng hai hàng generator để tính overall: 4,000 real test được dùng chung ở cả hai hàng. File `error_cases.csv` giữ ID/probability/threshold và nhóm lỗi của từng ảnh để truy vết; các nhóm đã được tính lại từ prediction CSV và khớp.

### 4.8. Rescue/harm phụ: threshold chung 0.5

| Fusion | Nhóm | N | Rescue | Harm | Net gain | Both correct | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| Local + Forensic | Overall | 8,000 | 872 | 63 | +809 | 5,901 | 1,164 |
| Local + Forensic | STARGAN | 6,000 | 716 | 14 | +702 | 4,307 | 963 |
| Local + Forensic | StableDiffusion_Inpainting | 6,000 | 167 | 54 | +113 | 5,573 | 206 |
| Global + Local + Forensic | Overall | 8,000 | 299 | 43 | +256 | 5,712 | 1,946 |
| Global + Local + Forensic | STARGAN | 6,000 | 182 | 10 | +172 | 4,193 | 1,615 |
| Global + Local + Forensic | StableDiffusion_Inpainting | 6,000 | 121 | 38 | +83 | 5,503 | 338 |

Bảng này lấy từ `rescue_harm_fixed_0.5.csv`, chỉ là phân tích phụ. Các baseline có threshold validation rất xa 0.5, nên không thay các net gain chính `+416/+503` bằng `+809/+256` khi diễn giải metric ở mục 4.1/4.5.

## 5. Diễn giải kết quả

1. `Local-only` vẫn là baseline mạnh nhất theo macro AUROC (0.959689) và macro BAcc (0.883625) trong ba cấu hình Global/Local. Trong bốn cấu hình chưa fusion với Forensic, Local có AUROC cao nhất và Forensic có BAcc cao nhất; cả hai fusion có Forensic vượt các mốc này.
2. `Global + Local` cải thiện so với `Global-only` về macro AUROC: `+0.018736`, và macro BAcc: `+0.019875`.
3. Tuy nhiên `Global + Local` vẫn thấp hơn `Local-only` khoảng `0.001011` AUROC và `0.006125` BAcc. Vì vậy chưa có bằng chứng rằng việc thêm Global vào Local tạo ra cải thiện trên pilot này.
4. `STARGAN` khó hơn `StableDiffusion_Inpainting` đối với cả ba baseline Global/Local. Forensic có xu hướng ngược lại: AUROC/BAcc trên STARGAN cao hơn trên StableDiffusion_Inpainting. Không nên chỉ báo cáo metric gộp mà bỏ qua từng generator.
5. So với Local-only, Forensic cao hơn `0.015125` macro BAcc (**1.5125 điểm phần trăm**), nhưng thấp hơn khoảng `0.001174` macro AUROC. Đây không phải bằng chứng một model tốt hơn ở mọi tiêu chí, cũng chưa phải cải thiện có ý nghĩa thống kê.
6. `Local + Forensic` đạt macro AUROC **0.981896**, cao hơn Local-only khoảng `0.022207`, và BAcc **0.935625**, tăng **5.2000 điểm phần trăm**. Rescue `798` lớn hơn harm `382`, net gain **416 ảnh** trên cùng test manifest. Đây là bằng chứng thực nghiệm về giá trị bổ sung của Forensic khi fusion với Local trong pilot này.
7. `Global + Local + Forensic` đạt macro AUROC **0.981608**, tăng khoảng `0.022931` so với Global + Local; BAcc **0.940375**, tăng **6.2875 điểm phần trăm**. Rescue `806`, harm `303`, net gain **503 ảnh** so với baseline hai nhánh.
8. Fusion hai nhánh có AUROC cao nhất; fusion ba nhánh có BAcc/Accuracy/F1 cao nhất. Thêm Global vào Local + Forensic tăng BAcc `0.004750` (**0.4750 điểm phần trăm**, 38 ảnh đúng thêm overall), nhưng giảm AUROC khoảng `0.000288`. Không có cấu hình tốt nhất ở mọi metric.
9. Hai fusion cải thiện BAcc trên cả STARGAN và StableDiffusion_Inpainting so với baseline đối chiếu tương ứng. Tuy nhiên AUROC diffusion của các fusion vẫn thấp hơn Local-only/Global + Local. Net gain dương không có nghĩa không gây lỗi mới, và một seed/hai unseen generators chưa chứng minh cải thiện có ý nghĩa thống kê hay tổng quát cho mọi generator.

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

### 7.3. Artifact fusion đã kiểm tra

Nguồn kết quả: `D:/UNI_STUDY/AIO/AI_is_AI/facetrace_fusion_artifacts.zip`; ZIP giữ ngoài Git, không chứa lại dataset ảnh hay feature cache DINOv3 lớn.

- SHA256 ZIP: `4b399ca468566a62adb180a55af805e69fd58f9a05ec5248d299e3482810eca0`.
- Source commit của notebook 12: `eda43be00bb7338b707074f17daa2cfba2cced1c`; device **CPU**.
- Timestamp Local + Forensic: `2026-10-07T16:52:21.627061+00:00`; Global + Local + Forensic: `2026-10-07T16:52:21.846041+00:00`.
- Input Forensic: `/kaggle/input/datasets/ntklinhfitus/facetrace-spectral-artifacts/forensic_only`.
- Input baseline: `/kaggle/input/datasets/nguyentrann0703/phase-e-full-artifacts/aiface_phase_e_full_artifacts`.
- SHA256 metadata baseline: `830e19046e93e3b829086964eb87291d458743121571bfee09f61aeaec741260`; metadata Forensic: `0d27a691f147b8ce45dbdee8781be1d37d0ba59f02ad1b0217b161c722ed8666`.
- Environment: Python `3.13.15`, torch `2.11.0+cpu`, NumPy `2.1.3`, pandas `2.3.3`, scikit-learn `1.6.1`, PyYAML `6.0.3`; GPU `null`.
- `input_tests.log`: 4/4 test pass; `fusion.log` xác nhận cả hai fusion hoàn tất.

ZIP có:

```text
comparison.csv, comparison_by_generator.csv
fusion_inputs.json, notebook_session.json
fusion.log, input_tests.log
{local_forensic,global_local_forensic}/
  config_resolved.yaml, checkpoint.pt, train_log.csv
  predictions_val.csv, predictions_test.csv
  val_metrics.json, test_metrics_overall.json, test_metrics_by_generator.csv
  confusion_matrix.csv, environment.json, run_metadata.json
  error_cases.csv, rescue_harm.csv, rescue_harm_fixed_0.5.csv
input_evidence/
  baseline_run_metadata.json
  {global_only,local_only,global_local,forensic_only}/...
source/{src,scripts,configs,tests}/...
```

Mỗi fusion có `8,000` prediction validation và `8,000` prediction test. `train_log.csv` chỉ có header vì fusion không training; `checkpoint.pt` fusion lưu recipe/threshold/tham chiếu input, không phải checkpoint một MLP mới. Phần `input_evidence` giữ prediction test, metric và cấu hình/metadata có sẵn của các nhánh để truy vết, nhưng không đóng gói lại feature cache và classifier checkpoint baseline; tái tạo validation baseline từ đầu vẫn cần giữ dataset artifact baseline gốc.

Kiểm tra trực tiếp đã hoàn tất:

- Metadata: cả ba manifest SHA256, seed `42`, generator mapping, fixed split và image size `224` trùng nhau; hai input không phải smoke. Metadata/source SHA256 trong fusion khớp các file evidence/snapshot.
- Prediction: đủ `8,000` ID duy nhất, cùng test label/generator/split cho sáu cấu hình; validation/test ID không giao. Prediction/metadata Forensic trong ZIP fusion khớp byte-for-byte ZIP notebook 11.
- Xác suất test: Global + Local, Local + Forensic và Global + Local + Forensic khớp mean của các nhánh tương ứng sau ghép ID.
- Metric: tính lại overall/macro và per-generator test cho sáu cấu hình; tính lại metric validation và chọn lại threshold cho hai fusion; các kết quả khớp JSON/CSV. Metric validation baseline được ghi từ JSON evidence, không claim đã chạy lại feature cache baseline ở local.
- Rescue/harm: tính lại mọi nhóm overall/per-generator tại threshold validation và common `0.5`; `error_cases.csv` và count/net gain khớp. Net gain overall khớp chênh lệch số ảnh dự đoán đúng.

Các Input artifact version và URL Saved Version notebook 11/12 chưa được lưu trong metadata; cần bổ sung khi bàn giao. Các checksum ZIP/metadata/source đã ghi là bằng chứng hiện có, không thay bằng URL hoặc version suy đoán.

## 8. Giới hạn

- Đây là pilot, mỗi fake generator có 2,000 ảnh; không đại diện cho toàn bộ AI-Face.
- Real images trong pilot chỉ dùng `imdb_wiki`, nên có nguy cơ real-source shortcut.
- Generator-disjointness được đảm bảo theo manifest; identity-disjointness chưa được chứng minh đầy đủ từ metadata.
- Chỉ có hai unseen generators, nên chưa thể khái quát cho mọi GAN hoặc diffusion generator.
- Chưa chạy seed 43/44, robustness hoặc confidence interval.
- Hai fusion có Forensic và rescue/harm cho bằng chứng bổ sung trên pilot này, nhưng chỉ với equal-probability mean và threshold validation đã khóa; chưa đánh giá calibration hay weighted fusion.
- Single-bin Forensic không có `subset_metadata.json`; metadata ghi rõ không biết original sampling seed và chưa đọc quality gate gốc. Manifest/class counts/generator mapping được xác minh, không được suy ra đã kiểm định toàn bộ quality gate gốc.
- Checksum/ID giữa các nhánh đã được đối chiếu; URL Saved Version notebook 11/12 và version của hai dataset artifact Input vẫn cần bổ sung để hoàn thiện provenance.

## 9. Trạng thái hoàn thành

### Đã hoàn tất

- Fixed generator-disjoint split và held-out test manifest.
- Full run trên Kaggle với `smoke: false`.
- Global-only, Local-only và Global + Local.
- Forensic-only full run; đã kiểm tra artifact, prediction validation/test và checkpoint.
- Fusion `Local + Forensic` và `Global + Local + Forensic` trên CPU, threshold chọn từ validation.
- Đối chiếu metadata/checksum/ID/label/generator giữa các nhánh; xác minh recipe fusion và tính lại metric từ CSV.
- Rescue/harm/net gain tổng thể và theo từng generator; có bảng riêng threshold chung 0.5 và error cases theo ID.
- Metric overall, metric theo từng unseen generator, prediction và checkpoint.
- Shortcut diagnostic trên train/validation.

### Bước tiếp theo / provenance còn cần bổ sung

- Bổ sung URL Saved Version notebook 11/12 và version của các dataset artifact Input.
- Seed 43/44, robustness và confidence interval; các mục này chưa được chạy trong pilot seed 42.
- Giữ full artifact baseline gốc để tái tạo validation từ feature cache/checkpoint; ZIP fusion hiện giữ compact evidence chứ không thay thế toàn bộ baseline Output.

Kết luận ở thời điểm cập nhật: **Pilot Phase E seed 42 đã có đủ sáu cấu hình, metric và rescue/harm trên cùng test manifest; phần thực nghiệm Forensic/fusion của teammate đã hoàn tất.** Forensic bổ sung giá trị cho Global/Local trong run này: Local + Forensic có AUROC cao nhất; Global + Local + Forensic có BAcc/Accuracy/F1 cao nhất. Chưa kết luận có ý nghĩa thống kê hoặc tổng quát ngoài pilot; còn cần hoàn thiện provenance và các thí nghiệm multi-seed/robustness.
