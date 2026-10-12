# AI-Face Phase F — Multi-seed, Robustness, Confidence Interval và Model Selection

## 1. Mục tiêu

Phase F mở rộng pilot AI-Face Phase E seed 42 để kiểm tra ba vấn đề:

1. Kết quả có ổn định khi thay đổi seed huấn luyện hay không.
2. Các nhánh Global, Local và Spectral Forensic chịu được các biến đổi ảnh phổ biến đến mức nào.
3. Cấu hình nào nên được giữ làm model cuối mà không dùng test unseen để tuning.

Phase F vẫn dùng cùng bài toán `real vs fully AI-generated face`, cùng fixed
generator-disjoint split và cùng preprocessing của Phase E. Không đổi dataset,
generator mapping hoặc test IDs giữa các seed.

## 2. Phạm vi và nguyên tắc khóa trước khi chạy

### 2.1. Dataset và split

Dataset: `nguyentrann0703/aiface-pilot-32k-single-bin`.

Split ID cố định:

```text
split_id: generator_disjoint_fixed_v1
split_seed: 42

Train: VQGAN, stylegan3, StableDiffusion1.5, latent_diffusion
Val:   AttGAN, Palette
Test:  STARGAN, StableDiffusion_Inpainting
```

Các file `train.csv`, `val.csv` và `test.csv` của seed 42 phải được tái sử dụng
nguyên vẹn. `run_seed` chỉ điều khiển khởi tạo/training order, không được tạo lại
split.

### 2.2. Sáu cấu hình được báo cáo

Mỗi run seed chỉ train ba branch độc lập:

```text
Global-only      -> p_global
Local-only       -> p_local
Forensic-only    -> p_forensic
```

Ba cấu hình còn lại được tạo bằng late probability fusion, không train thêm neural
network:

```text
Global + Local            = mean(p_global, p_local)
Local + Forensic          = mean(p_local, p_forensic)
Global + Local + Forensic = mean(p_global, p_local, p_forensic)
```

Do đó mỗi seed có sáu thư mục kết quả nhưng chỉ có ba checkpoint branch cần train.
Không ghép branch khác seed, ví dụ Global seed 43 với Forensic seed 44.

### 2.3. Quy tắc checkpoint và threshold

Đối với mỗi branch:

1. Train trên `train.csv`.
2. Sau mỗi epoch đánh giá `val.csv`.
3. Chọn checkpoint theo thứ tự:
   - validation AUROC cao hơn;
   - nếu hòa, validation Balanced Accuracy cao hơn;
   - nếu vẫn hòa, epoch sớm hơn.
4. Sau khi checkpoint đã khóa, chọn threshold trên validation theo:
   - Balanced Accuracy cao hơn;
   - nếu hòa, F1 cao hơn;
   - nếu vẫn hòa, threshold gần `0.5` hơn.

Fusion cũng chọn threshold riêng trên fused validation probabilities theo cùng quy
tắc. Không chọn checkpoint, threshold hoặc công thức fusion bằng nhãn test.

## 3. Seed 43 và 44

### 3.1. Ma trận chạy

| Seed | Notebook | Nội dung | Output riêng |
|---:|---|---|---|
| 43 | `10_aiface_phase_e_baseline.ipynb` | Global + Local | `outputs/aiface_phase_e_seed43_baseline/` |
| 43 | `11_aiface_phase_e_spectral.ipynb` | Forensic-only | `outputs/aiface_phase_e_seed43_spectral/` |
| 43 | `12_aiface_phase_e_fusion.ipynb` | Ba late fusions | `outputs/aiface_phase_e_seed43_fusion/` |
| 44 | `10_aiface_phase_e_baseline.ipynb` | Global + Local | `outputs/aiface_phase_e_seed44_baseline/` |
| 44 | `11_aiface_phase_e_spectral.ipynb` | Forensic-only | `outputs/aiface_phase_e_seed44_spectral/` |
| 44 | `12_aiface_phase_e_fusion.ipynb` | Ba late fusions | `outputs/aiface_phase_e_seed44_fusion/` |

Notebook 09 audit không cần chạy lại. Notebook 05/06 là pipeline Residual/Who Is
AI cũ, không dùng cho Phase F AI-Face.

### 3.2. Cách đổi seed

Không đổi seed bằng một setting ngẫu nhiên của Kaggle. Phải đổi config truyền vào
runner:

```yaml
experiment:
  seed: 43
  output_dir: outputs/aiface_phase_e_seed43_baseline
```

hoặc seed `44` với output folder tương ứng. Notebook 11 phải ghi seed vào config
tạm `spectral_run.yaml` trước khi chạy full. Notebook 12 không tự đổi seed; nó
nhận artifact baseline và forensic cùng seed, rồi kiểm tra metadata alignment.

### 3.3. Kiểm tra bắt buộc sau mỗi run

`run_metadata.json` phải xác nhận:

```text
seed: 43 hoặc 44 (training run seed)
split_mode: generator_disjoint_fixed
smoke: false
random fallback: disabled trong log
```

Và phải kiểm tra:

- checksum ba manifest giống seed 42;
- generator train/val/test không thay đổi;
- số lượng và `image_id` của test giống seed 42;
- checkpoint và threshold được chọn từ validation;
- không có test prediction nào được dùng trong lúc train/chọn checkpoint.

## 4. Robustness

### 4.1. Đối tượng đánh giá

Robustness là inference-only. Không train lại trên ảnh bị corruption, không chọn
checkpoint mới và không đổi threshold theo từng corruption.

Mỗi seed dùng checkpoint và threshold đã khóa trên ảnh clean. Đánh giá trên đúng
`test.csv` của seed 42, giữ nguyên `image_id`, `label`, `generator_id` và `split`.

Các branch phải áp dụng corruption trước preprocessing riêng của branch:

```text
RGB corruption -> DINO preprocessing -> Global/Local
RGB corruption -> spectral preprocessing -> Forensic FFT/MLP
```

### 4.2. Conditions cố định

Mỗi condition bắt đầu từ ảnh RGB clean; không nối nhiều corruption trong cùng một
condition.

| Tên | Định nghĩa |
|---|---|
| `clean` | Decode RGB, không biến đổi thêm |
| `jpeg_q70` | Encode/decode JPEG quality 70, subsampling 2 |
| `resize_112_224` | Bilinear resize xuống `112x112`, rồi resize lên `224x224` |
| `blur_r1` | Gaussian blur radius `1.0` trên ảnh trước resize |
| `center_crop_80` | Crop vuông ở tâm, cạnh bằng `floor(0.8 * min(width, height))`, rồi bilinear resize về `224x224` |

`center_crop_80` là crop deterministic, không random crop. Crop được áp dụng trên
ảnh RGB đã decode, trước resize chuẩn của DINOv3 hoặc spectral branch. Không dùng
crop condition cho việc chọn checkpoint.

### 4.3. Metrics và output

Với mỗi `seed x model x condition`, lưu:

- AUROC và Balanced Accuracy;
- Accuracy và F1 bổ sung;
- metric riêng cho `STARGAN` và `StableDiffusion_Inpainting`;
- macro-average trên hai unseen generators;
- `delta = metric_corrupted - metric_clean`;
- số dòng input và checksum test IDs.

Artifact đề xuất:

```text
robustness/
├── predictions_<seed>.csv
├── metrics_by_generator.csv
├── metrics_macro.csv
├── deltas_vs_clean.csv
├── corruption_config.yaml
└── run_metadata.json
```

Không kết luận robustness từ một condition tốt nhất. Báo cáo cả clean score và
delta của từng condition.

## 5. Confidence interval và bootstrap

### 5.1. Đơn vị bootstrap

Bootstrap trên test predictions đã khóa, không chạy lại model. Dùng:

```text
bootstrap_replicates: 2000
bootstrap_seed: 2026
confidence_level: 95%
```

Resample theo các cell `generator_id x label` để giữ cân bằng class và bảo đảm mỗi
unseen generator vẫn có real/fake. Dùng cùng bootstrap indices cho mọi model trong
cùng comparison để tính paired difference.

Vì real pool được dùng chung khi tính metric theo từng unseen generator, phải dùng
cùng chỉ số bootstrap của real pool cho các generator comparisons; không coi các
hàng real lặp lại trong hai bảng generator là ảnh độc lập mới.

### 5.2. Đại lượng cần tính

Cho mỗi model và condition:

- AUROC 95% percentile CI;
- Balanced Accuracy 95% percentile CI;
- Accuracy/F1 CI nếu cần trong phụ lục.

Cho các comparison chính:

```text
Local + Forensic - Local-only
Global + Local + Forensic - Global + Local
Forensic-only - Local-only
```

Tính CI của paired difference bằng chính bootstrap samples chung. Lưu cả phân phối
bootstrap hoặc summary quantiles để tái lập.

Bootstrap CI là uncertainty có điều kiện trên fixed split và checkpoint đã chọn. Nó
không bao gồm uncertainty do đổi dataset, đổi generator split hoặc chọn checkpoint.
Training variation phải báo riêng bằng mean ± sample SD qua seed 42/43/44.

### 5.3. Artifact bootstrap

```text
uncertainty/
├── bootstrap_config.yaml
├── ci_by_model_condition.csv
├── paired_delta_ci.csv
├── seed_summary_mean_sd.csv
└── bootstrap_metadata.json
```

## 6. Chọn model cuối

### 6.1. Tập dùng để chọn

Model/configuration chỉ được chọn bằng validation generators (`AttGAN`, `Palette`)
qua ba run seeds. Test generators chỉ được mở sau khi quyết định đã khóa.

Không chọn model theo test AUROC, test BAcc, rescue/harm hoặc một unseen generator
riêng lẻ.

### 6.2. Quy tắc xếp hạng đã định trước

1. Primary: mean validation macro AUROC qua seed 42/43/44 cao hơn.
2. Tie-break 1: mean validation macro Balanced Accuracy cao hơn.
3. Tie-break 2: sample SD của validation macro AUROC thấp hơn.
4. Tie-break 3: worst-generator validation AUROC cao hơn.
5. Nếu vẫn hòa, chọn cấu hình đơn giản hơn và giữ late fusion equal-probability.

Không lấy run seed tốt nhất làm headline. Bảng phải có từng seed và mean ± SD.

### 6.3. Khóa và đánh giá test

Sau khi chọn configuration, ghi một `model_selection.json` gồm:

- danh sách candidates;
- validation metrics từng seed;
- mean, SD và tie-break result;
- configuration được chọn;
- checkpoint/seed policy;
- thời điểm khóa quyết định.

Sau đó mới báo cáo test unseen cho configuration đã khóa. Test report phải giữ đủ
metric theo `STARGAN`, `StableDiffusion_Inpainting`, macro metric và CI. Không quay
lại sửa model selection dựa trên test score.

## 7. Code/notebook cần có trước khi chạy

Đã có:

- `notebooks/10_aiface_phase_e_baseline.ipynb`;
- `notebooks/11_aiface_phase_e_spectral.ipynb`;
- `notebooks/12_aiface_phase_e_fusion.ipynb`;
- runner baseline, spectral và fusion.

Đã triển khai các thành phần Phase F thuần AI-Face:

- `scripts/run_aiface_phase_f.py`: inference-only robustness trên fixed test manifest;
- `scripts/run_aiface_phase_f_kaggle.py`: wrapper chạy runner trên Kaggle;
- `src/evaluation/aiface_phase_f.py`: corruption, alignment, six late-fusion views và paired bootstrap;
- `scripts/bootstrap_aiface_phase_f.py`: CI 95% theo generator/class cell và seed summary mean ± SD;
- `scripts/select_aiface_model.py`: model selection validation-only, không đọc test predictions;
- `configs/aiface_phase_f.yaml`: khóa conditions, seed bootstrap và selection rule;
- `tests/test_aiface_phase_f.py`: kiểm tra deterministic crop/corruption, alignment, bootstrap và selection;
- Notebook 10/11/12 có `RUN_SEED`; chỉ đổi training seed, không tạo lại fixed manifest.

Không dùng lại runner Residual của Week 3 để chạy các bước này.

## 8. Thứ tự thực hiện

1. Tạo config/output riêng cho seed 43 và chạy Notebook 10 full.
2. Tạo config/output riêng cho seed 43 và chạy Notebook 11 full.
3. Attach hai artifact cùng seed và chạy Notebook 12.
4. Lặp lại bước 1–3 cho seed 44.
5. Kiểm tra alignment và hoàn tất clean validation/test predictions.
6. Chạy robustness trên test IDs cố định cho seed 42/43/44.
7. Chạy bootstrap CI và paired deltas.
8. Chọn model bằng validation mean/SD và ghi `model_selection.json`.
9. Khóa configuration, cập nhật báo cáo và chỉ sau đó chốt test headline.

### 8.1. Lệnh phân tích sau khi Kaggle đã lưu artifact

Robustness dùng đúng ba artifact cùng seed: baseline Global/Local, forensic và
fusion. Ví dụ:

```bash
python scripts/run_aiface_phase_f.py \
  --data-root /path/to/aiface-pilot-32k \
  --baseline-root /path/to/seed43_baseline \
  --forensic-root /path/to/seed43_spectral \
  --fusion-root /path/to/seed43_fusion \
  --output-dir outputs/aiface_phase_f_seed43
```

Sau khi có Phase F cho các seed, bootstrap dùng predictions đã khóa:

```bash
python scripts/bootstrap_aiface_phase_f.py \
  --phase-f-dir outputs/aiface_phase_f_seed42 \
  --phase-f-dir outputs/aiface_phase_f_seed43 \
  --phase-f-dir outputs/aiface_phase_f_seed44 \
  --output-dir outputs/aiface_phase_f_uncertainty
```

Model selection chỉ đọc validation artifacts:

```bash
python scripts/select_aiface_model.py \
  --baseline-root <seed42-baseline> --forensic-root <seed42-forensic> --fusion-root <seed42-fusion> \
  --baseline-root <seed43-baseline> --forensic-root <seed43-forensic> --fusion-root <seed43-fusion> \
  --baseline-root <seed44-baseline> --forensic-root <seed44-forensic> --fusion-root <seed44-fusion> \
  --output-dir outputs/aiface_phase_f_model_selection
```

## 9. Definition of Done

- [ ] Seed 43 và 44 có đủ ba branch checkpoint và sáu cấu hình output.
- [ ] Mọi run dùng cùng `generator_disjoint_fixed_v1` và cùng test IDs.
- [ ] Checkpoint/threshold được chọn từ validation.
- [ ] Có đủ năm conditions clean/JPEG/resize/blur/center-crop.
- [ ] Robustness có metric macro, per-generator và delta so với clean.
- [ ] Bootstrap 2.000 lần, seed 2026, CI 95% và paired differences đã lưu.
- [ ] Có bảng mean ± SD qua seed 42/43/44.
- [ ] Model cuối được chọn từ validation trước khi đọc test để ra quyết định.
- [ ] Có artifact/config/checksum/provenance và cập nhật `aiface-phase-e-report.md`.
