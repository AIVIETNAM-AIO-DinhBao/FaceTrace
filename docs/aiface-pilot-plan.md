# Kế hoạch Pilot AI-Face

## 1. Mục tiêu

Xây dựng một pilot benchmark nhỏ từ AI-Face v2 để kiểm tra pipeline Global/Local/Forensic trên các generator chưa xuất hiện trong tập train.

Pilot này không thay thế benchmark AI-Face đầy đủ. Mục tiêu là:

- kiểm tra data pipeline và generator-disjoint split;
- ước lượng chi phí preprocessing trên Kaggle;
- so sánh Frozen DINOv3 Global, Local và forensic branch;
- phát hiện shortcut/leakage trước khi mở rộng dataset.

## 2. Dữ liệu hiện có

Root local:

```text
data/AI_Face/
├── AI_Face_annotationsV2/
│   ├── train_data.csv
│   └── test_data.csv
├── OneDrive_1_10-7-2026.zip
├── OneDrive_2_10-7-2026.zip
└── imdb_wiki.zip
```

Hai file `OneDrive_*.zip` là archive lồng nhau:

- `OneDrive_1_10-7-2026.zip` chứa các archive GAN đã chọn;
- `OneDrive_2_10-7-2026.zip` chứa các archive diffusion đã chọn.

Không giải nén toàn bộ AI-Face. Chỉ lấy các inner archive nằm trong danh sách bên dưới.

## 3. Phạm vi pilot

Chỉ sử dụng ảnh real và ảnh fully AI-generated từ GAN/diffusion. Tạm thời loại bỏ toàn bộ `deepfakes/` vì các subset này chủ yếu là face-swap, face-reenactment hoặc video-derived manipulation.

### 3.1. Generator được chọn

| Family | Generator | Inner archive | Vai trò |
| --- | --- | --- | --- |
| GAN | `taming_transformer_VQGAN` | `taming_transformer_VQGAN.zip` | Train |
| GAN | `stylegan3` | `stylegan3.zip` | Train |
| GAN | `AttGAN` | `AttGAN.zip` | Validation |
| GAN | `STARGAN` | `STARGAN.zip` | Unseen test |
| Diffusion | `StableDiffusion1.5` | `StableDiffusion1.5.zip` | Train |
| Diffusion | `latent_diffusion` | `latent_diffusion.zip` | Train |
| Diffusion | `Palette` | `Palette.zip` | Validation |
| Diffusion | `StableDiffusion_Inpainting` | `StableDiffusion_Inpainting.zip` | Unseen test |

Annotation đã xác nhận số lượng tổng của các generator này đủ lớn để lấy 2.000 ảnh mỗi generator. Không chọn `CommercialTools/DALL-E2`, `CommercialTools/IF` hoặc `CommercialTools/Midjourney` cho pilot test vì số lượng annotation hiện tại quá nhỏ.

### 3.2. Real data

Dùng `imdb_wiki.zip` cho pilot để tránh phải tải/giải nén `FFHQ.zip` 89 GB.

Lấy tổng cộng 16.000 ảnh real, chia thành:

```text
train: 8.000 real
val:   4.000 real
test:  4.000 real
```

Real được lấy từ `imdb_wiki` và stratified theo các trường demographic trong annotation. `identity_hint` được parse từ filename để audit, nhưng subset pilot hiện **chưa enforced identity-disjointness** giữa train/val/test; đây là limitation phải ghi rõ trong mọi kết quả.

Chỉ dùng `imdb_wiki` là lựa chọn tiết kiệm cho pilot, không phải cấu hình cuối cùng. Real-source control với FFHQ sẽ là follow-up.

## 4. Kích thước pilot

Lấy tối đa 2.000 fake images cho mỗi generator:

```text
8 generators × 2.000 = 16.000 fake images
16.000 real images
Tổng = 32.000 images
```

### 4.1. Split generator-disjoint

```text
Train:
  VQGAN, stylegan3, StableDiffusion1.5, latent_diffusion
  4 generators × 2.000 = 8.000 fake
  8.000 real

Validation:
  AttGAN, Palette
  2 generators × 2.000 = 4.000 fake
  4.000 real

Test unseen:
  STARGAN, StableDiffusion_Inpainting
  2 generators × 2.000 = 4.000 fake
  4.000 real
```

Generator trong validation/test không được dùng để chọn checkpoint, threshold hoặc hyperparameter của train.

## 5. Xử lý annotation

Không dùng trực tiếp train/test split mặc định của AI-Face cho unseen evaluation. Hai CSV hiện tại là image-level split; tất cả generator đều xuất hiện ở cả train và test với tỷ lệ xấp xỉ 80/20.

Quy trình cần làm:

1. Đọc và gộp `train_data.csv` và `test_data.csv` thành một bảng annotation duy nhất.
2. Chuẩn hóa `Image Path` và derive các cột:
   - `family`: `GANs`, `DMs`, `Real`, `deepfakes`;
   - `generator_id`: tên folder generator;
   - `real_source`: `imdb_wiki` trong pilot;
   - `identity_id`: prefix `nm...` nếu parse được.
3. Lọc 8 generator trong pilot và loại bỏ `deepfakes`.
4. Lấy mẫu deterministic với seed 42, tối đa 2.000 fake mỗi generator.
5. Chia real theo identity trước khi gán vào train/val/test.
6. Cân bằng fake/real trong từng split.
7. Tạo manifest với local path thực tế, không giữ path ảo `/AI_Face_imagesV2/...`.

## 6. Xử lý archive và dung lượng

Không extract toàn bộ inner archive. Chỉ extract các generator đã chọn, sau đó chỉ giữ các image có trong subset manifest.

Ước lượng:

- selected ZIP archives: khoảng 14,6 GB nếu tải nguyên các archive đã chọn;
- sample sau khi extract: khoảng 1,5–3 GB, tùy kích thước ảnh thực tế;
- annotation, cache và temporary files: cần thêm khoảng 10–20 GB;
- nên có ít nhất 25–35 GB dung lượng trống khi tạo subset.

Nếu Kaggle package chỉ chứa subset 32.000 ảnh, không upload các outer archive và không upload ảnh ngoài subset.

## 7. Artifact cần tạo

Dự kiến output sau bước tạo subset:

```text
data/AI_Face/aiface_pilot_32k/
├── images/
├── manifests/
│   ├── all.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── audit/
    ├── generator_counts.csv
    ├── family_counts.csv
    ├── class_counts.csv
    ├── source_identity_overlap.csv
    ├── duplicate_report.csv
    ├── missing_paths.csv
    └── subset_metadata.json
```

`subset_metadata.json` phải ghi:

- source annotation filenames;
- selected generator IDs;
- sample cap mỗi generator;
- seed;
- split mapping;
- class counts;
- real-source limitation;
- archive names và checksum nếu có thể.

## 8. Quality gates trước khi train

Chỉ chạy DINOv3/forensic pipeline sau khi tất cả điều kiện sau đạt:

- tổng fake = 16.000 và real = 16.000;
- mỗi generator có tối đa 2.000 ảnh;
- không có generator overlap giữa train, val và test;
- exact image path overlap bằng 0;
- identity overlap bằng 0 nếu identity parse được;
- không có path đến `deepfakes/` trong manifest;
- tất cả local path trong manifest tồn tại và đọc được;
- train/val/test có cùng preprocessing;
- test chỉ được mở để tính metric sau khi checkpoint và threshold đã được chốt.

## 9. Thứ tự thực hiện

### Phase A — Audit annotation

- Kiểm tra lại path và count theo generator.
- Tạo bảng mapping generator/family/split.
- Xác định khả năng parse identity từ `imdb_wiki`.

### Phase B — Tạo subset

- Extract selected inner archives.
- Sample 2.000 fake mỗi generator.
- Sample 16.000 real theo demographic strata; parse `identity_hint` chỉ để audit overlap.
- Tạo manifest và audit reports.

### Phase C — Smoke test

- Mở ngẫu nhiên một số ảnh ở mỗi generator.
- Kiểm tra kích thước, mode màu, file hỏng và label.
- Chạy một batch qua DINOv3 processor.

### Phase D — Kaggle package

- Đóng gói chỉ `aiface_pilot_32k` và code pipeline.
- Không upload outer archives.
- Ghi rõ dataset version, seed và manifest checksum trong artifact.

### Phase E — Main pilot experiment

- Global-only.
- Local-only.
- Forensic-only, ưu tiên Radial FFT + MLP.
- Global + Local.
- Local + Forensic.
- Global + Local + Forensic.
- Báo cáo metric theo từng unseen generator và macro-average.

## 10. Giới hạn cần ghi trong báo cáo

- 2.000 ảnh mỗi generator là pilot, chưa phải benchmark scale lớn.
- Real pilot chỉ dùng `imdb_wiki`, có nguy cơ real-source shortcut.
- Generator-disjointness được thiết kế lại; split mặc định của AI-Face không phải unseen-generator split.
- Chưa claim identity-disjoint nếu metadata không đủ để xác minh.
- Kết quả pilot không được diễn giải thành generalization trên tất cả AI generators.

## 11. Trạng thái thực hiện (2026-10-07)

- Phase A — Audit annotation: **hoàn tất** trong `notebooks/09_aiface_dataset_audit.ipynb` và `docs/aiface-audit.md`.
- Phase B — Tạo subset: **hoàn tất**; subset hiện tại có 32.000 ảnh tại `data/AI_Face/aiface_pilot_32k/`.
- Exact duplicate repair: **hoàn tất**; đã thay 3 ảnh real bị trùng byte và lưu `audit/repair_report.json`.
- Phase C — Quality gate: **đạt**; `audit/quality_gate.json` ghi 32.000 file RGB đọc được, cân bằng lớp, không duplicate SHA-256 và generator-disjoint. DINOv3 smoke test **đã pass trên MPS** trong env `video-highlight`.
- Identity audit: **chưa đạt identity-disjointness**; số overlap identity heuristic được lưu trong `audit/quality_gate.json`. Không được gọi split này là identity-disjoint.
- Phase D — Kaggle package: **chưa chạy**. Chỉ upload subset sau khi nhóm chấp nhận limitation real-source/identity này.
- Phase E — Main pilot: **chưa chạy**.
