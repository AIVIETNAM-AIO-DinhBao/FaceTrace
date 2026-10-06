# Báo cáo Audit Dataset AI-Face v2 — Phase A, B và C

**Ngày thực hiện:** 2026-10-07  
**Dataset:** AI-Face v2  
**Subset pilot:** `aiface_pilot_32k`  
**Seed sampling:** `42`  
**Trạng thái:** Phase A và B hoàn tất; Phase C đạt quality gate dữ liệu.

## 1. Mục tiêu và phạm vi

Báo cáo này ghi lại toàn bộ quy trình chuẩn bị dataset cho pilot unseen-generator của FaceTrace. Mục tiêu là xác nhận rằng dataset có thể dùng để so sánh các nhánh Global, Local và Forensic của pipeline trên các generator không xuất hiện trong train.

Phạm vi main experiment chỉ gồm:

- ảnh real từ `imdb_wiki`;
- ảnh fully AI-generated thuộc hai family `GANs` và `DMs`;
- không đưa các subset `deepfakes/` vào pilot vì chúng là face swap, reenactment hoặc video-derived manipulation.

Báo cáo này chỉ đánh giá tính toàn vẹn, provenance và cách chia dữ liệu. Chưa có kết luận về AUROC, balanced accuracy hoặc sức mạnh của mô hình.

## 2. Nguồn dữ liệu và artifact evidence

Các nguồn local đã được kiểm tra:

```text
data/AI_Face/AI_Face_annotationsV2/train_data.csv
data/AI_Face/AI_Face_annotationsV2/test_data.csv
data/AI_Face/OneDrive_1_10-7-2026.zip
data/AI_Face/OneDrive_2_10-7-2026.zip
data/AI_Face/imdb_wiki.zip
```

Artifact dùng để tái lập:

| Artifact | Vai trò |
| --- | --- |
| `notebooks/09_aiface_dataset_audit.ipynb` | Audit annotation, archive và protocol |
| `scripts/create_aiface_subset.py` | Tạo subset generator-disjoint |
| `scripts/repair_aiface_subset.py` | Thay các ảnh real bị duplicate |
| `scripts/audit_aiface_subset.py` | Chạy quality gate trên toàn bộ subset |
| `docs/aiface-pilot-plan.md` | Protocol và trạng thái phase |
| `docs/aiface-audit.md` | Báo cáo audit chi tiết |

Artifact kết quả:

```text
data/AI_Face/aiface_pilot_32k/
├── images/
├── manifests/{train,val,test,all}.csv
├── audit/counts.csv
├── audit/duplicate_report.csv
├── audit/quality_gate.json
├── audit/repair_report.json
└── subset_metadata.json
```

### Evidence record

Các số liệu trong báo cáo được bind vào các artifact local sau; không dùng kết quả suy đoán từ model:

| Evidence ID | Nguồn | Nội dung xác nhận |
| --- | --- | --- |
| `E-A1` | `data/AI_Face/audit_outputs/audit_summary.json` | Quy mô annotation, generator presence và kiểm tra split gốc |
| `E-A2` | `data/AI_Face/aiface_pilot_32k/subset_metadata.json` | Seed, generator mapping, real source và limitation |
| `E-A3` | `data/AI_Face/aiface_pilot_32k/audit/quality_gate.json` | Counts, decode, mode, hash uniqueness và generator-disjointness |
| `E-A4` | `data/AI_Face/aiface_pilot_32k/audit/repair_report.json` | Ba thay thế duplicate và provenance candidate |
| `E-A5` | `data/AI_Face/aiface_pilot_32k/manifests/*.csv` | Image-level manifest và split assignment |

## 3. Phase A — Audit annotation và archive

### 3.1. Quy mô annotation

| Annotation split gốc | Tổng ảnh | Fake | Real |
| --- | ---: | ---: | ---: |
| `train_data.csv` | 1.317.236 | 996.761 | 320.475 |
| `test_data.csv` | 329.309 | 248.899 | 80.410 |
| Tổng | 1.646.545 | 1.245.660 | 400.885 |

Annotation có 18 generator groups được derive từ `Image Path`: 10 GAN groups và 8 diffusion groups, trong đó `CommercialTools` được tách thành các nhóm con theo folder.

### 3.2. Vì sao không dùng split gốc

Split gốc là image-level split, không phải generator-held-out split:

- không có exact path overlap giữa hai CSV gốc;
- các generator pilot đều xuất hiện trong cả train và test;
- vì vậy `test_data.csv` không thể đại diện cho unseen-generator test.

Pilot tạo lại split theo generator ở Phase B.

### 3.3. Kiểm tra archive

Các outer archive cần thiết đã pass CRC. Các inner archive của 8 generator pilot đều tồn tại. Đã stream-extract ảnh mẫu từ từng generator và từ `imdb_wiki`; các ảnh mẫu đều được PIL đọc thành công.

Resolution mẫu quan sát được gồm `200x200`, `244x244` và `256x256`. Do đó preprocessing resize/normalization phải được cố định trong pipeline train và ghi cùng artifact run.

### 3.4. Kết luận Phase A

Phase A **đạt** cho mục tiêu tạo pilot:

- nguồn dữ liệu hợp lệ và đủ lớn cho các generator đã chọn;
- archive cần thiết có thể đọc;
- split gốc được xác định là không phù hợp cho unseen-generator evaluation;
- đã xác định rõ các rủi ro real-source và identity leakage.

## 4. Phase B — Tạo subset pilot 32K

### 4.1. Quy mô và cân bằng lớp

```text
Fake:  16.000 ảnh
Real:  16.000 ảnh
Tổng:  32.000 ảnh
```

Mỗi generator fake đóng góp 2.000 ảnh. Real chỉ lấy từ `imdb_wiki` để giữ dung lượng pilot khoảng 1,5 GB và không phải giải nén toàn bộ `FFHQ`.

### 4.2. Generator-disjoint mapping

| Family | Generator | Vai trò | Fake images |
| --- | --- | --- | ---: |
| GAN | `taming_transformer_VQGAN` | Train | 2.000 |
| GAN | `stylegan3` | Train | 2.000 |
| GAN | `AttGAN` | Validation | 2.000 |
| GAN | `STARGAN` | Unseen test | 2.000 |
| Diffusion | `StableDiffusion1.5` | Train | 2.000 |
| Diffusion | `latent_diffusion` | Train | 2.000 |
| Diffusion | `Palette` | Validation | 2.000 |
| Diffusion | `StableDiffusion_Inpainting` | Unseen test | 2.000 |

Split tổng hợp:

| Split | Fake | Real | Tổng |
| --- | ---: | ---: | ---: |
| Train | 8.000 | 8.000 | 16.000 |
| Validation | 4.000 | 4.000 | 8.000 |
| Unseen test | 4.000 | 4.000 | 8.000 |

Không có fake generator nào xuất hiện ở nhiều split. `STARGAN` và `StableDiffusion_Inpainting` chỉ xuất hiện ở test, nên test fake được giữ cho đánh giá cuối cùng sau khi chốt checkpoint và threshold.

### 4.3. Sampling và manifest

Subset được tạo deterministic với seed `42`. Manifest giữ các trường:

```text
image_id
image_path
label
class_name
split
family
generator_id
real_source
original_path
annotation_split
identity_hint
```

Path trong manifest là path local tương đối dưới `aiface_pilot_32k/images/`; không giữ path ảo của archive làm đường dẫn train.

### 4.4. Kết luận Phase B

Phase B **hoàn tất**. Derived subset đã được tạo độc lập với archive gốc, có manifest train/val/test/all và có mapping generator-disjoint rõ ràng.

## 5. Phase C — Quality gate và sửa dữ liệu

### 5.1. Duplicate repair

Exact SHA-256 audit ban đầu phát hiện 3 nhóm duplicate trong ảnh real. Các nhóm này đến từ nhiều file `imdb_wiki` khác nhau nhưng có cùng nội dung byte.

Đã thay 3 dòng manifest bằng candidate real chưa được chọn và ghi lại provenance trong:

```text
data/AI_Face/aiface_pilot_32k/audit/repair_report.json
```

Archive và annotation gốc không bị chỉnh sửa.

### 5.2. Kết quả quality gate cuối

| Kiểm tra | Kết quả |
| --- | ---: |
| Manifest rows | 32.000 |
| Train / validation / test | 16.000 / 8.000 / 8.000 |
| Fake/real trong từng split | 1:1 |
| Unique SHA-256 | 32.000 |
| Exact duplicate groups | 0 |
| File decode được bằng PIL | 32.000/32.000 |
| Image mode | RGB |
| Generator overlap giữa split | 0 |
| Missing local files | 0 |

Kết quả máy được lưu tại `audit/quality_gate.json` với `passed: true`.

### 5.3. Kiểm tra trực quan

Đã xem thử ảnh đại diện từ nhánh fake và real sau repair. Ảnh hiển thị đúng, không có lỗi decode hoặc file rỗng. Kiểm tra trực quan này chỉ là sanity check, không thay thế quality gate trên toàn bộ 32.000 ảnh.

### 5.4. Identity và real-source limitation

`identity_hint` được suy ra từ filename để audit, nhưng annotation không có identity ID chính thức. Audit heuristic hiện ghi nhận:

| Đại lượng | Giá trị |
| --- | ---: |
| Identity hint ở train | 4.218 |
| Identity hint ở validation | 2.669 |
| Identity hint ở test | 2.599 |
| Overlap train/validation | 1.383 |
| Overlap train/test | 1.384 |
| Overlap validation/test | 1.027 |

Vì vậy subset này **generator-disjoint nhưng chưa identity-disjoint**. Ngoài ra, toàn bộ real pilot đến từ `imdb_wiki`. Hai điểm này có thể tạo shortcut theo source/identity và phải được ghi rõ khi diễn giải kết quả.

### 5.5. Kết luận Phase C

Phase C **đạt quality gate dữ liệu**, nhưng chưa phải chứng nhận identity-disjoint. Dataset đủ sạch để chạy pilot engineering/smoke test; kết luận scientific về generalization cần giữ đúng phạm vi `unseen GAN/diffusion generators trong AI-Face v2`.

## 6. Quyết định hiện tại

AI-Face v2 và subset `aiface_pilot_32k` được chấp nhận làm dataset chính cho pilot vì:

1. có nhiều generator thuộc cả GAN và diffusion;
2. có thể dựng generator-held-out split;
3. subset cân bằng, đọc được và không còn exact duplicate;
4. dung lượng phù hợp để đóng gói Kaggle.

Chưa được claim rằng pilot đã chứng minh generalization trên mọi AI generator hoặc đã kiểm soát identity leakage hoàn toàn.

## 7. Smoke test DINOv3 trên local

Đã triển khai `scripts/smoke_test_aiface_dinov3.py`. Script kiểm tra manifest train/validation, official image processor, device MPS/CPU, trạng thái frozen của backbone, shape global/patch feature và NaN/Inf; script không train classifier.

Sau khi đăng nhập Hugging Face và chạy trong conda environment `video-highlight`, smoke test đã **PASS trên MPS**:

| Kiểm tra | Kết quả |
| --- | --- |
| Train/validation batch | `2 x 3 x 224 x 224` |
| Global feature | `2 x 768` |
| Patch feature | `2 x 196 x 768` |
| Backbone | Frozen |
| NaN/Inf | Không có |
| Device | Apple MPS |

Lần chạy đầu từng bị `401 gated repo`; sau khi cấp quyền, checkpoint tải và forward thành công. Artifact kết quả được lưu tại:

```text
outputs/aiface_smoke_test/dinov3_smoke.json
```

```bash
conda activate video-highlight
python scripts/smoke_test_aiface_dinov3.py --device auto --batch-size 2
```

## 8. Việc còn lại sau Phase A–C

1. Chốt version preprocessing, checkpoint policy và threshold policy.
2. Publish package private `nguyentrann0703/aiface-pilot-32k` cùng code pipeline lên Kaggle; không upload outer archive.
3. Chạy các baseline `Global-only`, `Local-only`, `Forensic-only (Radial FFT + MLP)` và các fusion ablation.
4. Báo cáo metric theo từng unseen generator và macro-average, kèm limitation identity/real-source.

## 9. Lệnh tái lập quality gate và smoke test

```bash
python3 scripts/repair_aiface_subset.py
python3 scripts/audit_aiface_subset.py
python3 scripts/smoke_test_aiface_dinov3.py
```

Lệnh audit không sửa raw archive; nó chỉ kiểm tra derived subset và cập nhật các report trong thư mục `audit/`.
