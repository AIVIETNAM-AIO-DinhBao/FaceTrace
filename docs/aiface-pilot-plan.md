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
- identity overlap phải được audit và ghi vào artifact; nếu khác 0 thì phải đánh dấu
  rõ là chưa identity-disjoint và không được dùng làm claim đã loại bỏ identity leakage;
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

#### 9.1. Hai hình thức package

Pilot có hai hình thức package để giảm rủi ro Kaggle không mount được 32.000 file
nhỏ:

```text
kaggle_upload/aiface_pilot_32k/
kaggle_upload/aiface_pilot_32k_archive/aiface_pilot_32k.tar
```

Package file rời dễ kiểm tra thủ công nhưng có thể gây lỗi mount hoặc timeout khi
Kaggle dựng filesystem. Package archive là phương án ưu tiên nếu Kaggle báo lỗi
`ERRORED_MOUNTING_DATASET` hoặc retry mount nhiều lần.

Không upload các outer archive AI-Face hoặc ảnh ngoài subset. Archive chỉ được chứa
ảnh đã chọn, manifest, `subset_metadata.json` và audit cần thiết.

#### 9.2. Kiểm tra bắt buộc sau khi mount

Một run Kaggle chỉ được xem là hợp lệ khi log có đủ các dấu hiệu sau:

```text
Extracted dataset archive
Split mode: generator_disjoint_fixed
train generators: ...
val generators: ...
test generators: ...
test manifest: ...
```

Runner phải xác nhận:

- `train/manifest.csv`, `val/manifest.csv` và `test/manifest.csv` tồn tại;
- tất cả path trong manifest tồn tại sau khi giải nén;
- `subset_metadata.json` được đọc thành công;
- generator test không giao với generator train/validation;
- test manifest được giữ riêng và không được dùng để chọn checkpoint hoặc threshold;
- runner không gọi random split fallback.

Nếu log hiển thị `in_domain_random_fallback`, run đó chỉ là smoke test và không được
dùng trong báo cáo unseen-generator.

### Phase E — Main pilot experiment

#### 9.3. Mục tiêu và câu hỏi kiểm định

Phase E là thí nghiệm chính để kiểm tra câu hỏi:

> Local và forensic representation có cải thiện khả năng generalization của DINOv3
> trên generator chưa xuất hiện trong tập train hay không?

Đây không phải là một cuộc thi tối ưu accuracy trên random split. Mỗi cấu hình phải
được huấn luyện trên bốn generator train, chọn checkpoint bằng validation generators,
và chỉ mở test generators ở bước đánh giá cuối.

#### 9.4. Các cấu hình bắt buộc

Tất cả cấu hình phải dùng cùng fixed split, cùng preprocessing, cùng label mapping và
cùng seed policy.

1. **Global-only**

   Frozen DINOv3 tạo global feature, sau đó đưa qua classifier nhỏ. Đây là baseline
   chính để đo giá trị gia tăng của Local và Forensic.

2. **Local-only**

   Frozen DINOv3 tạo patch features. Patch được tổng hợp bằng quy tắc cố định trước
   classifier; không thay đổi pooling theo từng generator.

3. **Forensic-only**

   Ưu tiên Radial FFT + MLP:

   ```text
   RGB
   → fixed resize/crop
   → Hann window
   → 2D FFT
   → log magnitude
   → radial averaging
   → MLP nhỏ
   ```

   Nhánh này được dùng để kiểm tra riêng thông tin phổ tần số. Không gọi kết quả của
   TinyCNN residual ở tập Who Is AI là kết quả forensic của AI-Face pilot.

4. **Global + Local**

   Kết hợp hai xác suất độc lập bằng late probability fusion. Đây là mốc so sánh để
   biết forensic có bổ sung ngoài hai representation DINOv3 hay không.

5. **Local + Forensic**

   Dùng để kiểm tra forensic có bổ trợ cho local representation khi không có global
   feature hay không.

6. **Global + Local + Forensic**

   Là cấu hình đầy đủ, nhưng không mặc định là model cuối. Chỉ giữ cấu hình này nếu
   cải thiện được macro metric trên unseen generators và không phụ thuộc vào một seed
   hoặc một generator đơn lẻ.

#### 9.5. Fusion policy

Cấu hình fusion chính là late probability fusion:

```text
p_fusion = mean(p_branch_1, p_branch_2, ...)
```

Không chọn trọng số hoặc threshold bằng test set. Nếu thử weighted fusion, trọng số
chỉ được chọn trên validation generators và phải được ghi trong resolved config trước
khi chạy test. Feature/latent fusion chỉ là ablation bổ sung, không thay thế protocol
late fusion chính.

#### 9.6. Đánh giá theo generator

Không chỉ báo cáo một metric gộp cho toàn bộ test. Mỗi run phải lưu:

- AUROC của từng unseen generator;
- Balanced Accuracy của từng unseen generator;
- F1 và Accuracy bổ sung;
- macro-average trên hai unseen generators;
- confusion matrix;
- threshold được chọn trên validation;
- số lượng real/fake của từng generator.

Bảng kết quả chính có dạng:

| Model | STARGAN AUROC | SD-Inpainting AUROC | Macro AUROC | Macro BAcc |
| --- | ---: | ---: | ---: | ---: |
| Global-only | ... | ... | ... | ... |
| Local-only | ... | ... | ... | ... |
| Forensic-only | ... | ... | ... | ... |
| Global + Local | ... | ... | ... | ... |
| Local + Forensic | ... | ... | ... | ... |
| Global + Local + Forensic | ... | ... | ... | ... |

Kết luận chỉ được dựa trên test unseen sau khi checkpoint, preprocessing và threshold
đã khóa.

#### 9.7. Artifact bắt buộc của Phase E

Mỗi cấu hình phải lưu độc lập:

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

`run_metadata.json` phải ghi:

- git commit hoặc source version;
- Kaggle dataset slug/version;
- manifest checksum;
- split mode và generator mapping;
- seed và device;
- preprocessing;
- checkpoint selection rule;
- threshold rule;
- fusion rule;
- thời điểm chạy.

#### 9.8. Shortcut diagnostic trước khi kết luận

Ngoài audit thống kê ở Notebook 00, phải chạy một classifier độc lập bằng các feature
đơn giản:

- file size;
- luminance;
- saturation;
- gray-pixel fraction;
- edge statistics;
- resolution và image format nếu có.

Logistic Regression phải được train trên train split và đánh giá trên validation. Nếu
shortcut đạt AUROC cao bất thường, kết quả model chính phải được gắn cờ và phân tích
riêng; không được diễn giải là bằng chứng forensic khi chưa kiểm soát shortcut.

#### 9.9. Rescue/harm analysis

Với các cặp `Global + Local`, `Local + Forensic` và `Global + Local + Forensic`, phải
lưu prediction từng ảnh để tính:

- baseline sai, candidate đúng: rescue;
- baseline đúng, candidate sai: harm;
- `net_gain = rescue - harm`;
- các nhóm trên toàn bộ test và tách theo từng unseen generator.

Phân tích rescue/harm phải dùng cùng test images và cùng threshold với bảng metric,
không được tính lại trên một subset thuận tiện hơn.

### Phase F — Kiểm chứng và báo cáo

Sau Phase E, nhóm mới chạy các thí nghiệm bổ sung và xác minh đa seed:

- xác minh/replay seed 42 và hoàn tất seed 43/44 cho các cấu hình chính;
- JPEG compression, resize, blur và crop robustness;
- rescue/harm theo từng unseen generator;
- so sánh late fusion với feature fusion nếu còn tài nguyên;
- chọn model cuối dựa trên macro AUROC và độ ổn định qua seed;
- cập nhật `docs/aiface-phase-e-report.md` và báo cáo nghiên cứu.

### Phase G — Tiêu chí hoàn thành và phân công

#### 9.10. Tiêu chí hoàn thành Phase E

Phase E chỉ được đánh dấu **hoàn tất** khi:

- sáu cấu hình bắt buộc đã chạy;
- tất cả run dùng `generator_disjoint_fixed`;
- test generators không xuất hiện trong train;
- có metric riêng cho từng unseen generator và macro-average;
- có predictions, config và checkpoint tương ứng;
- shortcut diagnostic đã chạy;
- rescue/harm đã lưu cho các fusion chính;
- log chứng minh runner không random split lại;
- limitation identity và real-source được giữ nguyên trong báo cáo.

Nếu mới chạy baseline hoặc chỉ có một forensic branch, trạng thái phải ghi là
`Phase E — incomplete` (hoặc mô tả cụ thể các hạng mục còn thiếu), không ghi là đã
hoàn thành.

### 10. Phân công và bàn giao Phase E

#### 10.1. Owner — dataset và Global/Local

##### 10.1.1. Mục tiêu

Phụ trách đưa subset `aiface_pilot_32k` lên Kaggle, xác nhận fixed
generator-disjoint split và chạy các baseline DINOv3 làm mốc so sánh cho forensic
branch.

Phần việc này trả lời ba câu hỏi:

1. DINOv3 Global-only hoạt động thế nào trên hai generator chưa thấy?
2. Local-only có cải thiện so với Global-only hay không?
3. Global + Local có bổ sung thực sự trên unseen generators hay chỉ tốt hơn trong
   in-domain setting?

##### 10.1.2. Dataset và split bắt buộc

Dataset dùng chung:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

Kaggle có thể hiển thị dataset này ở dạng đã giải nén với các thư mục `train/`,
`val/` và `test/`; runner sẽ đọc trực tiếp các manifest trong layout đó.

Generator mapping:

| Split | Fake generators | Real |
| --- | --- | --- |
| Train | `taming_transformer_VQGAN`, `stylegan3`, `StableDiffusion1.5`, `latent_diffusion` | `imdb_wiki` |
| Validation | `AttGAN`, `Palette` | `imdb_wiki` |
| Test unseen | `STARGAN`, `StableDiffusion_Inpainting` | `imdb_wiki` |

Không được dùng split image-level mặc định của AI-Face. Không được để test unseen
tham gia chọn checkpoint, threshold hoặc hyperparameter.

##### 10.1.3. Việc cần thực hiện

###### 10.1.3.1. Dùng dataset shared và xác nhận Kaggle

Dataset shared của nhóm:

```text
nguyentrann0703/aiface-pilot-32k-single-bin
```

Kaggle hiện có thể hiển thị archive này như layout đã giải nén. Không cần upload lại
dataset riêng; runner sẽ đọc trực tiếp các thư mục `train/`, `val/` và `test/` nếu
chúng được mount thành công.

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

###### 10.1.3.2. Chạy baseline

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

##### 10.1.4. Artifact phải lưu

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

##### 10.1.5. Metric và bàn giao

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

##### 10.1.6. Shortcut diagnostic

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

##### 10.1.7. Điều kiện hoàn thành phần việc

Phần việc được xem là hoàn tất khi:

- Kaggle run dùng `generator_disjoint_fixed`;
- ba baseline đã chạy thành công;
- có prediction cho toàn bộ test images;
- có metric riêng cho hai unseen generators và macro-average;
- artifact có config, checkpoint, metadata và Kaggle URL;
- prediction format sẵn sàng để ghép với forensic branch.

Sau khi teammate bàn giao forensic predictions, dùng cùng test IDs để chạy fusion,
rescue/harm và cập nhật bảng Phase E.

##### 10.1.8. Entry points đã triển khai

Phần Owner Phase E dùng runner riêng, không dùng runner Residual của Week 3:

```text
configs/aiface_phase_e.yaml
scripts/run_aiface_phase_e.py
scripts/run_aiface_phase_e_kaggle.py
notebooks/10_aiface_phase_e_baseline.ipynb
```

Runner Kaggle dùng dataset `nguyentrann0703/aiface-pilot-32k-single-bin`, tự giải
nén file `.bin`, bắt buộc fixed `train/val/test` manifests và tắt random fallback.
Notebook chỉ điều phối runner và hiển thị artifact; logic train/evaluation nằm
trong script để local và Kaggle dùng cùng protocol.


#### 10.2. Teammate — spectral forensic

##### 10.2.1. Mục tiêu

Implement và đánh giá một forensic branch độc lập dựa trên phổ tần số. Nhánh này
nhằm kiểm tra liệu thông tin phổ có bổ sung cho representation Global/Local của
DINOv3 trên generator chưa xuất hiện trong train hay không.

Đây là baseline frequency-domain có kiểm soát, không phải reproduction đầy đủ của
SPAI, AIDE hoặc một paper cụ thể.

##### 10.2.2. Pipeline bắt buộc

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

##### 10.2.3. Dataset và protocol dùng chung

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

##### 10.2.4. File cần tạo

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

##### 10.2.5. Test và smoke test

###### 10.2.5.1. Unit test

Kiểm tra:

- output FFT không có NaN/Inf;
- radial profile có kích thước cố định;
- Hann window có đúng shape và range hợp lệ;
- batch size khác nhau cho cùng output dimension;
- model forward trả probability hợp lệ;
- dataset giữ đúng label và `image_id`.

###### 10.2.5.2. Smoke test

Chạy trên subset nhỏ trước khi gửi Kaggle:

- đọc được train/val/test manifest;
- forward một batch thành công;
- loss giảm hoặc training loop hoàn tất vài bước;
- prediction CSV có đủ image IDs;
- không có duplicate hoặc missing IDs.

##### 10.2.6. Các cấu hình cần chạy

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

##### 10.2.7. Artifact phải bàn giao

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

##### 10.2.8. Kiểm tra rescue/harm

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

##### 10.2.9. Điều kiện hoàn thành phần việc

Phần việc được xem là hoàn tất khi:

- Radial FFT + MLP forward/training smoke test pass;
- unit tests pass;
- Forensic-only đã chạy trên fixed split;
- predictions có đủ và đúng thứ tự test IDs;
- metrics có theo từng unseen generator và macro-average;
- checkpoint/config/environment đã lưu;
- fusion predictions có thể ghép trực tiếp với baseline;
- không thay đổi protocol hoặc code baseline Global/Local.


## 11. Giới hạn cần ghi trong báo cáo

- 2.000 ảnh mỗi generator là pilot, chưa phải benchmark scale lớn.
- Real pilot chỉ dùng `imdb_wiki`, có nguy cơ real-source shortcut.
- Generator-disjointness được thiết kế lại; split mặc định của AI-Face không phải unseen-generator split.
- Chưa claim identity-disjoint nếu metadata không đủ để xác minh.
- Kết quả pilot không được diễn giải thành generalization trên tất cả AI generators.

## 12. Trạng thái thực hiện (2026-10-13)

- Phase A — Audit annotation: **hoàn tất** trong `notebooks/09_aiface_dataset_audit.ipynb` và `docs/aiface-audit.md`.
- Phase B — Tạo subset: **hoàn tất**; subset hiện tại có 32.000 ảnh tại `data/AI_Face/aiface_pilot_32k/`.
- Exact duplicate repair: **hoàn tất**; đã thay 3 ảnh real bị trùng byte và lưu `audit/repair_report.json`.
- Phase C — Quality gate: **đạt**; `audit/quality_gate.json` ghi 32.000 file RGB đọc được, cân bằng lớp, không duplicate SHA-256 và generator-disjoint. DINOv3 smoke test **đã pass trên MPS** trong env `video-highlight`.
- Identity audit: **chưa đạt identity-disjointness**; số overlap identity heuristic được lưu trong `audit/quality_gate.json`. Không được gọi split này là identity-disjoint.
- Phase D — Kaggle package: **package và audit đã hoàn tất**; dataset shared dùng cho nhóm là `nguyentrann0703/aiface-pilot-32k-archive`.
- Phase D — Kaggle execution: **đã xác nhận thành công cho package single-bin**; dataset `nguyentrann0703/aiface-pilot-32k-single-bin` được giải nén vào workspace bởi runner và fixed manifests được đọc thành công. Dataset archive nhiều file trước đó vẫn từng gặp lỗi mount hạ tầng.
- Phase E — Main pilot: **đã hoàn tất cho seed 43 và seed 44**; mỗi seed có Global-only, Local-only, Forensic-only và ba late-probability fusions, cùng metric theo generator và rescue/harm. Seed 42 là run gốc và đang được chạy lại ba notebook để xác minh artifact đầy đủ.
- Phase F — Robustness, seed và báo cáo cuối: **code đã triển khai nhưng chưa chạy**. Còn robustness năm điều kiện, bootstrap CI 2.000 replicates, model selection validation-only và cập nhật báo cáo cuối.
- Tất cả kết quả chính dùng `generator_disjoint_fixed_v1`; không claim identity-disjoint vì audit identity chỉ là heuristic.
