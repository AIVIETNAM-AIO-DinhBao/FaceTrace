# AI-Face v2 Audit Report

Ngày audit: 2026-10-07

Notebook tái lập: [09_aiface_dataset_audit.ipynb](../notebooks/09_aiface_dataset_audit.ipynb)

## 1. Quyết định

**AI-Face v2 được chọn làm dataset chính của nghiên cứu hiện tại**, với điều kiện:

- main experiment chỉ dùng các subset `Real`, `GANs` và `DMs`;
- loại `deepfakes/` khỏi main experiment;
- không dùng trực tiếp split mặc định `train_data.csv`/`test_data.csv` cho unseen-generator evaluation;
- tạo lại generator-disjoint split;
- ghi rõ identity/source-disjointness chưa được xác minh đầy đủ từ annotation hiện tại.

Trong giai đoạn đầu, nhóm sử dụng subset `aiface_pilot_32k` theo kế hoạch tại [aiface-pilot-plan.md](aiface-pilot-plan.md).

Subset hiện đã được tạo và quality gate chạy lại sau khi sửa duplicate. Artifact chính:

- `data/AI_Face/aiface_pilot_32k/manifests/{train,val,test,all}.csv`;
- `data/AI_Face/aiface_pilot_32k/audit/quality_gate.json`;
- `data/AI_Face/aiface_pilot_32k/audit/duplicate_report.csv`;
- `data/AI_Face/aiface_pilot_32k/audit/repair_report.json`.

## 2. File đã kiểm tra

```text
data/AI_Face/AI_Face_annotationsV2/train_data.csv
data/AI_Face/AI_Face_annotationsV2/test_data.csv
data/AI_Face/OneDrive_1_10-7-2026.zip
data/AI_Face/OneDrive_2_10-7-2026.zip
data/AI_Face/imdb_wiki.zip
```

## 3. Kết quả annotation

| Split mặc định | Tổng ảnh | Fake | Real |
| --- | ---: | ---: | ---: |
| Train | 1.317.236 | 996.761 | 320.475 |
| Test | 329.309 | 248.899 | 80.410 |
| Tổng | 1.646.545 | 1.245.660 | 400.885 |

Annotation chứa đúng 18 generator/fake generator groups trong path:

- 10 GAN groups;
- 8 diffusion groups, trong đó `CommercialTools` được tách thành DALL-E2, IF và Midjourney.

Các generator đã chọn cho pilot đều có annotation đủ để lấy 2.000 ảnh:

| Family | Generator | Tổng ảnh annotation |
| --- | --- | ---: |
| GAN | `taming_transformer_VQGAN` | 50.000 |
| GAN | `stylegan3` | 26.770 |
| GAN | `AttGAN` | 6.005 |
| GAN | `STARGAN` | 5.648 |
| Diffusion | `StableDiffusion1.5` | 18.111 |
| Diffusion | `latent_diffusion` | 20.000 |
| Diffusion | `Palette` | 6.000 |
| Diffusion | `StableDiffusion_Inpainting` | 20.989 |

## 4. Kiểm tra split mặc định

Split mặc định **không phải generator-disjoint**.

- Exact image path overlap giữa train và test: `0`.
- Tất cả 8 generator pilot đều xuất hiện ở cả train và test.
- Tỷ lệ train/test của từng generator xấp xỉ 80/20.

Kết luận: `test_data.csv` hiện tại là image-level split phục vụ benchmark/fairness, không thể dùng trực tiếp làm unseen-generator test.

## 5. Kiểm tra archive

Đã kiểm tra outer archive:

- `OneDrive_1_10-7-2026.zip`: không có lỗi CRC;
- `OneDrive_2_10-7-2026.zip`: không có lỗi CRC;
- `imdb_wiki.zip`: không có lỗi CRC.

Tất cả inner archive cần cho pilot đều có mặt:

```text
OneDrive_1:
  taming_transformer_VQGAN.zip
  stylegan3.zip
  AttGAN.zip
  STARGAN.zip

OneDrive_2:
  StableDiffusion1.5.zip
  latent_diffusion.zip
  Palette.zip
  StableDiffusion_Inpainting.zip
```

Đây là archive-level audit. Ngoài ra, đã stream-extract một image mẫu từ cả 8 inner archive và một image từ `imdb_wiki`; tất cả đều được PIL đọc thành công.

Kích thước mẫu quan sát được gồm 200×200, 244×244 và 256×256, vì vậy preprocessing resize/normalization phải được cố định và ghi vào protocol.

Chưa kiểm tra toàn bộ image member trong từng inner archive.

## 6. Leakage và identity limitation

Exact path không bị trùng giữa train/test mặc định. Tuy nhiên, điều này không chứng minh identity-disjointness.

Annotation chỉ có các cột:

```text
Image Path
Predicted Gender
Predicted Age
Skin Tone
Target
Skin Tone Group
Intersection
```

Tên file `imdb_wiki` có dạng `nm...` hoặc numeric ID, nhưng schema không cung cấp identity ID chính thức. Vì vậy:

- có thể audit heuristic từ filename;
- chưa thể claim identity-disjoint một cách chắc chắn;
- pilot phải ghi rõ limitation này;
- cần chạy exact/perceptual duplicate audit sau khi extract subset.

## 7. Class và source risk

Toàn bộ annotation AI-Face bị lệch lớp về fake, khoảng 1.245M fake so với 401K real. Pilot sẽ cân bằng lại thành:

```text
16.000 fake + 16.000 real = 32.000 ảnh
```

Pilot chỉ dùng `imdb_wiki` làm real source để tiết kiệm dung lượng. Đây là một real-source limitation và có thể tạo shortcut giữa real-source với fake generator. FFHQ sẽ được xem xét ở follow-up, không tải toàn bộ ở giai đoạn pilot.

## 8. Kết quả quality gate của subset pilot

Sau khi thay 3 real images bị duplicate byte, quality gate cho kết quả:

| Kiểm tra | Kết quả |
| --- | ---: |
| Manifest rows | 32.000 |
| Train / validation / test | 16.000 / 8.000 / 8.000 |
| Fake / real trong mỗi split | 1:1 |
| Unique SHA-256 | 32.000 |
| Duplicate groups | 0 |
| File decode được bằng PIL | 32.000/32.000 |
| Mode ảnh | RGB |
| Generator overlap giữa split | 0 |

Ba thay thế được ghi trong `audit/repair_report.json`. Đây là sửa trên derived subset, không thay đổi archive/annotation gốc.

Identity heuristic vẫn có overlap giữa các split; subset vì vậy **không được gọi là identity-disjoint**. Real source cũng chỉ là `imdb_wiki`, nên kết quả pilot cần được diễn giải trong phạm vi này.

## 9. Điều kiện trước khi train

Dataset được xem là sẵn sàng cho smoke test sau khi hoàn thành các bước sau:

1. Extract một số image mẫu từ cả 8 inner archive. **Đã pass.**
2. Xác nhận local path trong manifest đọc được bằng PIL. **Đã pass trên toàn bộ 32.000 ảnh.**
3. Kiểm tra kích thước, mode màu, file hỏng và label. **Đã pass trên toàn bộ subset.**
4. Chạy duplicate audit trên subset. **Đã pass: 0 duplicate groups.**
5. Xác nhận manifest có đúng 2.000 fake/generator và 1:1 fake/real theo split. **Đã pass.**

Sau smoke test mẫu, AI-Face v2 được **phê duyệt làm dataset chính cho pilot experiments**. `aiface_pilot_32k` đã đạt các gate về file, class balance, exact duplicate và generator-disjointness, nên có thể chuyển sang đóng gói Kaggle. Tuy nhiên, identity-disjointness chưa được đảm bảo và real source chỉ là `imdb_wiki`; kết luận unseen-generator cuối cùng vẫn phải ghi rõ phạm vi: **unseen GAN/diffusion face generators trong AI-Face v2**, không phải mọi nguồn AI nói chung.
