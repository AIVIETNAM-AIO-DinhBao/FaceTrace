# AI-Face Complete Results — Phase E/F

**Mục đích:** tài liệu số liệu đầy đủ để viết paper về khả năng bổ sung của local representation và low-level spectral forensic evidence đối với global DINOv3 trên unseen AI-face generators.

**Trạng thái:** Phase E và Phase F đã hoàn tất trên pilot; tất cả bảng dưới đây được dựng từ artifact local đã kiểm tra. Các con số mô tả pilot này, không phải kết luận cho mọi generator.

## 1. Research question và phạm vi

> Các đặc trưng local và forensic mức thấp có cung cấp thông tin bổ sung cho biểu diễn toàn ảnh của DINOv3 khi tổng quát hóa sang nguồn khuôn mặt AI chưa xuất hiện trong huấn luyện hay không?

Phạm vi thực nghiệm là real face versus fully AI-generated face. Không đưa các deepfake-video manipulation subset vào main experiment.

## 2. Dataset và fixed split

Dataset: `nguyentrann0703/aiface-pilot-32k-single-bin`. Split mode: `generator_disjoint_fixed_v1`; test generators không xuất hiện trong train/validation. Test có 8.000 ảnh, gồm 4.000 real và 4.000 fake.

| Split | Real | Fake generators | Total |
|---|---|---|---|
| Train | 8,000 | StableDiffusion1.5 (2,000), latent_diffusion (2,000), stylegan3 (2,000), taming_transformer_VQGAN (2,000) | 16,000 |
| Validation | 4,000 | AttGAN (2,000), Palette (2,000) | 8,000 |
| Held-out test | 4,000 | STARGAN (2,000), StableDiffusion_Inpainting (2,000) | 8,000 |

Mỗi dòng per-generator test dùng chung 4.000 real test images; do đó mỗi generator row có `N=6,000` và không phải là một test set độc lập với các dòng khác.

### 2.1. Manifest metadata theo training seed

| Seed | Split mode | Test rows | Device | Conditions | Manifest SHA256 |
|---|---|---|---|---|---|
| 42 | generator_disjoint_fixed | 8000 | mps | clean, jpeg_q70, resize_112_224, blur_r1, center_crop_80 | train=691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88; val=65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac; test=b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf |
| 43 | generator_disjoint_fixed | 8000 | mps | clean, jpeg_q70, resize_112_224, blur_r1, center_crop_80 | train=691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88; val=65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac; test=b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf |
| 44 | generator_disjoint_fixed | 8000 | cuda | clean, jpeg_q70, resize_112_224, blur_r1, center_crop_80 | train=691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88; val=65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac; test=b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf |

Manifest SHA256 giống nhau giữa seed 42/43/44. `run_seed` chỉ thay đổi khởi tạo/thứ tự training; fixed manifest không được tạo lại.

## 3. Model và protocol

| Model | Definition |
|---|---|
| Global-only | Frozen DINOv3 global image representation + classifier |
| Local-only | Frozen DINOv3 patch representation, mean pooled + classifier |
| Global + Local | (p_global + p_local) / 2 |
| Forensic-only | RGB → Hann FFT2 → log magnitude → 64 radial bins/channel → 192-D vector → MLP 192→128→2 |
| Local + Forensic | (p_local + p_forensic) / 2 |
| Global + Local + Forensic | (p_global + p_local + p_forensic) / 3 |

Checkpoint và threshold được chọn trên validation; test chỉ được mở sau khi khóa checkpoint/threshold. Fusion dùng equal-weight late probability fusion, không train lại fusion classifier.

## 4. Phase E — clean unseen-generator results

Bảng dưới đây chứa toàn bộ clean test rows của ba seed. Các metric là trên cùng 8.000 test IDs; macro metric là trung bình theo STARGAN và StableDiffusion_Inpainting.

### 4.1. Clean macro/overall metrics theo seed

| Seed | Model | Threshold | N | Macro AUROC | Macro BAcc | Accuracy | F1 |
|---|---|---|---|---|---|---|---|
| 42 | Global-only | 0.000002 | 8000 | 0.939942 | 0.857625 | 0.857625 | 0.852671 |
| 42 | Local-only | 0.000889 | 8000 | 0.959689 | 0.883625 | 0.883625 | 0.885104 |
| 42 | Global + Local | 0.001555 | 8000 | 0.958677 | 0.877500 | 0.877500 | 0.873939 |
| 42 | Forensic-only | 0.472834 | 8000 | 0.958504 | 0.898875 | 0.898875 | 0.897763 |
| 42 | Local + Forensic | 0.313311 | 8000 | 0.981894 | 0.935625 | 0.935625 | 0.935163 |
| 42 | Global + Local + Forensic | 0.220520 | 8000 | 0.981606 | 0.940375 | 0.940375 | 0.940008 |
| 43 | Global-only | 0.001590 | 8000 | 0.928105 | 0.830875 | 0.830875 | 0.818365 |
| 43 | Local-only | 0.000193 | 8000 | 0.957311 | 0.880750 | 0.880750 | 0.879849 |
| 43 | Global + Local | 0.003223 | 8000 | 0.951657 | 0.869500 | 0.869500 | 0.866120 |
| 43 | Forensic-only | 0.626965 | 8000 | 0.974958 | 0.928500 | 0.928500 | 0.926384 |
| 43 | Local + Forensic | 0.370399 | 8000 | 0.987320 | 0.958375 | 0.958375 | 0.958380 |
| 43 | Global + Local + Forensic | 0.273983 | 8000 | 0.985797 | 0.958625 | 0.958625 | 0.958744 |
| 44 | Global-only | 0.000000 | 8000 | 0.937628 | 0.852625 | 0.852625 | 0.849866 |
| 44 | Local-only | 0.000236 | 8000 | 0.953547 | 0.868250 | 0.868250 | 0.868184 |
| 44 | Global + Local | 0.000436 | 8000 | 0.953908 | 0.869250 | 0.869250 | 0.865863 |
| 44 | Forensic-only | 0.508303 | 8000 | 0.975703 | 0.929500 | 0.929500 | 0.928896 |
| 44 | Local + Forensic | 0.353171 | 8000 | 0.988160 | 0.958875 | 0.958875 | 0.958756 |
| 44 | Global + Local + Forensic | 0.244771 | 8000 | 0.987310 | 0.959750 | 0.959750 | 0.959679 |

### 4.2. Mean ± sample SD qua seed — clean

| Condition | Model | Seeds | N seeds | Macro AUROC mean | Macro AUROC SD | Macro BAcc mean | Macro BAcc SD |
|---|---|---|---|---|---|---|---|
| clean | Forensic-only | 42,43,44 | 3 | 0.969722 | 0.009722 | 0.918958 | 0.017400 |
| clean | Global + Local | 42,43,44 | 3 | 0.954747 | 0.003585 | 0.872083 | 0.004693 |
| clean | Global + Local + Forensic | 42,43,44 | 3 | 0.984904 | 0.002955 | 0.952917 | 0.010876 |
| clean | Global-only | 42,43,44 | 3 | 0.935225 | 0.006273 | 0.847042 | 0.014222 |
| clean | Local + Forensic | 42,43,44 | 3 | 0.985791 | 0.003401 | 0.950958 | 0.013281 |
| clean | Local-only | 42,43,44 | 3 | 0.956849 | 0.003097 | 0.877542 | 0.008174 |

### 4.3. Validation metrics và threshold

| Seed | Model | Validation macro AUROC | Validation macro BAcc | Worst-generator AUROC | Threshold |
|---|---|---|---|---|---|
| 42 | Global-only | 0.955553 | 0.886750 | 0.941784 | 0.000002 |
| 42 | Local-only | 0.955251 | 0.883125 | 0.932381 | 0.000889 |
| 42 | Global + Local | 0.959663 | 0.891125 | 0.940677 | 0.001555 |
| 42 | Forensic-only | 0.961035 | 0.901125 | 0.947487 | 0.472834 |
| 42 | Local + Forensic | 0.981139 | 0.935875 | 0.979718 | 0.313311 |
| 42 | Global + Local + Forensic | 0.982982 | 0.941125 | 0.981805 | 0.220520 |
| 43 | Global-only | 0.953167 | 0.882500 | 0.935387 | 0.001590 |
| 43 | Local-only | 0.954517 | 0.884250 | 0.934599 | 0.000193 |
| 43 | Global + Local | 0.962447 | 0.898000 | 0.946659 | 0.003223 |
| 43 | Forensic-only | 0.979992 | 0.934375 | 0.968665 | 0.626965 |
| 43 | Local + Forensic | 0.988002 | 0.961250 | 0.986029 | 0.370399 |
| 43 | Global + Local + Forensic | 0.988380 | 0.962750 | 0.988206 | 0.273983 |
| 44 | Global-only | 0.955984 | 0.885000 | 0.940540 | 0.000000 |
| 44 | Local-only | 0.954696 | 0.881750 | 0.931142 | 0.000236 |
| 44 | Global + Local | 0.960393 | 0.892000 | 0.940815 | 0.000436 |
| 44 | Forensic-only | 0.975194 | 0.921750 | 0.964674 | 0.508305 |
| 44 | Local + Forensic | 0.988258 | 0.955250 | 0.987246 | 0.353171 |
| 44 | Global + Local + Forensic | 0.988233 | 0.957125 | 0.988204 | 0.244770 |

### 4.4. Validation-only model selection

| Rank | Model | Val AUROC mean | Val AUROC SD | Val BAcc mean | Val BAcc SD | Seeds | Worst-generator AUROC |
|---|---|---|---|---|---|---|---|
| 1 | Global + Local + Forensic | 0.986532 | 0.003075 | 0.953667 | 0.011220 | 42,43,44 | 0.986072 |
| 2 | Local + Forensic | 0.985800 | 0.004039 | 0.950792 | 0.013262 | 42,43,44 | 0.984331 |
| 3 | Forensic-only | 0.972074 | 0.009856 | 0.919083 | 0.016785 | 42,43,44 | 0.960275 |
| 4 | Global + Local | 0.960834 | 0.001444 | 0.893708 | 0.003742 | 42,43,44 | 0.942717 |
| 5 | Global-only | 0.954901 | 0.001517 | 0.884750 | 0.002136 | 42,43,44 | 0.939237 |
| 6 | Local-only | 0.954821 | 0.000383 | 0.883042 | 0.001252 | 42,43,44 | 0.932707 |

Model được chọn trước khi dùng test: **Global + Local + Forensic** (`global_local_forensic`). Test metric không tham gia model selection.

### 4.5. Shortcut diagnostic seed 42

| model | split | balanced_accuracy | accuracy | f1 | auroc |
|---|---|---|---|---|---|
| file_size_only | train | 0.876125 | 0.876125 | 0.859193 | 0.780116 |
| file_size_only | val | 0.747625 | 0.747625 | 0.664674 | 0.537581 |
| simple_features | train | 0.807313 | 0.807312 | 0.800130 | 0.860288 |
| simple_features | val | 0.638750 | 0.638750 | 0.550544 | 0.600553 |

Shortcut features có tín hiệu nhất định trên validation; vì vậy không diễn giải kết quả như bằng chứng thuần túy của semantic hoặc forensic artifact nếu chưa loại trừ real-source shortcut.

## 5. Phase F — robustness đầy đủ

Inference được chạy trên cùng test IDs với checkpoint đã khóa. Mỗi seed có 5 conditions × 6 models = 30 rows; tổng cộng 90 macro rows.

| Seed | Condition | Model | Threshold | N | Macro AUROC | Macro BAcc | Accuracy | F1 |
|---|---|---|---|---|---|---|---|---|
| 42 | clean | Global-only | 0.000002 | 8000 | 0.939942 | 0.857625 | 0.857625 | 0.852671 |
| 42 | clean | Local-only | 0.000889 | 8000 | 0.959689 | 0.883625 | 0.883625 | 0.885104 |
| 42 | clean | Global + Local | 0.001555 | 8000 | 0.958677 | 0.877500 | 0.877500 | 0.873939 |
| 42 | clean | Forensic-only | 0.472834 | 8000 | 0.958504 | 0.898875 | 0.898875 | 0.897763 |
| 42 | clean | Local + Forensic | 0.313311 | 8000 | 0.981894 | 0.935625 | 0.935625 | 0.935163 |
| 42 | clean | Global + Local + Forensic | 0.220520 | 8000 | 0.981606 | 0.940375 | 0.940375 | 0.940008 |
| 42 | JPEG quality 70 | Global-only | 0.000002 | 8000 | 0.890144 | 0.802875 | 0.802875 | 0.801560 |
| 42 | JPEG quality 70 | Local-only | 0.000889 | 8000 | 0.926863 | 0.840500 | 0.840500 | 0.845521 |
| 42 | JPEG quality 70 | Global + Local | 0.001555 | 8000 | 0.923416 | 0.836500 | 0.836500 | 0.835264 |
| 42 | JPEG quality 70 | Forensic-only | 0.472834 | 8000 | 0.483275 | 0.572500 | 0.572500 | 0.368304 |
| 42 | JPEG quality 70 | Local + Forensic | 0.313311 | 8000 | 0.699322 | 0.699375 | 0.699375 | 0.613157 |
| 42 | JPEG quality 70 | Global + Local + Forensic | 0.220520 | 8000 | 0.725529 | 0.722250 | 0.722250 | 0.654969 |
| 42 | resize 112→224 | Global-only | 0.000002 | 8000 | 0.915363 | 0.817625 | 0.817625 | 0.833618 |
| 42 | resize 112→224 | Local-only | 0.000889 | 8000 | 0.939209 | 0.818750 | 0.818750 | 0.841772 |
| 42 | resize 112→224 | Global + Local | 0.001555 | 8000 | 0.936921 | 0.837875 | 0.837875 | 0.853694 |
| 42 | resize 112→224 | Forensic-only | 0.472834 | 8000 | 0.706162 | 0.643000 | 0.643000 | 0.570268 |
| 42 | resize 112→224 | Local + Forensic | 0.313311 | 8000 | 0.882759 | 0.806125 | 0.806125 | 0.798545 |
| 42 | resize 112→224 | Global + Local + Forensic | 0.220520 | 8000 | 0.897054 | 0.819625 | 0.819625 | 0.817550 |
| 42 | Gaussian blur radius 1 | Global-only | 0.000002 | 8000 | 0.938406 | 0.841375 | 0.841375 | 0.827745 |
| 42 | Gaussian blur radius 1 | Local-only | 0.000889 | 8000 | 0.955240 | 0.877625 | 0.877625 | 0.878912 |
| 42 | Gaussian blur radius 1 | Global + Local | 0.001555 | 8000 | 0.954873 | 0.869875 | 0.869875 | 0.864928 |
| 42 | Gaussian blur radius 1 | Forensic-only | 0.472834 | 8000 | 0.682318 | 0.532250 | 0.532250 | 0.674722 |
| 42 | Gaussian blur radius 1 | Local + Forensic | 0.313311 | 8000 | 0.861306 | 0.563500 | 0.563500 | 0.690261 |
| 42 | Gaussian blur radius 1 | Global + Local + Forensic | 0.220520 | 8000 | 0.868091 | 0.571375 | 0.571375 | 0.693812 |
| 42 | center crop 80% | Global-only | 0.000002 | 8000 | 0.924003 | 0.814875 | 0.814875 | 0.836552 |
| 42 | center crop 80% | Local-only | 0.000889 | 8000 | 0.936000 | 0.757500 | 0.757500 | 0.804396 |
| 42 | center crop 80% | Global + Local | 0.001555 | 8000 | 0.936431 | 0.782250 | 0.782250 | 0.820301 |
| 42 | center crop 80% | Forensic-only | 0.472834 | 8000 | 0.858518 | 0.752875 | 0.752875 | 0.717208 |
| 42 | center crop 80% | Local + Forensic | 0.313311 | 8000 | 0.940672 | 0.874500 | 0.874500 | 0.884278 |
| 42 | center crop 80% | Global + Local + Forensic | 0.220520 | 8000 | 0.951969 | 0.875250 | 0.875250 | 0.885076 |
| 43 | clean | Global-only | 0.001590 | 8000 | 0.928105 | 0.830875 | 0.830875 | 0.818365 |
| 43 | clean | Local-only | 0.000193 | 8000 | 0.957311 | 0.880750 | 0.880750 | 0.879849 |
| 43 | clean | Global + Local | 0.003223 | 8000 | 0.951657 | 0.869500 | 0.869500 | 0.866120 |
| 43 | clean | Forensic-only | 0.626965 | 8000 | 0.974958 | 0.928500 | 0.928500 | 0.926384 |
| 43 | clean | Local + Forensic | 0.370399 | 8000 | 0.987320 | 0.958375 | 0.958375 | 0.958380 |
| 43 | clean | Global + Local + Forensic | 0.273983 | 8000 | 0.985797 | 0.958625 | 0.958625 | 0.958744 |
| 43 | JPEG quality 70 | Global-only | 0.001590 | 8000 | 0.872598 | 0.776625 | 0.776625 | 0.763593 |
| 43 | JPEG quality 70 | Local-only | 0.000193 | 8000 | 0.924127 | 0.835000 | 0.835000 | 0.834004 |
| 43 | JPEG quality 70 | Global + Local | 0.003223 | 8000 | 0.911043 | 0.822250 | 0.822250 | 0.821715 |
| 43 | JPEG quality 70 | Forensic-only | 0.626965 | 8000 | 0.446388 | 0.548750 | 0.548750 | 0.256590 |
| 43 | JPEG quality 70 | Local + Forensic | 0.370399 | 8000 | 0.723416 | 0.748000 | 0.748000 | 0.690418 |
| 43 | JPEG quality 70 | Global + Local + Forensic | 0.273983 | 8000 | 0.769159 | 0.762250 | 0.762250 | 0.720294 |
| 43 | resize 112→224 | Global-only | 0.001590 | 8000 | 0.893471 | 0.810000 | 0.810000 | 0.823625 |
| 43 | resize 112→224 | Local-only | 0.000193 | 8000 | 0.935620 | 0.837875 | 0.837875 | 0.852530 |
| 43 | resize 112→224 | Global + Local | 0.003223 | 8000 | 0.924893 | 0.825875 | 0.825875 | 0.842117 |
| 43 | resize 112→224 | Forensic-only | 0.626965 | 8000 | 0.705241 | 0.560125 | 0.560125 | 0.244039 |
| 43 | resize 112→224 | Local + Forensic | 0.370399 | 8000 | 0.897766 | 0.841000 | 0.841000 | 0.828155 |
| 43 | resize 112→224 | Global + Local + Forensic | 0.273983 | 8000 | 0.907046 | 0.840375 | 0.840375 | 0.837055 |
| 43 | Gaussian blur radius 1 | Global-only | 0.001590 | 8000 | 0.925563 | 0.809250 | 0.809250 | 0.784341 |
| 43 | Gaussian blur radius 1 | Local-only | 0.000193 | 8000 | 0.951813 | 0.868875 | 0.868875 | 0.867030 |
| 43 | Gaussian blur radius 1 | Global + Local | 0.003223 | 8000 | 0.948156 | 0.860125 | 0.860125 | 0.852976 |
| 43 | Gaussian blur radius 1 | Forensic-only | 0.626965 | 8000 | 0.674964 | 0.502875 | 0.502875 | 0.661733 |
| 43 | Gaussian blur radius 1 | Local + Forensic | 0.370399 | 8000 | 0.889146 | 0.524875 | 0.524875 | 0.675156 |
| 43 | Gaussian blur radius 1 | Global + Local + Forensic | 0.273983 | 8000 | 0.898811 | 0.542750 | 0.542750 | 0.683016 |
| 43 | center crop 80% | Global-only | 0.001590 | 8000 | 0.905679 | 0.805000 | 0.805000 | 0.824443 |
| 43 | center crop 80% | Local-only | 0.000193 | 8000 | 0.934395 | 0.774125 | 0.774125 | 0.814267 |
| 43 | center crop 80% | Global + Local | 0.003223 | 8000 | 0.927148 | 0.792625 | 0.792625 | 0.824834 |
| 43 | center crop 80% | Forensic-only | 0.626965 | 8000 | 0.832543 | 0.671625 | 0.671625 | 0.559450 |
| 43 | center crop 80% | Local + Forensic | 0.370399 | 8000 | 0.914750 | 0.864375 | 0.864375 | 0.875702 |
| 43 | center crop 80% | Global + Local + Forensic | 0.273983 | 8000 | 0.930665 | 0.863125 | 0.863125 | 0.875497 |
| 44 | clean | Global-only | 0.000000 | 8000 | 0.937628 | 0.852625 | 0.852625 | 0.849866 |
| 44 | clean | Local-only | 0.000236 | 8000 | 0.953547 | 0.868250 | 0.868250 | 0.868184 |
| 44 | clean | Global + Local | 0.000436 | 8000 | 0.953908 | 0.869250 | 0.869250 | 0.865863 |
| 44 | clean | Forensic-only | 0.508303 | 8000 | 0.975703 | 0.929500 | 0.929500 | 0.928896 |
| 44 | clean | Local + Forensic | 0.353171 | 8000 | 0.988160 | 0.958875 | 0.958875 | 0.958756 |
| 44 | clean | Global + Local + Forensic | 0.244771 | 8000 | 0.987310 | 0.959750 | 0.959750 | 0.959679 |
| 44 | JPEG quality 70 | Global-only | 0.000000 | 8000 | 0.891831 | 0.804000 | 0.804000 | 0.808920 |
| 44 | JPEG quality 70 | Local-only | 0.000236 | 8000 | 0.914579 | 0.819500 | 0.819500 | 0.821331 |
| 44 | JPEG quality 70 | Global + Local | 0.000436 | 8000 | 0.914151 | 0.824625 | 0.824625 | 0.824119 |
| 44 | JPEG quality 70 | Forensic-only | 0.508303 | 8000 | 0.450889 | 0.564875 | 0.564875 | 0.328640 |
| 44 | JPEG quality 70 | Local + Forensic | 0.353171 | 8000 | 0.681678 | 0.717750 | 0.717750 | 0.638604 |
| 44 | JPEG quality 70 | Global + Local + Forensic | 0.244771 | 8000 | 0.714851 | 0.739125 | 0.739125 | 0.679957 |
| 44 | resize 112→224 | Global-only | 0.000000 | 8000 | 0.910554 | 0.814250 | 0.814250 | 0.832431 |
| 44 | resize 112→224 | Local-only | 0.000236 | 8000 | 0.933747 | 0.819250 | 0.819250 | 0.840221 |
| 44 | resize 112→224 | Global + Local | 0.000436 | 8000 | 0.929593 | 0.828750 | 0.828750 | 0.846033 |
| 44 | resize 112→224 | Forensic-only | 0.508303 | 8000 | 0.695672 | 0.564500 | 0.564500 | 0.255874 |
| 44 | resize 112→224 | Local + Forensic | 0.353171 | 8000 | 0.896089 | 0.815375 | 0.815375 | 0.791943 |
| 44 | resize 112→224 | Global + Local + Forensic | 0.244771 | 8000 | 0.902087 | 0.826250 | 0.826250 | 0.812668 |
| 44 | Gaussian blur radius 1 | Global-only | 0.000000 | 8000 | 0.937926 | 0.847875 | 0.847875 | 0.839721 |
| 44 | Gaussian blur radius 1 | Local-only | 0.000236 | 8000 | 0.947905 | 0.863250 | 0.863250 | 0.862942 |
| 44 | Gaussian blur radius 1 | Global + Local | 0.000436 | 8000 | 0.948876 | 0.861500 | 0.861500 | 0.856588 |
| 44 | Gaussian blur radius 1 | Forensic-only | 0.508303 | 8000 | 0.689276 | 0.540625 | 0.540625 | 0.675783 |
| 44 | Gaussian blur radius 1 | Local + Forensic | 0.353171 | 8000 | 0.845109 | 0.594000 | 0.594000 | 0.698701 |
| 44 | Gaussian blur radius 1 | Global + Local + Forensic | 0.244771 | 8000 | 0.853421 | 0.604125 | 0.604125 | 0.702378 |
| 44 | center crop 80% | Global-only | 0.000000 | 8000 | 0.919677 | 0.797875 | 0.797875 | 0.825019 |
| 44 | center crop 80% | Local-only | 0.000236 | 8000 | 0.914838 | 0.758875 | 0.758875 | 0.804816 |
| 44 | center crop 80% | Global + Local | 0.000436 | 8000 | 0.922231 | 0.772750 | 0.772750 | 0.813155 |
| 44 | center crop 80% | Forensic-only | 0.508303 | 8000 | 0.821645 | 0.693125 | 0.693125 | 0.613325 |
| 44 | center crop 80% | Local + Forensic | 0.353171 | 8000 | 0.902123 | 0.849125 | 0.849125 | 0.861376 |
| 44 | center crop 80% | Global + Local + Forensic | 0.244771 | 8000 | 0.930424 | 0.847000 | 0.847000 | 0.860114 |

### 5.1. Delta so với clean

| Seed | Condition | Model | Δ AUROC vs clean | Δ BAcc vs clean |
|---|---|---|---|---|
| 42 | clean | Global-only | 0 | 0 |
| 42 | clean | Local-only | 0 | 0 |
| 42 | clean | Global + Local | 0 | 0 |
| 42 | clean | Forensic-only | 0 | 0 |
| 42 | clean | Local + Forensic | 0 | 0 |
| 42 | clean | Global + Local + Forensic | 0 | 0 |
| 42 | JPEG quality 70 | Global-only | -0.049798 | -0.054750 |
| 42 | JPEG quality 70 | Local-only | -0.032826 | -0.043125 |
| 42 | JPEG quality 70 | Global + Local | -0.035261 | -0.041000 |
| 42 | JPEG quality 70 | Forensic-only | -0.475229 | -0.326375 |
| 42 | JPEG quality 70 | Local + Forensic | -0.282572 | -0.236250 |
| 42 | JPEG quality 70 | Global + Local + Forensic | -0.256077 | -0.218125 |
| 42 | resize 112→224 | Global-only | -0.024578 | -0.040000 |
| 42 | resize 112→224 | Local-only | -0.020480 | -0.064875 |
| 42 | resize 112→224 | Global + Local | -0.021757 | -0.039625 |
| 42 | resize 112→224 | Forensic-only | -0.252342 | -0.255875 |
| 42 | resize 112→224 | Local + Forensic | -0.099134 | -0.129500 |
| 42 | resize 112→224 | Global + Local + Forensic | -0.084552 | -0.120750 |
| 42 | Gaussian blur radius 1 | Global-only | -0.001535 | -0.016250 |
| 42 | Gaussian blur radius 1 | Local-only | -0.004448 | -0.006000 |
| 42 | Gaussian blur radius 1 | Global + Local | -0.003805 | -0.007625 |
| 42 | Gaussian blur radius 1 | Forensic-only | -0.276186 | -0.366625 |
| 42 | Gaussian blur radius 1 | Local + Forensic | -0.120588 | -0.372125 |
| 42 | Gaussian blur radius 1 | Global + Local + Forensic | -0.113516 | -0.369000 |
| 42 | center crop 80% | Global-only | -0.015939 | -0.042750 |
| 42 | center crop 80% | Local-only | -0.023689 | -0.126125 |
| 42 | center crop 80% | Global + Local | -0.022247 | -0.095250 |
| 42 | center crop 80% | Forensic-only | -0.099986 | -0.146000 |
| 42 | center crop 80% | Local + Forensic | -0.041222 | -0.061125 |
| 42 | center crop 80% | Global + Local + Forensic | -0.029638 | -0.065125 |
| 43 | clean | Global-only | 0 | 0 |
| 43 | clean | Local-only | 0 | 0 |
| 43 | clean | Global + Local | 0 | 0 |
| 43 | clean | Forensic-only | 0 | 0 |
| 43 | clean | Local + Forensic | 0 | 0 |
| 43 | clean | Global + Local + Forensic | 0 | 0 |
| 43 | JPEG quality 70 | Global-only | -0.055508 | -0.054250 |
| 43 | JPEG quality 70 | Local-only | -0.033184 | -0.045750 |
| 43 | JPEG quality 70 | Global + Local | -0.040614 | -0.047250 |
| 43 | JPEG quality 70 | Forensic-only | -0.528570 | -0.379750 |
| 43 | JPEG quality 70 | Local + Forensic | -0.263904 | -0.210375 |
| 43 | JPEG quality 70 | Global + Local + Forensic | -0.216638 | -0.196375 |
| 43 | resize 112→224 | Global-only | -0.034635 | -0.020875 |
| 43 | resize 112→224 | Local-only | -0.021692 | -0.042875 |
| 43 | resize 112→224 | Global + Local | -0.026764 | -0.043625 |
| 43 | resize 112→224 | Forensic-only | -0.269717 | -0.368375 |
| 43 | resize 112→224 | Local + Forensic | -0.089554 | -0.117375 |
| 43 | resize 112→224 | Global + Local + Forensic | -0.078751 | -0.118250 |
| 43 | Gaussian blur radius 1 | Global-only | -0.002543 | -0.021625 |
| 43 | Gaussian blur radius 1 | Local-only | -0.005498 | -0.011875 |
| 43 | Gaussian blur radius 1 | Global + Local | -0.003501 | -0.009375 |
| 43 | Gaussian blur radius 1 | Forensic-only | -0.299994 | -0.425625 |
| 43 | Gaussian blur radius 1 | Local + Forensic | -0.098174 | -0.433500 |
| 43 | Gaussian blur radius 1 | Global + Local + Forensic | -0.086986 | -0.415875 |
| 43 | center crop 80% | Global-only | -0.022426 | -0.025875 |
| 43 | center crop 80% | Local-only | -0.022916 | -0.106625 |
| 43 | center crop 80% | Global + Local | -0.024509 | -0.076875 |
| 43 | center crop 80% | Forensic-only | -0.142415 | -0.256875 |
| 43 | center crop 80% | Local + Forensic | -0.072570 | -0.094000 |
| 43 | center crop 80% | Global + Local + Forensic | -0.055132 | -0.095500 |
| 44 | clean | Global-only | 0 | 0 |
| 44 | clean | Local-only | 0 | 0 |
| 44 | clean | Global + Local | 0 | 0 |
| 44 | clean | Forensic-only | 0 | 0 |
| 44 | clean | Local + Forensic | 0 | 0 |
| 44 | clean | Global + Local + Forensic | 0 | 0 |
| 44 | JPEG quality 70 | Global-only | -0.045797 | -0.048625 |
| 44 | JPEG quality 70 | Local-only | -0.038968 | -0.048750 |
| 44 | JPEG quality 70 | Global + Local | -0.039757 | -0.044625 |
| 44 | JPEG quality 70 | Forensic-only | -0.524815 | -0.364625 |
| 44 | JPEG quality 70 | Local + Forensic | -0.306482 | -0.241125 |
| 44 | JPEG quality 70 | Global + Local + Forensic | -0.272459 | -0.220625 |
| 44 | resize 112→224 | Global-only | -0.027073 | -0.038375 |
| 44 | resize 112→224 | Local-only | -0.019800 | -0.049000 |
| 44 | resize 112→224 | Global + Local | -0.024315 | -0.040500 |
| 44 | resize 112→224 | Forensic-only | -0.280032 | -0.365000 |
| 44 | resize 112→224 | Local + Forensic | -0.092071 | -0.143500 |
| 44 | resize 112→224 | Global + Local + Forensic | -0.085223 | -0.133500 |
| 44 | Gaussian blur radius 1 | Global-only | 0.000299 | -0.004750 |
| 44 | Gaussian blur radius 1 | Local-only | -0.005642 | -0.005000 |
| 44 | Gaussian blur radius 1 | Global + Local | -0.005032 | -0.007750 |
| 44 | Gaussian blur radius 1 | Forensic-only | -0.286428 | -0.388875 |
| 44 | Gaussian blur radius 1 | Local + Forensic | -0.143050 | -0.364875 |
| 44 | Gaussian blur radius 1 | Global + Local + Forensic | -0.133888 | -0.355625 |
| 44 | center crop 80% | Global-only | -0.017951 | -0.054750 |
| 44 | center crop 80% | Local-only | -0.038709 | -0.109375 |
| 44 | center crop 80% | Global + Local | -0.031677 | -0.096500 |
| 44 | center crop 80% | Forensic-only | -0.154059 | -0.236375 |
| 44 | center crop 80% | Local + Forensic | -0.086037 | -0.109750 |
| 44 | center crop 80% | Global + Local + Forensic | -0.056886 | -0.112750 |

### 5.2. Metric theo generator

Bảng sau có 180 dòng: 3 seed × 5 condition × 6 model × 2 held-out generators.

| Seed | Condition | Model | Generator | N | N real | N fake | AUROC | BAcc | Accuracy | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | clean | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.893705 | 0.784125 | 0.819833 | 0.714700 |
| 42 | clean | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.986179 | 0.931125 | 0.917833 | 0.887366 |
| 42 | clean | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.925259 | 0.836125 | 0.847667 | 0.778155 |
| 42 | clean | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994119 | 0.931125 | 0.911000 | 0.881333 |
| 42 | clean | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.923204 | 0.808875 | 0.841167 | 0.749277 |
| 42 | clean | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994151 | 0.946125 | 0.932667 | 0.907126 |
| 42 | clean | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.965203 | 0.910375 | 0.910167 | 0.871145 |
| 42 | clean | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.951805 | 0.887375 | 0.894833 | 0.845759 |
| 42 | clean | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.974465 | 0.922625 | 0.929333 | 0.894893 |
| 42 | clean | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.989323 | 0.948625 | 0.946667 | 0.922668 |
| 42 | clean | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.972128 | 0.924750 | 0.932000 | 0.898507 |
| 42 | clean | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.991085 | 0.956000 | 0.952833 | 0.931725 |
| 42 | JPEG quality 70 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.843493 | 0.750750 | 0.770333 | 0.667631 |
| 42 | JPEG quality 70 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.936795 | 0.855000 | 0.839833 | 0.789393 |
| 42 | JPEG quality 70 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.877137 | 0.788750 | 0.795167 | 0.714651 |
| 42 | JPEG quality 70 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.976588 | 0.892250 | 0.864167 | 0.827367 |
| 42 | JPEG quality 70 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.874186 | 0.768500 | 0.793667 | 0.691272 |
| 42 | JPEG quality 70 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.972646 | 0.904500 | 0.884333 | 0.847606 |
| 42 | JPEG quality 70 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.179354 | 0.450875 | 0.599167 | 0.009881 |
| 42 | JPEG quality 70 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.787196 | 0.694125 | 0.761333 | 0.579071 |
| 42 | JPEG quality 70 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.449444 | 0.528625 | 0.659833 | 0.209221 |
| 42 | JPEG quality 70 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.949200 | 0.870125 | 0.887500 | 0.828984 |
| 42 | JPEG quality 70 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.495681 | 0.560375 | 0.679333 | 0.297297 |
| 42 | JPEG quality 70 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.955377 | 0.884125 | 0.895167 | 0.844037 |
| 42 | resize 112→224 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.871866 | 0.789000 | 0.766500 | 0.709758 |
| 42 | resize 112→224 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.958861 | 0.846250 | 0.804667 | 0.768196 |
| 42 | resize 112→224 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.896883 | 0.802375 | 0.759333 | 0.720696 |
| 42 | resize 112→224 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.981535 | 0.835125 | 0.781167 | 0.752311 |
| 42 | resize 112→224 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.894819 | 0.812875 | 0.785167 | 0.735481 |
| 42 | resize 112→224 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.979022 | 0.862875 | 0.818500 | 0.785334 |
| 42 | resize 112→224 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.647394 | 0.590125 | 0.664167 | 0.422139 |
| 42 | resize 112→224 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.764930 | 0.695875 | 0.734667 | 0.592839 |
| 42 | resize 112→224 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.804177 | 0.721875 | 0.762500 | 0.627451 |
| 42 | resize 112→224 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.961341 | 0.890375 | 0.874833 | 0.833074 |
| 42 | resize 112→224 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.824829 | 0.745000 | 0.773667 | 0.659990 |
| 42 | resize 112→224 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.969278 | 0.894250 | 0.873167 | 0.834241 |
| 42 | Gaussian blur radius 1 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.891735 | 0.746750 | 0.804667 | 0.661663 |
| 42 | Gaussian blur radius 1 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.985078 | 0.936000 | 0.930833 | 0.901682 |
| 42 | Gaussian blur radius 1 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.916924 | 0.826500 | 0.840000 | 0.766082 |
| 42 | Gaussian blur radius 1 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.993557 | 0.928750 | 0.908167 | 0.877908 |
| 42 | Gaussian blur radius 1 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.915818 | 0.792500 | 0.830500 | 0.727419 |
| 42 | Gaussian blur radius 1 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.993927 | 0.947250 | 0.933667 | 0.908506 |
| 42 | Gaussian blur radius 1 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.648069 | 0.531625 | 0.385833 | 0.512631 |
| 42 | Gaussian blur radius 1 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.716567 | 0.532875 | 0.386667 | 0.513614 |
| 42 | Gaussian blur radius 1 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.760296 | 0.554625 | 0.421167 | 0.523790 |
| 42 | Gaussian blur radius 1 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.962316 | 0.572375 | 0.433000 | 0.538023 |
| 42 | Gaussian blur radius 1 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.765582 | 0.561000 | 0.431167 | 0.526958 |
| 42 | Gaussian blur radius 1 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.970599 | 0.581750 | 0.445000 | 0.543711 |
| 42 | center crop 80% | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.869298 | 0.792125 | 0.755500 | 0.710936 |
| 42 | center crop 80% | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.978708 | 0.837625 | 0.785833 | 0.755564 |
| 42 | center crop 80% | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.886277 | 0.756375 | 0.676833 | 0.672411 |
| 42 | center crop 80% | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.985723 | 0.758625 | 0.678333 | 0.674426 |
| 42 | center crop 80% | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.887909 | 0.779500 | 0.709833 | 0.694293 |
| 42 | center crop 80% | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.984952 | 0.785000 | 0.713500 | 0.699318 |
| 42 | center crop 80% | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.924983 | 0.830250 | 0.846500 | 0.772424 |
| 42 | center crop 80% | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.792053 | 0.675500 | 0.743333 | 0.550758 |
| 42 | center crop 80% | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.934808 | 0.856750 | 0.834500 | 0.788137 |
| 42 | center crop 80% | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.946536 | 0.892250 | 0.858167 | 0.823773 |
| 42 | center crop 80% | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.926471 | 0.857375 | 0.834833 | 0.788744 |
| 42 | center crop 80% | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.977466 | 0.893125 | 0.858667 | 0.824576 |
| 43 | clean | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.873294 | 0.732375 | 0.788167 | 0.640045 |
| 43 | clean | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.982917 | 0.929375 | 0.919500 | 0.888169 |
| 43 | clean | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.925301 | 0.826625 | 0.847167 | 0.769424 |
| 43 | clean | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.989322 | 0.934875 | 0.919333 | 0.890249 |
| 43 | clean | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.912522 | 0.802625 | 0.833333 | 0.739719 |
| 43 | clean | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.990792 | 0.936375 | 0.922500 | 0.893763 |
| 43 | clean | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.983660 | 0.948875 | 0.951667 | 0.928430 |
| 43 | clean | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.966256 | 0.908125 | 0.924500 | 0.883518 |
| 43 | clean | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.982966 | 0.956625 | 0.957167 | 0.936963 |
| 43 | clean | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.991674 | 0.960125 | 0.959500 | 0.940601 |
| 43 | clean | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.977917 | 0.952625 | 0.953667 | 0.931796 |
| 43 | clean | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.993678 | 0.964625 | 0.961667 | 0.944229 |
| 43 | JPEG quality 70 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.818241 | 0.704125 | 0.746667 | 0.602718 |
| 43 | JPEG quality 70 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.926954 | 0.849125 | 0.843333 | 0.786655 |
| 43 | JPEG quality 70 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.878252 | 0.770250 | 0.793833 | 0.693432 |
| 43 | JPEG quality 70 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.970003 | 0.899750 | 0.880167 | 0.842082 |
| 43 | JPEG quality 70 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.860213 | 0.759125 | 0.781167 | 0.678580 |
| 43 | JPEG quality 70 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.961872 | 0.885375 | 0.865333 | 0.823965 |
| 43 | JPEG quality 70 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.129563 | 0.471625 | 0.628333 | 0.002683 |
| 43 | JPEG quality 70 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.763213 | 0.625875 | 0.731167 | 0.434630 |
| 43 | JPEG quality 70 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.504654 | 0.619000 | 0.724000 | 0.423398 |
| 43 | JPEG quality 70 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.942179 | 0.877000 | 0.896000 | 0.840164 |
| 43 | JPEG quality 70 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.587907 | 0.637625 | 0.729167 | 0.471888 |
| 43 | JPEG quality 70 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.950410 | 0.886875 | 0.895333 | 0.845852 |
| 43 | resize 112→224 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.841547 | 0.771875 | 0.758833 | 0.691537 |
| 43 | resize 112→224 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.945395 | 0.848125 | 0.809667 | 0.771417 |
| 43 | resize 112→224 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.898758 | 0.810500 | 0.786500 | 0.733735 |
| 43 | resize 112→224 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.972482 | 0.865250 | 0.823000 | 0.788867 |
| 43 | resize 112→224 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.880634 | 0.796000 | 0.771667 | 0.717293 |
| 43 | resize 112→224 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.969152 | 0.855750 | 0.811500 | 0.777581 |
| 43 | resize 112→224 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.621411 | 0.532125 | 0.680833 | 0.152280 |
| 43 | resize 112→224 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.789071 | 0.588125 | 0.718167 | 0.318969 |
| 43 | resize 112→224 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.835052 | 0.766375 | 0.816167 | 0.691123 |
| 43 | resize 112→224 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.960481 | 0.915625 | 0.915667 | 0.878599 |
| 43 | resize 112→224 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.848507 | 0.773625 | 0.802667 | 0.698728 |
| 43 | resize 112→224 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.965585 | 0.907125 | 0.891667 | 0.854391 |
| 43 | Gaussian blur radius 1 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.869712 | 0.691875 | 0.769500 | 0.570363 |
| 43 | Gaussian blur radius 1 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.981414 | 0.926625 | 0.926000 | 0.893218 |
| 43 | Gaussian blur radius 1 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.915141 | 0.806875 | 0.832167 | 0.743831 |
| 43 | Gaussian blur radius 1 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.988486 | 0.930875 | 0.914833 | 0.884572 |
| 43 | Gaussian blur radius 1 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.905945 | 0.778875 | 0.822167 | 0.708709 |
| 43 | Gaussian blur radius 1 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.990367 | 0.941375 | 0.930500 | 0.903316 |
| 43 | Gaussian blur radius 1 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.675508 | 0.508375 | 0.350000 | 0.502170 |
| 43 | Gaussian blur radius 1 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.674420 | 0.497375 | 0.342667 | 0.493710 |
| 43 | Gaussian blur radius 1 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.814252 | 0.522125 | 0.368833 | 0.509138 |
| 43 | Gaussian blur radius 1 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.964040 | 0.527625 | 0.372500 | 0.513377 |
| 43 | Gaussian blur radius 1 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.819881 | 0.537875 | 0.392000 | 0.516821 |
| 43 | Gaussian blur radius 1 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.977741 | 0.547625 | 0.398500 | 0.524443 |
| 43 | center crop 80% | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.837481 | 0.766625 | 0.742500 | 0.684758 |
| 43 | center crop 80% | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.973877 | 0.843375 | 0.793667 | 0.762289 |
| 43 | center crop 80% | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.883066 | 0.769500 | 0.699000 | 0.684817 |
| 43 | center crop 80% | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.985724 | 0.778750 | 0.705167 | 0.693255 |
| 43 | center crop 80% | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.871979 | 0.781375 | 0.723833 | 0.697241 |
| 43 | center crop 80% | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.982318 | 0.803875 | 0.738833 | 0.718317 |
| 43 | center crop 80% | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.919888 | 0.797625 | 0.840500 | 0.736581 |
| 43 | center crop 80% | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.745198 | 0.545625 | 0.672500 | 0.251429 |
| 43 | center crop 80% | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.927305 | 0.846625 | 0.822167 | 0.775226 |
| 43 | center crop 80% | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.902196 | 0.882125 | 0.845833 | 0.810800 |
| 43 | center crop 80% | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.913817 | 0.847125 | 0.819333 | 0.774449 |
| 43 | center crop 80% | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.947514 | 0.879125 | 0.840667 | 0.806242 |
| 44 | clean | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.886377 | 0.779250 | 0.809833 | 0.706759 |
| 44 | clean | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.988878 | 0.926000 | 0.907667 | 0.876284 |
| 44 | clean | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.912989 | 0.805625 | 0.826667 | 0.740648 |
| 44 | clean | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994105 | 0.930875 | 0.910167 | 0.880514 |
| 44 | clean | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.912983 | 0.796000 | 0.828833 | 0.730941 |
| 44 | clean | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994833 | 0.942500 | 0.926500 | 0.899841 |
| 44 | clean | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.980907 | 0.940500 | 0.939667 | 0.912433 |
| 44 | clean | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.970500 | 0.918500 | 0.925000 | 0.888779 |
| 44 | clean | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.981576 | 0.948375 | 0.952833 | 0.929654 |
| 44 | clean | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994744 | 0.969375 | 0.966833 | 0.951546 |
| 44 | clean | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.979193 | 0.947500 | 0.952167 | 0.928625 |
| 44 | clean | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.995427 | 0.972000 | 0.968500 | 0.954115 |
| 44 | JPEG quality 70 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.836775 | 0.747625 | 0.757833 | 0.663735 |
| 44 | JPEG quality 70 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.946886 | 0.860375 | 0.833000 | 0.790025 |
| 44 | JPEG quality 70 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.854333 | 0.748375 | 0.768667 | 0.664572 |
| 44 | JPEG quality 70 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.974824 | 0.890625 | 0.863500 | 0.826004 |
| 44 | JPEG quality 70 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.856079 | 0.751500 | 0.776833 | 0.668646 |
| 44 | JPEG quality 70 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.972222 | 0.897750 | 0.874333 | 0.837008 |
| 44 | JPEG quality 70 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.130635 | 0.459125 | 0.611667 | 0.002568 |
| 44 | JPEG quality 70 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.771143 | 0.670625 | 0.752667 | 0.533627 |
| 44 | JPEG quality 70 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.404711 | 0.543875 | 0.674833 | 0.236399 |
| 44 | JPEG quality 70 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.958645 | 0.891625 | 0.906667 | 0.858084 |
| 44 | JPEG quality 70 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.465964 | 0.573500 | 0.690333 | 0.324364 |
| 44 | JPEG quality 70 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.963738 | 0.904750 | 0.911167 | 0.869202 |
| 44 | resize 112→224 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.859422 | 0.784875 | 0.758500 | 0.704587 |
| 44 | resize 112→224 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.961687 | 0.843625 | 0.797667 | 0.763813 |
| 44 | resize 112→224 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.885932 | 0.796250 | 0.760167 | 0.715444 |
| 44 | resize 112→224 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.981562 | 0.842250 | 0.790833 | 0.760542 |
| 44 | resize 112→224 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.880990 | 0.800500 | 0.772500 | 0.721599 |
| 44 | resize 112→224 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.978197 | 0.857000 | 0.810167 | 0.777929 |
| 44 | resize 112→224 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.592298 | 0.510125 | 0.666500 | 0.075751 |
| 44 | resize 112→224 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.799045 | 0.618875 | 0.739000 | 0.397692 |
| 44 | resize 112→224 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.817178 | 0.700250 | 0.776167 | 0.584596 |
| 44 | resize 112→224 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.974999 | 0.930500 | 0.929667 | 0.898411 |
| 44 | resize 112→224 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.828177 | 0.724125 | 0.782333 | 0.627283 |
| 44 | resize 112→224 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.975996 | 0.928375 | 0.918500 | 0.886832 |
| 44 | Gaussian blur radius 1 | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.887920 | 0.760375 | 0.806500 | 0.681831 |
| 44 | Gaussian blur radius 1 | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.987932 | 0.935375 | 0.923167 | 0.893999 |
| 44 | Gaussian blur radius 1 | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.902334 | 0.797500 | 0.820167 | 0.730048 |
| 44 | Gaussian blur radius 1 | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.993477 | 0.929000 | 0.907833 | 0.877736 |
| 44 | Gaussian blur radius 1 | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.903227 | 0.779125 | 0.818000 | 0.708177 |
| 44 | Gaussian blur radius 1 | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.994525 | 0.943875 | 0.927833 | 0.901613 |
| 44 | Gaussian blur radius 1 | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.611412 | 0.528625 | 0.393667 | 0.506511 |
| 44 | Gaussian blur radius 1 | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.767139 | 0.552625 | 0.409667 | 0.525710 |
| 44 | Gaussian blur radius 1 | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.711781 | 0.567250 | 0.460333 | 0.523122 |
| 44 | Gaussian blur radius 1 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.978437 | 0.620750 | 0.496000 | 0.568247 |
| 44 | Gaussian blur radius 1 | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.722625 | 0.573250 | 0.473500 | 0.524891 |
| 44 | Gaussian blur radius 1 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.984218 | 0.635000 | 0.514667 | 0.577726 |
| 44 | center crop 80% | Global-only | STARGAN | 6000 | 4000 | 2000 | 0.856610 | 0.776375 | 0.731833 | 0.693465 |
| 44 | center crop 80% | Global-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.982743 | 0.819375 | 0.760500 | 0.734920 |
| 44 | center crop 80% | Local-only | STARGAN | 6000 | 4000 | 2000 | 0.847433 | 0.757000 | 0.679167 | 0.673008 |
| 44 | center crop 80% | Local-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.982244 | 0.760750 | 0.681667 | 0.676381 |
| 44 | center crop 80% | Global + Local | STARGAN | 6000 | 4000 | 2000 | 0.859489 | 0.767750 | 0.697333 | 0.683182 |
| 44 | center crop 80% | Global + Local | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.984973 | 0.777750 | 0.704000 | 0.692308 |
| 44 | center crop 80% | Forensic-only | STARGAN | 6000 | 4000 | 2000 | 0.900562 | 0.779750 | 0.819667 | 0.709296 |
| 44 | center crop 80% | Forensic-only | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.742728 | 0.606500 | 0.704167 | 0.413998 |
| 44 | center crop 80% | Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.903070 | 0.820625 | 0.800667 | 0.746503 |
| 44 | center crop 80% | Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.901177 | 0.877625 | 0.838667 | 0.804286 |
| 44 | center crop 80% | Global + Local + Forensic | STARGAN | 6000 | 4000 | 2000 | 0.897879 | 0.819125 | 0.797167 | 0.744166 |
| 44 | center crop 80% | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 4000 | 2000 | 0.962968 | 0.874875 | 0.834333 | 0.800402 |

## 6. Bootstrap confidence intervals

Bootstrap dùng prediction đã khóa, không inference lại. Thiết lập: 2.000 replicates, bootstrap seed 2026, percentile CI 95%, stratified theo `generator_id × label`, shared indices cho paired differences.

Bootstrap metadata:

```json
{
  "phase_f_dirs": [
    "/Users/nguyentran0703/Downloads/FaceTrace/outputs/aiface_phase_f_seed42_complete",
    "/Users/nguyentran0703/Downloads/FaceTrace/outputs/aiface_phase_f_seed43",
    "/Users/nguyentran0703/Downloads/FaceTrace/outputs/aiface_phase_f_seed44"
  ],
  "seeds": [
    42,
    43,
    44
  ],
  "split_mode": "generator_disjoint_fixed",
  "manifest_sha256": [
    {
      "train": "691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88",
      "val": "65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac",
      "test": "b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf"
    },
    {
      "train": "691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88",
      "val": "65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac",
      "test": "b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf"
    },
    {
      "train": "691ab256db64729422b5c8e45f53f6e01c35c06dce16dca6bde1248121e27d88",
      "val": "65fdf40682bd2cc5b5ac8371dab3bee8049f5aa73c5d7660840741f38f5bf5ac",
      "test": "b26d47a377f90de699e4f0149470f225891a8fb42aea102278ccef6f32ca7caf"
    }
  ],
  "replicates": 2000,
  "bootstrap_seed": 2026,
  "confidence_level": 0.95,
  "scientific_claim_boundary": "conditional on fixed split and selected checkpoints"
}
```

### 6.1. CI theo model/condition

Bảng có 180 dòng; `Estimate` phải khớp point estimate trong `metrics_by_condition_model.csv`.

| Seed | Condition | Metric | Model | Estimate | CI 2.5% | CI 97.5% | Replicates | Bootstrap seed |
|---|---|---|---|---|---|---|---|---|
| 42 | clean | macro_auroc | Global-only | 0.939942 | 0.935363 | 0.944418 | 2000 | 2026 |
| 42 | clean | macro_auroc | Local-only | 0.959689 | 0.956273 | 0.963008 | 2000 | 2026 |
| 42 | clean | macro_auroc | Forensic-only | 0.958504 | 0.954459 | 0.962611 | 2000 | 2026 |
| 42 | clean | macro_auroc | Global + Local | 0.958677 | 0.955118 | 0.962052 | 2000 | 2026 |
| 42 | clean | macro_auroc | Local + Forensic | 0.981894 | 0.979472 | 0.984236 | 2000 | 2026 |
| 42 | clean | macro_auroc | Global + Local + Forensic | 0.981606 | 0.979113 | 0.984027 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Global-only | 0.857625 | 0.850500 | 0.864628 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Local-only | 0.883625 | 0.876750 | 0.890875 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Forensic-only | 0.898875 | 0.892125 | 0.905500 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Global + Local | 0.877500 | 0.870750 | 0.884375 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Local + Forensic | 0.935625 | 0.930250 | 0.941125 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | Global + Local + Forensic | 0.940375 | 0.935122 | 0.945500 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Global-only | 0.890144 | 0.883246 | 0.897031 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Local-only | 0.926863 | 0.921757 | 0.931974 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Forensic-only | 0.483275 | 0.474331 | 0.492435 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Global + Local | 0.923416 | 0.917881 | 0.928825 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Local + Forensic | 0.699322 | 0.690527 | 0.707799 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | Global + Local + Forensic | 0.725529 | 0.716884 | 0.734353 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Global-only | 0.802875 | 0.794872 | 0.811625 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Local-only | 0.840500 | 0.832997 | 0.848375 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Forensic-only | 0.572500 | 0.565500 | 0.579878 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Global + Local | 0.836500 | 0.829000 | 0.844131 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Local + Forensic | 0.699375 | 0.692500 | 0.706125 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | Global + Local + Forensic | 0.722250 | 0.714997 | 0.729375 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Global-only | 0.915363 | 0.909777 | 0.921010 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Local-only | 0.939209 | 0.934606 | 0.943772 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Forensic-only | 0.706162 | 0.694851 | 0.716791 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Global + Local | 0.936921 | 0.932244 | 0.941766 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Local + Forensic | 0.882759 | 0.876019 | 0.889229 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | Global + Local + Forensic | 0.897054 | 0.890833 | 0.903213 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Global-only | 0.817625 | 0.809750 | 0.825750 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Local-only | 0.818750 | 0.811122 | 0.826628 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Forensic-only | 0.643000 | 0.633125 | 0.652875 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Global + Local | 0.837875 | 0.830247 | 0.845750 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Local + Forensic | 0.806125 | 0.798250 | 0.814250 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | Global + Local + Forensic | 0.819625 | 0.812000 | 0.827750 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Global-only | 0.938406 | 0.933856 | 0.942926 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Local-only | 0.955240 | 0.951530 | 0.958755 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Forensic-only | 0.682318 | 0.670881 | 0.693558 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Global + Local | 0.954873 | 0.951157 | 0.958400 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Local + Forensic | 0.861306 | 0.853640 | 0.868847 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | Global + Local + Forensic | 0.868091 | 0.860524 | 0.875434 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Global-only | 0.841375 | 0.834625 | 0.848500 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Local-only | 0.877625 | 0.870500 | 0.884375 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Forensic-only | 0.532250 | 0.527000 | 0.537500 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local | 0.869875 | 0.863000 | 0.876625 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Local + Forensic | 0.563500 | 0.557750 | 0.569628 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local + Forensic | 0.571375 | 0.565250 | 0.577875 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Global-only | 0.924003 | 0.918831 | 0.928889 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Local-only | 0.936000 | 0.931277 | 0.940437 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Forensic-only | 0.858518 | 0.850909 | 0.866031 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Global + Local | 0.936431 | 0.931790 | 0.940983 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Local + Forensic | 0.940672 | 0.935486 | 0.945411 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | Global + Local + Forensic | 0.951969 | 0.947565 | 0.956089 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Global-only | 0.814875 | 0.807125 | 0.823000 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Local-only | 0.757500 | 0.749750 | 0.765625 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Forensic-only | 0.752875 | 0.744000 | 0.761250 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Global + Local | 0.782250 | 0.774375 | 0.790750 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Local + Forensic | 0.874500 | 0.867375 | 0.881375 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | Global + Local + Forensic | 0.875250 | 0.868125 | 0.882000 | 2000 | 2026 |
| 43 | clean | macro_auroc | Global-only | 0.928105 | 0.923282 | 0.932853 | 2000 | 2026 |
| 43 | clean | macro_auroc | Local-only | 0.957311 | 0.953736 | 0.960919 | 2000 | 2026 |
| 43 | clean | macro_auroc | Forensic-only | 0.974958 | 0.971929 | 0.977992 | 2000 | 2026 |
| 43 | clean | macro_auroc | Global + Local | 0.951657 | 0.947900 | 0.955342 | 2000 | 2026 |
| 43 | clean | macro_auroc | Local + Forensic | 0.987320 | 0.985084 | 0.989362 | 2000 | 2026 |
| 43 | clean | macro_auroc | Global + Local + Forensic | 0.985797 | 0.983583 | 0.987916 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Global-only | 0.830875 | 0.823500 | 0.838375 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Local-only | 0.880750 | 0.874000 | 0.887628 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Forensic-only | 0.928500 | 0.922875 | 0.934253 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Global + Local | 0.869500 | 0.862622 | 0.876375 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Local + Forensic | 0.958375 | 0.954125 | 0.962500 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | Global + Local + Forensic | 0.958625 | 0.954372 | 0.962753 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Global-only | 0.872598 | 0.865281 | 0.879870 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Local-only | 0.924127 | 0.918746 | 0.929314 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Forensic-only | 0.446388 | 0.438119 | 0.454921 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Global + Local | 0.911043 | 0.905088 | 0.916685 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Local + Forensic | 0.723416 | 0.713455 | 0.732778 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | Global + Local + Forensic | 0.769159 | 0.759969 | 0.778337 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Global-only | 0.776625 | 0.768250 | 0.785378 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Local-only | 0.835000 | 0.827250 | 0.842750 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Forensic-only | 0.548750 | 0.542750 | 0.554750 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Global + Local | 0.822250 | 0.814125 | 0.830375 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Local + Forensic | 0.748000 | 0.740250 | 0.755128 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | Global + Local + Forensic | 0.762250 | 0.754625 | 0.769878 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Global-only | 0.893471 | 0.886972 | 0.899865 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Local-only | 0.935620 | 0.930723 | 0.940399 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Forensic-only | 0.705241 | 0.694150 | 0.715804 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Global + Local | 0.924893 | 0.919528 | 0.929982 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Local + Forensic | 0.897766 | 0.890926 | 0.904374 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | Global + Local + Forensic | 0.907046 | 0.900841 | 0.912995 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Global-only | 0.810000 | 0.801625 | 0.818125 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Local-only | 0.837875 | 0.830622 | 0.845875 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Forensic-only | 0.560125 | 0.554375 | 0.566000 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Global + Local | 0.825875 | 0.817872 | 0.833753 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Local + Forensic | 0.841000 | 0.833500 | 0.848003 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | Global + Local + Forensic | 0.840375 | 0.832625 | 0.848125 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Global-only | 0.925563 | 0.920575 | 0.930571 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Local-only | 0.951813 | 0.947922 | 0.955600 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Forensic-only | 0.674964 | 0.663367 | 0.686551 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Global + Local | 0.948156 | 0.944127 | 0.952083 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Local + Forensic | 0.889146 | 0.881541 | 0.896061 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | Global + Local + Forensic | 0.898811 | 0.891763 | 0.905254 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Global-only | 0.809250 | 0.802125 | 0.816500 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Local-only | 0.868875 | 0.861625 | 0.875878 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Forensic-only | 0.502875 | 0.499250 | 0.506750 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local | 0.860125 | 0.853372 | 0.866875 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Local + Forensic | 0.524875 | 0.520875 | 0.529125 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local + Forensic | 0.542750 | 0.538000 | 0.547875 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Global-only | 0.905679 | 0.899674 | 0.911158 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Local-only | 0.934395 | 0.929704 | 0.939222 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Forensic-only | 0.832543 | 0.824025 | 0.840560 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Global + Local | 0.927148 | 0.921884 | 0.931939 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Local + Forensic | 0.914750 | 0.907927 | 0.920854 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | Global + Local + Forensic | 0.930665 | 0.924960 | 0.936061 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Global-only | 0.805000 | 0.796500 | 0.813125 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Local-only | 0.774125 | 0.766247 | 0.782000 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Forensic-only | 0.671625 | 0.664125 | 0.679250 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Global + Local | 0.792625 | 0.784625 | 0.800503 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Local + Forensic | 0.864375 | 0.857122 | 0.871375 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | Global + Local + Forensic | 0.863125 | 0.855625 | 0.870375 | 2000 | 2026 |
| 44 | clean | macro_auroc | Global-only | 0.937628 | 0.933233 | 0.942124 | 2000 | 2026 |
| 44 | clean | macro_auroc | Local-only | 0.953547 | 0.949955 | 0.957164 | 2000 | 2026 |
| 44 | clean | macro_auroc | Forensic-only | 0.975703 | 0.972707 | 0.978780 | 2000 | 2026 |
| 44 | clean | macro_auroc | Global + Local | 0.953908 | 0.950202 | 0.957572 | 2000 | 2026 |
| 44 | clean | macro_auroc | Local + Forensic | 0.988160 | 0.986108 | 0.990158 | 2000 | 2026 |
| 44 | clean | macro_auroc | Global + Local + Forensic | 0.987310 | 0.985148 | 0.989390 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Global-only | 0.852625 | 0.845125 | 0.860250 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Local-only | 0.868250 | 0.861250 | 0.875253 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Forensic-only | 0.929500 | 0.923875 | 0.935250 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Global + Local | 0.869250 | 0.862125 | 0.876000 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Local + Forensic | 0.958875 | 0.954625 | 0.963000 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | Global + Local + Forensic | 0.959750 | 0.955625 | 0.963753 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Global-only | 0.891831 | 0.885433 | 0.898276 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Local-only | 0.914579 | 0.909069 | 0.920130 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Forensic-only | 0.450889 | 0.442527 | 0.459548 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Global + Local | 0.914151 | 0.908620 | 0.919847 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Local + Forensic | 0.681678 | 0.672981 | 0.690451 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | Global + Local + Forensic | 0.714851 | 0.706157 | 0.723955 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Global-only | 0.804000 | 0.795625 | 0.812253 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Local-only | 0.819500 | 0.811622 | 0.827750 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Forensic-only | 0.564875 | 0.557997 | 0.571250 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Global + Local | 0.824625 | 0.816622 | 0.832628 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Local + Forensic | 0.717750 | 0.711000 | 0.724250 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | Global + Local + Forensic | 0.739125 | 0.732000 | 0.746000 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Global-only | 0.910554 | 0.905057 | 0.916443 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Local-only | 0.933747 | 0.928998 | 0.938428 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Forensic-only | 0.695672 | 0.684395 | 0.706766 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Global + Local | 0.929593 | 0.924731 | 0.934530 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Local + Forensic | 0.896089 | 0.889724 | 0.902325 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | Global + Local + Forensic | 0.902087 | 0.895959 | 0.908258 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Global-only | 0.814250 | 0.806497 | 0.822503 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Local-only | 0.819250 | 0.811625 | 0.827503 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Forensic-only | 0.564500 | 0.559125 | 0.570250 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Global + Local | 0.828750 | 0.820872 | 0.837003 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Local + Forensic | 0.815375 | 0.808375 | 0.822125 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | Global + Local + Forensic | 0.826250 | 0.819125 | 0.834000 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Global-only | 0.937926 | 0.933610 | 0.942391 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Local-only | 0.947905 | 0.943972 | 0.951690 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Forensic-only | 0.689276 | 0.678197 | 0.700549 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Global + Local | 0.948876 | 0.944960 | 0.952627 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Local + Forensic | 0.845109 | 0.837733 | 0.852598 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | Global + Local + Forensic | 0.853421 | 0.846290 | 0.860821 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Global-only | 0.847875 | 0.840625 | 0.854878 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Local-only | 0.863250 | 0.856000 | 0.870503 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Forensic-only | 0.540625 | 0.534750 | 0.546750 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local | 0.861500 | 0.854250 | 0.868500 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Local + Forensic | 0.594000 | 0.586750 | 0.601253 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | Global + Local + Forensic | 0.604125 | 0.596500 | 0.611750 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Global-only | 0.919677 | 0.914720 | 0.924672 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Local-only | 0.914838 | 0.909471 | 0.920086 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Forensic-only | 0.821645 | 0.813156 | 0.830360 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Global + Local | 0.922231 | 0.917154 | 0.927185 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Local + Forensic | 0.902123 | 0.894995 | 0.908712 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | Global + Local + Forensic | 0.930424 | 0.924845 | 0.935476 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Global-only | 0.797875 | 0.790000 | 0.805625 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Local-only | 0.758875 | 0.751125 | 0.767128 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Forensic-only | 0.693125 | 0.684375 | 0.701878 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Global + Local | 0.772750 | 0.764875 | 0.780875 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Local + Forensic | 0.849125 | 0.841625 | 0.856625 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | Global + Local + Forensic | 0.847000 | 0.839375 | 0.854500 | 2000 | 2026 |

### 6.2. Paired-difference CI

Bảng có 90 dòng. Delta dương nghĩa là model ở vế trái tốt hơn model ở vế phải theo metric tương ứng.

| Seed | Condition | Metric | Comparison | Delta estimate | CI 2.5% | CI 97.5% | Replicates | Bootstrap seed |
|---|---|---|---|---|---|---|---|---|
| 42 | clean | macro_auroc | local_forensic-local_only | 0.022205 | 0.018511 | 0.026040 | 2000 | 2026 |
| 42 | clean | macro_auroc | global_local_forensic-global_local | 0.022929 | 0.019436 | 0.026488 | 2000 | 2026 |
| 42 | clean | macro_auroc | forensic_only-local_only | -0.001185 | -0.006157 | 0.004097 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | local_forensic-local_only | 0.052000 | 0.044125 | 0.060250 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | global_local_forensic-global_local | 0.062875 | 0.055372 | 0.070875 | 2000 | 2026 |
| 42 | clean | macro_balanced_accuracy | forensic_only-local_only | 0.015250 | 0.006000 | 0.024631 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | local_forensic-local_only | -0.227540 | -0.235552 | -0.219535 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | global_local_forensic-global_local | -0.197887 | -0.205370 | -0.190365 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_auroc | forensic_only-local_only | -0.443587 | -0.453383 | -0.433992 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | local_forensic-local_only | -0.141125 | -0.150625 | -0.131625 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | global_local_forensic-global_local | -0.114250 | -0.123628 | -0.105000 | 2000 | 2026 |
| 42 | JPEG quality 70 | macro_balanced_accuracy | forensic_only-local_only | -0.268000 | -0.278500 | -0.257625 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | local_forensic-local_only | -0.056450 | -0.061974 | -0.050931 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | global_local_forensic-global_local | -0.039867 | -0.044647 | -0.035269 | 2000 | 2026 |
| 42 | resize 112→224 | macro_auroc | forensic_only-local_only | -0.233047 | -0.244410 | -0.222056 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | local_forensic-local_only | -0.012625 | -0.023003 | -0.002500 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | global_local_forensic-global_local | -0.018250 | -0.027375 | -0.008997 | 2000 | 2026 |
| 42 | resize 112→224 | macro_balanced_accuracy | forensic_only-local_only | -0.175750 | -0.188250 | -0.163872 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | local_forensic-local_only | -0.093934 | -0.101371 | -0.086496 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | global_local_forensic-global_local | -0.086782 | -0.093955 | -0.079589 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_auroc | forensic_only-local_only | -0.272922 | -0.284284 | -0.261399 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | local_forensic-local_only | -0.314125 | -0.322875 | -0.304500 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | global_local_forensic-global_local | -0.298500 | -0.307625 | -0.289250 | 2000 | 2026 |
| 42 | Gaussian blur radius 1 | macro_balanced_accuracy | forensic_only-local_only | -0.345375 | -0.354128 | -0.336497 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | local_forensic-local_only | 0.004672 | 0.001130 | 0.008107 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | global_local_forensic-global_local | 0.015538 | 0.013167 | 0.017807 | 2000 | 2026 |
| 42 | center crop 80% | macro_auroc | forensic_only-local_only | -0.077482 | -0.085410 | -0.069727 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | local_forensic-local_only | 0.117000 | 0.109000 | 0.125128 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | global_local_forensic-global_local | 0.093000 | 0.085372 | 0.100753 | 2000 | 2026 |
| 42 | center crop 80% | macro_balanced_accuracy | forensic_only-local_only | -0.004625 | -0.015750 | 0.006003 | 2000 | 2026 |
| 43 | clean | macro_auroc | local_forensic-local_only | 0.030009 | 0.026737 | 0.033451 | 2000 | 2026 |
| 43 | clean | macro_auroc | global_local_forensic-global_local | 0.034140 | 0.030948 | 0.037436 | 2000 | 2026 |
| 43 | clean | macro_auroc | forensic_only-local_only | 0.017646 | 0.013398 | 0.022175 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | local_forensic-local_only | 0.077625 | 0.070372 | 0.085003 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | global_local_forensic-global_local | 0.089125 | 0.082250 | 0.095875 | 2000 | 2026 |
| 43 | clean | macro_balanced_accuracy | forensic_only-local_only | 0.047750 | 0.039497 | 0.056750 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | local_forensic-local_only | -0.200711 | -0.208863 | -0.192947 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | global_local_forensic-global_local | -0.141884 | -0.148631 | -0.134912 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_auroc | forensic_only-local_only | -0.477739 | -0.486961 | -0.468074 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | local_forensic-local_only | -0.087000 | -0.095750 | -0.078500 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | global_local_forensic-global_local | -0.060000 | -0.068125 | -0.052000 | 2000 | 2026 |
| 43 | JPEG quality 70 | macro_balanced_accuracy | forensic_only-local_only | -0.286250 | -0.296253 | -0.276872 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | local_forensic-local_only | -0.037854 | -0.042541 | -0.033543 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | global_local_forensic-global_local | -0.017847 | -0.021023 | -0.014690 | 2000 | 2026 |
| 43 | resize 112→224 | macro_auroc | forensic_only-local_only | -0.230379 | -0.241994 | -0.219367 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | local_forensic-local_only | 0.003125 | -0.005875 | 0.011375 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | global_local_forensic-global_local | 0.014500 | 0.007125 | 0.022000 | 2000 | 2026 |
| 43 | resize 112→224 | macro_balanced_accuracy | forensic_only-local_only | -0.277750 | -0.287375 | -0.268747 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | local_forensic-local_only | -0.062667 | -0.068996 | -0.056339 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | global_local_forensic-global_local | -0.049345 | -0.055071 | -0.043603 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_auroc | forensic_only-local_only | -0.276849 | -0.288976 | -0.265201 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | local_forensic-local_only | -0.344000 | -0.351753 | -0.336125 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | global_local_forensic-global_local | -0.317375 | -0.325503 | -0.308875 | 2000 | 2026 |
| 43 | Gaussian blur radius 1 | macro_balanced_accuracy | forensic_only-local_only | -0.366000 | -0.373750 | -0.357750 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | local_forensic-local_only | -0.019645 | -0.024202 | -0.015186 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | global_local_forensic-global_local | 0.003517 | 0.000528 | 0.006478 | 2000 | 2026 |
| 43 | center crop 80% | macro_auroc | forensic_only-local_only | -0.101852 | -0.109735 | -0.093946 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | local_forensic-local_only | 0.090250 | 0.082747 | 0.097753 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | global_local_forensic-global_local | 0.070500 | 0.064000 | 0.077625 | 2000 | 2026 |
| 43 | center crop 80% | macro_balanced_accuracy | forensic_only-local_only | -0.102500 | -0.113000 | -0.091875 | 2000 | 2026 |
| 44 | clean | macro_auroc | local_forensic-local_only | 0.034613 | 0.030874 | 0.038291 | 2000 | 2026 |
| 44 | clean | macro_auroc | global_local_forensic-global_local | 0.033401 | 0.029919 | 0.036846 | 2000 | 2026 |
| 44 | clean | macro_auroc | forensic_only-local_only | 0.022156 | 0.017596 | 0.026668 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | local_forensic-local_only | 0.090625 | 0.082750 | 0.098375 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | global_local_forensic-global_local | 0.090500 | 0.083125 | 0.097750 | 2000 | 2026 |
| 44 | clean | macro_balanced_accuracy | forensic_only-local_only | 0.061250 | 0.052375 | 0.070000 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | local_forensic-local_only | -0.232901 | -0.240645 | -0.225180 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | global_local_forensic-global_local | -0.199300 | -0.206789 | -0.191718 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_auroc | forensic_only-local_only | -0.463690 | -0.473372 | -0.454394 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | local_forensic-local_only | -0.101750 | -0.110750 | -0.092372 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | global_local_forensic-global_local | -0.085500 | -0.094000 | -0.076750 | 2000 | 2026 |
| 44 | JPEG quality 70 | macro_balanced_accuracy | forensic_only-local_only | -0.254625 | -0.265250 | -0.244372 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | local_forensic-local_only | -0.037658 | -0.041872 | -0.033537 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | global_local_forensic-global_local | -0.027507 | -0.031026 | -0.024240 | 2000 | 2026 |
| 44 | resize 112→224 | macro_auroc | forensic_only-local_only | -0.238075 | -0.249214 | -0.227289 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | local_forensic-local_only | -0.003875 | -0.012750 | 0.005250 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | global_local_forensic-global_local | -0.002500 | -0.010878 | 0.005878 | 2000 | 2026 |
| 44 | resize 112→224 | macro_balanced_accuracy | forensic_only-local_only | -0.254750 | -0.264375 | -0.245122 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | local_forensic-local_only | -0.102796 | -0.109784 | -0.095771 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | global_local_forensic-global_local | -0.095455 | -0.102203 | -0.088650 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_auroc | forensic_only-local_only | -0.258630 | -0.269948 | -0.247277 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | local_forensic-local_only | -0.269250 | -0.279125 | -0.260000 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | global_local_forensic-global_local | -0.257375 | -0.266503 | -0.248122 | 2000 | 2026 |
| 44 | Gaussian blur radius 1 | macro_balanced_accuracy | forensic_only-local_only | -0.322625 | -0.331500 | -0.313750 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | local_forensic-local_only | -0.012715 | -0.016935 | -0.008434 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | global_local_forensic-global_local | 0.008193 | 0.005635 | 0.010690 | 2000 | 2026 |
| 44 | center crop 80% | macro_auroc | forensic_only-local_only | -0.093193 | -0.101673 | -0.084850 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | local_forensic-local_only | 0.090250 | 0.082500 | 0.098375 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | global_local_forensic-global_local | 0.074250 | 0.067000 | 0.081625 | 2000 | 2026 |
| 44 | center crop 80% | macro_balanced_accuracy | forensic_only-local_only | -0.065750 | -0.076625 | -0.054625 | 2000 | 2026 |

## 7. Rescue/harm của forensic fusion

`Rescue` là baseline sai nhưng fusion đúng; `Harm` là baseline đúng nhưng fusion sai; `Net gain = rescue − harm`. Các bảng dưới đây dùng threshold validation riêng của từng model.

| Seed | Fusion | Group | N | Rescue | Harm | Net gain | Both correct | Both wrong | Baseline threshold | Fusion threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | Local + Forensic | overall | 8000 | 798 | 382 | 416 | 6687 | 133 | 0.000889 | 0.313311 |
| 42 | Local + Forensic | STARGAN | 6000 | 787 | 297 | 490 | 4789 | 127 | 0.000889 | 0.313311 |
| 42 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 452 | 238 | 214 | 5228 | 82 | 0.000889 | 0.313311 |
| 42 | Global + Local + Forensic | overall | 8000 | 806 | 303 | 503 | 6717 | 174 | 0.001555 | 0.220520 |
| 42 | Global + Local + Forensic | STARGAN | 6000 | 794 | 249 | 545 | 4798 | 159 | 0.001555 | 0.220520 |
| 42 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 314 | 193 | 121 | 5403 | 90 | 0.001555 | 0.220520 |
| 43 | Local + Forensic | overall | 8000 | 815 | 194 | 621 | 6852 | 139 | 0.000193 | 0.370399 |
| 43 | Local + Forensic | STARGAN | 6000 | 786 | 126 | 660 | 4957 | 131 | 0.000193 | 0.370399 |
| 43 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 387 | 146 | 241 | 5370 | 97 | 0.000193 | 0.370399 |
| 43 | Global + Local + Forensic | overall | 8000 | 839 | 126 | 713 | 6830 | 205 | 0.003223 | 0.273983 |
| 43 | Global + Local + Forensic | STARGAN | 6000 | 812 | 90 | 722 | 4910 | 188 | 0.003223 | 0.273983 |
| 43 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 319 | 84 | 235 | 5451 | 146 | 0.003223 | 0.273983 |
| 44 | Local + Forensic | overall | 8000 | 936 | 211 | 725 | 6735 | 118 | 0.000236 | 0.353171 |
| 44 | Local + Forensic | STARGAN | 6000 | 926 | 169 | 757 | 4791 | 114 | 0.000236 | 0.353171 |
| 44 | Local + Forensic | StableDiffusion_Inpainting | 6000 | 470 | 130 | 340 | 5331 | 69 | 0.000236 | 0.353171 |
| 44 | Global + Local + Forensic | overall | 8000 | 902 | 178 | 724 | 6776 | 144 | 0.000436 | 0.244770 |
| 44 | Global + Local + Forensic | STARGAN | 6000 | 889 | 149 | 740 | 4824 | 138 | 0.000436 | 0.244770 |
| 44 | Global + Local + Forensic | StableDiffusion_Inpainting | 6000 | 357 | 105 | 252 | 5454 | 84 | 0.000436 | 0.244770 |

Net gain được báo cáo theo seed và group; không cộng các generator rows để suy ra overall vì 4.000 real images được dùng chung giữa hai generator groups.

## 8. Kết luận có thể dùng trong paper

1. Local-only vượt Global-only trên clean held-out test ở macro AUROC và macro BAcc trong cả ba seed; vì vậy global semantic representation đơn lẻ chưa phải baseline mạnh nhất trên pilot này.
2. Forensic-only có hiệu năng clean cao nhưng biến thiên qua seed lớn hơn các nhánh Local/fusion; điều này phù hợp với việc spectral signal nhạy với training initialization và degradation.
3. Local + Forensic và Global + Local + Forensic đều cho paired gain dương trên clean test; tuy nhiên fusion ba nhánh không thắng fusion hai nhánh ở mọi metric, nên cần báo cáo AUROC và BAcc riêng.
4. Robustness phải đọc theo từng condition: forensic spectral có thể giảm mạnh dưới JPEG/blur, trong khi Global/Local ổn định hơn. Không gộp các condition thành một điểm duy nhất.
5. Các kết luận chỉ áp dụng cho hai held-out generators của pilot: STARGAN và StableDiffusion_Inpainting. Không được viết thành tuyên bố tổng quát cho toàn bộ GAN/diffusion hoặc mọi AI generator.

## 9. Giới hạn và nguy cơ diễn giải

- Pilot dùng 2.000 fake ảnh mỗi generator và chỉ có hai held-out generators.
- Real pool có nguy cơ real-source shortcut; shortcut diagnostic validation AUROC không bằng random.
- Generator-disjointness được đảm bảo theo manifest; identity-disjointness chưa được chứng minh đầy đủ từ metadata công khai.
- Bootstrap CI phản ánh uncertainty do resampling test images trên fixed split/checkpoints; không phải uncertainty trên tập generator mới hay uncertainty do retraining.
- Mean ± SD qua ba seed mô tả training variation, không phải confidence interval của population-level performance.
- Equal-weight late fusion chưa phải calibrated/learned fusion; chưa đánh giá weighted fusion hoặc calibration.

## 10. Provenance map

| Evidence ID | Evidence group | Path | Used for |
|---|---|---|---|
| E01 | Dataset/split | outputs/aiface_phase_f_seed42_complete/run_metadata.json | Dataset, fixed split, manifest checksums |
| E02 | Phase E seed 42 baseline | downloads/aiface-seed42-baseline/aiface_phase_e/ | Global/Local metrics, validation, shortcut |
| E03 | Phase E seed 42 forensic/fusion | downloads/aiface-seed42-fusion/ and downloads/aiface-seed42-spectral/ | Forensic and fusion validation/test/rescue-harm |
| E04 | Phase E seed 43 | downloads/aiface-seed43-baseline/, downloads/aiface-seed43-spectral-full/, downloads/aiface-seed43-fusion/ | Seed-43 baseline, forensic and fusion |
| E05 | Phase E seed 44 | downloads/aiface-seed44-baseline/, downloads/aiface-seed44-spectral-full/, downloads/aiface-seed44-fusion/ | Seed-44 baseline, forensic and fusion |
| E06 | Phase F robustness | outputs/aiface_phase_f_seed42_complete/, outputs/aiface_phase_f_seed43/, outputs/aiface_phase_f_seed44/ | Condition/model and per-generator metrics |
| E07 | Phase F uncertainty | outputs/aiface_phase_f_uncertainty/ | Bootstrap CIs and seed summary |
| E08 | Phase F selection | outputs/aiface_phase_f_model_selection_v2/ | Validation-only model ranking |

Các mã `E01`–`E08` là evidence IDs nội bộ cho việc viết paper. Mỗi nhóm numeric claim trong báo cáo phải giữ lại evidence ID tương ứng khi chuyển sang manuscript.

Các artifact output lớn và dataset không được commit vào Git. Báo cáo này chỉ giữ số liệu và đường dẫn evidence; khi viết paper cần lưu notebook URL, version label và ngày chạy tương ứng cho từng seed.

## 11. Kiểm tra nhất quán đã thực hiện

- 3 seed × 30 condition-model rows = 90 macro rows.
- 3 seed × 60 generator-model-condition rows = 180 per-generator rows.
- Bootstrap có 180 CI rows và 90 paired-difference rows; không có NaN.
- Bootstrap point estimates khớp metric source trong tolerance `1e-8`.
- Manifest checksum, generator mapping và test size 8.000 giống nhau giữa seed 42/43/44.
- Model selection rank 1 là `global_local_forensic`, chọn chỉ từ validation.

**Nguồn dữ liệu được dựng tự động bởi:** `scripts/build_aiface_complete_results.py`.
