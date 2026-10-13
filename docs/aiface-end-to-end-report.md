# AI-Face: Báo cáo nghiên cứu end-to-end

Ngày tổng hợp: 2026-10-13. Trạng thái: bản thảo nghiên cứu nội bộ, cần tác giả kiểm tra trước khi đưa vào manuscript.

Đây là tài liệu đọc chính cho toàn bộ nghiên cứu: từ câu hỏi và giả thuyết đến dataset, audit, phương pháp, kết quả, đối chiếu giả thuyết và giới hạn. [Bảng số liệu đầy đủ](aiface-complete-results.md) là phụ lục định lượng; [báo cáo A–C](aiface-phase-abc-report.md) và [báo cáo E](aiface-phase-e-report.md) giữ chi tiết lịch sử. Không dùng số liệu Olympic AI hoặc smoke test làm kết quả nghiên cứu AI-Face.

## 1. Tóm tắt nghiên cứu

Nghiên cứu kiểm tra liệu đặc trưng patch của DINOv3 và đặc trưng phổ Fourier có bổ sung cho đặc trưng toàn ảnh khi nhận diện khuôn mặt AI từ nguồn chưa xuất hiện trong training. Thí nghiệm dùng AI-Face pilot 32.000 ảnh, fixed generator-disjoint split, ba training seed 42/43/44, ba nhánh độc lập và ba tổ hợp late fusion. Backbone DINOv3 được đóng băng; nhánh forensic thực tế là Radial FFT + MLP. [E01–E05]

Trên clean test gồm STARGAN và StableDiffusion_Inpainting, Local-only đạt macro AUROC trung bình 0.956849, so với 0.935225 của Global-only. Local + Forensic đạt 0.985791; Global + Local + Forensic đạt 0.984904 và macro BAcc 0.952917. Tuy nhiên, với JPEG q70, AUROC của fusion ba nhánh giảm còn 0.736513, trong khi Local-only đạt 0.921856. Kết quả cho thấy tính bổ trợ trên clean held-out test nhưng không bảo đảm tính bổ trợ sau hậu xử lý. [E06–E08]

## 2. Động cơ, research question và nguồn gốc giả thuyết

### 2.1. Research question gốc

Trích [plan.md](plan.md), mục Tuần 1:

> Các đặc trưng cục bộ và dấu vết forensic mức thấp có cung cấp thông tin bổ sung cho đặc trưng toàn ảnh của DINOv3 trong việc generalization sang các nguồn khuôn mặt AI chưa từng xuất hiện trong quá trình huấn luyện hay không?

Động cơ của plan là tránh đánh giá chỉ bằng random image split: classifier có thể học dấu vết của nguồn dữ liệu, resolution hoặc JPEG thay vì tín hiệu chuyển giao được sang nguồn mới. DINOv3 được chọn vì cung cấp global và patch representations từ cùng backbone, cho phép kiểm soát backbone khi so sánh hai cách biểu diễn. Đây là lý do thiết kế, không phải bằng chứng DINOv3 chỉ học ngữ nghĩa hoặc chỉ học tần số thấp. [E01]

### 2.2. Hypothesis được vận hành hóa

Plan trình bày câu hỏi nhỏ và kỳ vọng, không định danh sẵn H1–H5. Các mã dưới đây là cách hệ thống hóa trong báo cáo này, không phải một preregistration được thực hiện trước thí nghiệm. H2 đã được điều chỉnh từ residual trong plan sang spectral forensic trong thực nghiệm. [E01, E05]

| Mã | Giả thuyết/kỳ vọng | Phép so sánh vận hành | Dấu hiệu ủng hộ |
|---|---|---|---|
| H1 | Patch representation giúp unseen-source detection so với global representation | L với G; G+L với G | AUROC/BAcc cao hơn trên held-out generators, xu hướng duy trì qua seed |
| H2 | Low-level forensic cung cấp tín hiệu bổ sung ngoài DINOv3 | L+F với L; G+L+F với G+L; F độc lập | Paired improvement và rescue lớn hơn harm |
| H3 | Thêm thành phần chỉ có giá trị khi fusion cải thiện baseline phù hợp | G+L với L; G+L+F với L+F | Lợi ích đo được, không mặc định ba nhánh tốt nhất |
| H4 | Tín hiệu bổ sung có thể duy trì khi ảnh bị hậu xử lý | Các cặp H2 dưới JPEG/resize/blur/crop | Paired gains vẫn dương; mức giảm so với clean được báo riêng |
| H5 | Fine-tuning nhẹ thay đổi đánh đổi adaptation/generalization so với frozen backbone | Frozen với LoRA trên cùng protocol | Chưa thực hiện trong AI-Face pilot |

H5 là câu hỏi mở, chưa có giả thuyết định hướng hoặc kết quả để chấp nhận/bác bỏ. H1/H3 hiện chủ yếu có bằng chứng mô tả; bootstrap đã lưu chỉ bao phủ ba cặp cụ thể ở mục 8. [E01, E08]

### 2.3. Plan gốc và thí nghiệm thực tế

| Nội dung | Plan gốc | AI-Face đã thực hiện |
|---|---|---|
| Phạm vi | Khuôn mặt do AI tạo hoặc chỉnh sửa | Các source groups GANs/DMs đã chọn; real từ imdb_wiki |
| Forensic | Image/high-frequency residual | Radial FFT + MLP; không có residual CNN trong kết quả này |
| Global/Local | Frozen DINOv3, một pooling cố định | CLS global; mean pooling patch tokens |
| Fusion | Mean/weighted average được đề xuất | Equal-weight probability mean |
| Ablation | Có đề xuất Global + Residual | Không có G+F; có L+F và G+L+F |
| Fine-tuning | LoRA nếu nguồn lực cho phép | Chưa làm |
| Kiểm chứng | Multi-seed và hậu xử lý | Ba seed, clean + bốn corruption, bootstrap |

Việc không có G+F có nghĩa không thể cô lập trực tiếp lợi ích của F khi chỉ bổ sung cho G. Có thể trả lời lợi ích của F khi bổ sung cho L hoặc G+L. [E01, E05–E08]

## 3. Dataset overview và phạm vi nhãn

AI-Face được chọn vì có annotation/path cho phép dựng split theo nguồn. Audit local ghi nhận 1.317.236 dòng train annotation và 329.309 dòng test annotation, tổng 1.646.545. Đây là quy mô annotation gốc, không phải số ảnh dùng để train trong pilot. Các source được chọn xuất hiện trong cả split gốc, nên không dùng split gốc để claim unseen-generator evaluation. [E02]

**Điều chỉnh thuật ngữ:** thư mục GANs/DMs không tự chứng minh mọi ảnh là sinh hoàn toàn từ đầu. Các source đã chọn gồm AttGAN, STARGAN, Palette và StableDiffusion_Inpainting; các nguồn editing/translation/inpainting cần được phân loại đúng theo nguồn gốc subset khi viết paper. Báo cáo này dùng phạm vi “khuôn mặt do AI tạo hoặc chỉnh sửa từ các nguồn đã chọn”, đúng với mục tiêu rộng trong plan. Chưa có audit provenance từng ảnh để bảo đảm “fully synthetic only”. Loại thư mục deepfake-video không đồng nghĩa đã loại mọi dạng manipulation. [E01–E03]

### 3.1. Nguồn và số ảnh

| Nhóm trong dataset | Source ID | Có trong annotation gộp | Lấy cho pilot | Split |
|---|---|---:|---:|---|
| GANs | taming_transformer_VQGAN | 50.000 | 2.000 | Train |
| GANs | stylegan3 | 26.770 | 2.000 | Train |
| GANs | AttGAN | 6.005 | 2.000 | Validation |
| GANs | STARGAN | 5.648 | 2.000 | Test |
| DMs | StableDiffusion1.5 | 18.111 | 2.000 | Train |
| DMs | latent_diffusion | 20.000 | 2.000 | Train |
| DMs | Palette | 6.000 | 2.000 | Validation |
| DMs | StableDiffusion_Inpainting | 20.989 | 2.000 | Test |
| Real | imdb_wiki | Không dùng toàn bộ nguồn | 16.000 | Chia ba split |

Các nhãn family trên là grouping của dataset, không thay thế taxonomy kiến trúc hoặc taxonomy thao tác ảnh. Seed sampling cố định là 42; các run 43/44 không resample dataset. [E02, E03]

| Split | Real (label 0) | Fake (label 1) | Tổng |
|---|---:|---:|---:|
| Train | 8.000 | 8.000 | 16.000 |
| Validation | 4.000 | 4.000 | 8.000 |
| Test | 4.000 | 4.000 | 8.000 |
| Tổng | 16.000 | 16.000 | 32.000 |

Train, validation và test không chia sẻ fake source ID. Đây là một fixed mixed-family holdout, không phải leave-one-generator-out nhiều fold hoặc GAN-to-diffusion cross-family test. Real source imdb_wiki có mặt trong cả ba split, nhưng ảnh có ID riêng. [E03]

## 4. Audit dữ liệu và quality gate

### 4.1. Quy trình A–C

Phase A đọc annotation, derive source từ path, kiểm tra nguồn được chọn và archive. Phase B lấy 2.000 ảnh mỗi fake source cùng 16.000 real, tạo manifest. Phase C kiểm tra file/decode, mode, hash và split; ba duplicate real ban đầu được thay bằng candidate chưa được chọn. Raw archive không bị thay đổi. [E02–E04]

| Kiểm tra cuối | Kết quả |
|---|---:|
| Manifest rows | 32.000 |
| SHA-256 duy nhất | 32.000 |
| Exact duplicate groups | 0 |
| RGB images | 32.000 |
| Issues trong quality_gate.json | 0 |
| Fake source xuất hiện ở nhiều split | 0 |
| Quality gate passed | true |

Hash unique chỉ loại exact byte duplicates; không chứng minh không còn ảnh gần trùng, cùng nội dung sau nén lại hoặc cùng identity. [E03]

### 4.2. Identity hints và source shortcut

| Đại lượng | Số identity hints |
|---|---:|
| Train | 4.218 |
| Validation | 2.669 |
| Test | 2.599 |
| Giao train/validation | 1.383 |
| Giao train/test | 1.384 |
| Giao validation/test | 1.027 |

Đây là hint từ filename, không phải danh tính đã được xác thực. Ví dụ 1.383 là số hint giao giữa tập 4.218 hint train và 2.669 hint validation, không phải số ảnh overlap. Metadata ghi `identity_disjointness_verified=false`; không claim identity-disjointness. [E03]

Real chỉ có imdb_wiki tạo nguy cơ shortcut về camera, JPEG, resolution và cách crop. Cân bằng số lớp không loại được shortcut này. Kiểm tra shortcut định lượng ở mục 10 bổ sung bằng chứng nhưng không đủ xác nhận nguyên nhân của hiệu năng model. [E03, E09]

## 5. Đóng gói, smoke test và luồng end-to-end

Dataset runtime dùng `nguyentrann0703/aiface-pilot-32k-single-bin`. Notebook/script phải giải quyết đúng manifest cố định và tắt random fallback. Giai đoạn đóng gói từng gặp lỗi mount dataset; đây là lịch sử vận hành, không phải một condition đánh giá model. [E05, E06]

```text
AI-Face annotation + archive
  -> audit nguồn/path -> subset 32K -> repair duplicate -> quality gate
  -> fixed manifest train 16K / validation 8K / test 8K
  -> đóng gói dataset -> smoke test
  -> ba nhánh độc lập, mỗi nhánh train với seed 42/43/44
       RGB -> DINOv3 frozen -> CLS -----------------> G classifier
                          -> mean spatial patches -> L classifier
       RGB -> Hann FFT -> radial bins -> normalize -> F classifier
  -> chọn checkpoint và threshold bằng validation
  -> ghép probability theo image_id: G+L, L+F, G+L+F
  -> clean held-out test + rescue/harm
  -> JPEG / resize / blur / crop, checkpoint và threshold cố định
  -> bootstrap paired CI + mean/SD seed + validation ranking
  -> đối chiếu H1–H5 và giới hạn
```

Smoke DINOv3 local ghi input `2×3×224×224`, global `2×768`, patches `2×196×768`, backbone frozen, không NaN/Inf, chạy MPS. Smoke kiểm tra pipeline hoạt động; không được đưa metric smoke vào các bảng research. [E04]

## 6. Phương pháp thực tế

### 6.1. Global branch (G)

Backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` đóng băng. Official image processor được cấu hình kích thước 224×224. Lấy CLS token 768 chiều và train MLP `768→256→2`, GELU, dropout 0. Feature có thể cache để train probe mà không cập nhật backbone. Global là tên cách lấy token; không có phép đo trong pilot chứng minh feature chỉ chứa semantic information. [E05]

### 6.2. Local branch (L)

Cùng backbone và processor với G. Bỏ CLS và register tokens, lấy 196 spatial patch tokens, mỗi token 768 chiều; mean pooling thành vector 768 chiều rồi train MLP riêng cùng cấu hình G. Local ở đây không phải một crop detector, không chọn top-k patch, không dùng patch supervision hoặc attention localization. Mean pooling làm mất thông tin vị trí tường minh; thành công của L không tự chứng minh model tìm được vùng giả mạo. [E05]

### 6.3. Spectral forensic branch (F)

Ảnh được EXIF-transpose, chuyển RGB, resize square bilinear antialias 224×224, rescale [0,1]. Không dùng ImageNet normalization cho FFT. Mỗi kênh nhân Hann đối xứng, FFT2 với norm backward, fftshift, lấy `log1p(abs(spectrum))`, trung bình theo 64 radial bins từ tâm đến góc. Ba kênh ghép thành vector 192 chiều. Feature mean/std chỉ fit trên train, sau đó giữ cố định cho validation/test. [E05]

Classifier là MLP `192→128→2`, GELU, dropout 0, có 24.962 tham số trainable. Radial average giảm chiều nhưng bỏ thông tin hướng/vị trí và không giữ phase; đây là lựa chọn cần kiểm chứng, không bảo đảm tránh overfit hoặc đạt generator-invariant features. Không có DCT, SRM, residual CNN hoặc AIDE trong nhánh được đánh giá. [E05]

### 6.4. Training và selection

Config Global/Local dùng 20 epoch, batch 32, AdamW, learning rate 0.001, weight decay 0.0001, cross-entropy. Config spectral dùng 20 epoch, MLP batch 32, learning rate 0.001, weight decay 0.0001. Ba nhánh có ba training seed; mỗi seed tạo sáu cấu hình đánh giá nhưng chỉ có ba branch cần train. [E05]

Checkpoint branch được chọn bằng validation AUROC rồi BAcc tại threshold 0.5. Sau đó threshold được chọn theo validation BAcc, tie-break F1 rồi khoảng cách tới 0.5. Mỗi fusion có threshold validation riêng. Không chọn threshold riêng trên test hoặc từng corruption. [E05]

### 6.5. Late fusion và ablation

Gọi `p_G`, `p_L`, `p_F` là probability lớp fake. Các công thức thực tế:

```text
p_GL  = (p_G + p_L) / 2
p_LF  = (p_L + p_F) / 2
p_GLF = (p_G + p_L + p_F) / 3
```

Ghép prediction theo image_id; seed, label, split và source phải tương ứng. Không lấy average hai lần kiểu `(p_GL+p_F)/2`, vì như vậy trọng số F sẽ khác. Không có feature concatenation hoặc train classifier fusion. Threshold rất khác nhau giữa các nhánh nên equal-probability fusion không đồng nghĩa probabilities đã calibrated. [E05, E06]

## 7. Định nghĩa metric và đơn vị phân tích

Primary metrics là AUROC và Balanced Accuracy (BAcc); Accuracy/F1 bổ sung. `BAcc=(TPR+TNR)/2`, predicted fake nếu probability ≥ threshold validation. AUROC dùng ranking, không phụ thuộc threshold. [E05]

Per-generator metric so 2.000 fake của một source với cùng 4.000 real test, N=6.000. Macro là trung bình metric trên hai source. Overall dùng đúng 8.000 ảnh duy nhất. Do số fake mỗi source bằng nhau và real pool dùng chung, overall AUROC/BAcc bằng macro AUROC/BAcc trong thiết kế này; Accuracy/F1 per-generator có class ratio khác overall nên không được trộn hai loại. [E03, E06]

Mean và sample SD (`ddof=1`) qua ba seed mô tả training variation. Seed không phải ba dataset hoặc ba generator split độc lập. Chênh lệch 0.01 trên thang [0,1] tương ứng 1 điểm phần trăm khi biểu diễn metric ở thang phần trăm. [E08]

## 8. Clean test, bootstrap và model selection

### 8.1. Clean mean ± sample SD, ba seed

| Model | Macro AUROC | Macro BAcc |
|---|---:|---:|
| G | 0.935225 ± 0.006273 | 0.847042 ± 0.014222 |
| L | 0.956849 ± 0.003097 | 0.877542 ± 0.008174 |
| G+L | 0.954747 ± 0.003585 | 0.872083 ± 0.004693 |
| F | 0.969722 ± 0.009722 | 0.918958 ± 0.017400 |
| L+F | 0.985791 ± 0.003401 | 0.950958 ± 0.013281 |
| G+L+F | 0.984904 ± 0.002955 | 0.952917 ± 0.010876 |

Nguồn: `seed_summary_mean_sd.csv`, condition clean [E08]. L tốt hơn G ở cả ba seed. G+L tốt hơn G nhưng không tốt hơn L về mean AUROC/BAcc. L+F có mean AUROC cao nhất, G+L+F có mean BAcc cao nhất; không có cấu hình thắng mọi tiêu chí.

### 8.2. Paired uncertainty cho tính bổ sung

Bootstrap có 2.000 replicates, seed 2026, percentile CI 95%, stratified theo source × label; dùng chung sample indices giữa models của từng comparison. Chỉ resample prediction đã lưu, không inference hoặc training lại. Real pool phải được dùng chung khi tính hai generator metrics. [E08]

| Seed | Δ AUROC L+F − L | 95% CI | Δ AUROC G+L+F − G+L | 95% CI |
|---|---:|---|---:|---|
| 42 | 0.022205 | [0.018511, 0.026040] | 0.022929 | [0.019436, 0.026488] |
| 43 | 0.030009 | [0.026737, 0.033451] | 0.034140 | [0.030948, 0.037436] |
| 44 | 0.034613 | [0.030874, 0.038291] | 0.033401 | [0.029919, 0.036846] |

Các khoảng không chứa 0 ủng hộ clean paired gain trong test set cố định. Không chuyển thành claim “có ý nghĩa trên mọi unseen generator”. Ba comparison đã bootstrap là L+F−L, G+L+F−G+L và F−L, với hai metrics trên năm conditions; không có CI trực tiếp cho L−G hoặc G+L+F−L+F trong artifact hiện tại. CI là theo từng comparison, không có hiệu chỉnh multiple comparisons. [E08]

### 8.3. Selection bằng validation

| Rank | Model | Validation macro AUROC mean | Validation macro BAcc mean |
|---|---|---:|---:|
| 1 | G+L+F | 0.986532 | 0.953667 |
| 2 | L+F | 0.985800 | 0.950792 |
| 3 | F | 0.972074 | 0.919083 |
| 4 | G+L | 0.960834 | 0.893708 |
| 5 | G | 0.954901 | 0.884750 |
| 6 | L | 0.954821 | 0.883042 |

Nguồn: `model_selection_v2` [E08]. Rule: validation macro AUROC mean, BAcc mean, AUROC SD tăng dần, rồi worst-generator criterion. Model recipe được chọn là G+L+F; đây không phải chọn seed tốt nhất hoặc ensemble cả ba seed.

**Giới hạn chronology:** JSON xác nhận thuật toán selection không dùng test metric. Tuy nhiên nghiên cứu đã xem các test results trong quá trình phát triển; không có bằng chứng ở đây để khẳng định toàn bộ model-selection decision được preregister và khóa trước lần đầu xem test. Vì vậy báo cáo pilot mang tính exploratory; không mô tả là một prospective blind final test. [E01, E06, E08]

## 9. Robustness: kiểm tra khả năng giữ lại lợi ích

Mỗi condition khởi đầu từ ảnh RGB gốc, không nối corruption. JPEG encode quality 70 và subsampling 2; resize xuống 112×112 rồi lên 224×224 bằng bilinear; blur Gaussian radius 1.0; crop hình vuông ở tâm với cạnh `int(0.8*min(width,height))` rồi resize 224×224. Crop 80% là 80% cạnh ngắn, không phải 80% diện tích. Sau đó từng nhánh áp dụng preprocessing của mình. [E07]

| Condition | G AUROC | L AUROC | G+L AUROC | F AUROC | L+F AUROC | G+L+F AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.935225 | 0.956849 | 0.954747 | 0.969722 | 0.985791 | 0.984904 |
| JPEG q70 | 0.884857 | 0.921856 | 0.916203 | 0.460184 | 0.701472 | 0.736513 |
| Resize 112→224 | 0.906463 | 0.936192 | 0.930469 | 0.702358 | 0.892205 | 0.902062 |
| Blur r1 | 0.933965 | 0.951653 | 0.950635 | 0.682186 | 0.865187 | 0.873441 |
| Center crop 80% | 0.916453 | 0.928411 | 0.928603 | 0.837569 | 0.919182 | 0.937686 |

| Condition | G BAcc | L BAcc | G+L BAcc | F BAcc | L+F BAcc | G+L+F BAcc |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.847042 | 0.877542 | 0.872083 | 0.918958 | 0.950958 | 0.952917 |
| JPEG q70 | 0.794500 | 0.831667 | 0.827792 | 0.562042 | 0.721708 | 0.741208 |
| Resize 112→224 | 0.813958 | 0.825292 | 0.830833 | 0.589208 | 0.820833 | 0.828750 |
| Blur r1 | 0.832833 | 0.869917 | 0.863833 | 0.525250 | 0.560792 | 0.572750 |
| Center crop 80% | 0.805917 | 0.763500 | 0.782542 | 0.705875 | 0.862667 | 0.861792 |

Các giá trị là mean qua ba seed, nguồn [E08]. SD, từng seed, delta, per-generator và CIs có trong phụ lục số liệu/CSV. Năm conditions gồm một clean và bốn corruption, không phải năm corruption.

F rất nhạy với JPEG/blur; fusion có F cũng bị giảm so với DINO-only baselines dưới các biến đổi này. Với crop, G+L+F vẫn có mean AUROC tốt hơn G+L, còn L+F có mean AUROC thấp hơn L nhưng BAcc cao hơn. Đây là ví dụ ranking và operating threshold có thể dẫn đến kết luận khác nhau. H4 không được ủng hộ như một khẳng định chung cho mọi corruption. [E07, E08]

## 10. Error analysis và shortcut diagnostic

Rescue = baseline sai, fusion đúng; harm = baseline đúng, fusion sai. Các model dùng threshold validation riêng. L+F được so với L; G+L+F được so với G+L. [E06]

| Seed | Fusion | Rescue | Harm | Net gain trên 8.000 ảnh |
|---|---|---:|---:|---:|
| 42 | L+F | 798 | 382 | 416 |
| 43 | L+F | 815 | 194 | 621 |
| 44 | L+F | 936 | 211 | 725 |
| 42 | G+L+F | 806 | 303 | 503 |
| 43 | G+L+F | 839 | 126 | 713 |
| 44 | G+L+F | 902 | 178 | 724 |

Đây là bằng chứng hành vi bổ trợ trên prediction, không chứng minh hai representation trực giao hoặc cho biết artifact vật lý nào được phát hiện. Muốn nói về vùng da, biên hòa trộn hay sensor noise cụ thể cần phân tích định tính/attribution riêng. [E06]

Shortcut diagnostic seed 42 fit Logistic Regression trên train với file size hoặc simple features: luminance mean/std, saturation, gray fraction, edge mean và file size. [E09]

| Shortcut | Train AUROC | Validation AUROC | Validation BAcc |
|---|---:|---:|---:|
| File size only | 0.780116 | 0.537581 | 0.747625 |
| Simple features | 0.860288 | 0.600553 | 0.638750 |

Các kết quả mô tả cho thấy feature đơn giản cũng có tín hiệu, nhưng chưa có CI cho diagnostic này và không đo phần trăm hiệu năng của neural model do shortcut gây ra. Không kết luận shortcut đã được loại bỏ chỉ vì AUROC logistic thấp hơn model chính. [E09]

## 11. Trả lời từng hypothesis và RQ

| Hypothesis | Kết luận theo bằng chứng hiện có | Giới hạn |
|---|---|---|
| H1: local có ích so với global | Được ủng hộ mô tả trên clean ở ba seed | Chưa có paired CI L−G; không suy ra localization |
| H2: forensic bổ sung DINOv3 | Được ủng hộ cho L+F−L và G+L+F−G+L trên clean, CI dương | Không có G+F; chưa khẳng định cơ chế của tín hiệu |
| H3: mọi thành phần thêm vào đều hữu ích | Không có sự cải thiện đồng nhất | G+L kém L về mean clean metrics; G+L+F thấp hơn L+F về AUROC nhưng cao hơn BAcc |
| H4: tính bổ trợ bền qua degradation | Không được ủng hộ như khẳng định chung | JPEG/blur làm F và fusion suy giảm mạnh; crop cho kết quả tùy metric |
| H5: LoRA so với frozen | Chưa trả lời | Không có thí nghiệm LoRA AI-Face |

**Trả lời RQ:** Trên fixed AI-Face pilot, patch-mean DINOv3 là baseline tốt hơn CLS-only, và spectral forensic bổ sung lợi ích clean khi ghép với L hoặc G+L. Tính bổ trợ có điều kiện theo input processing; nó không chuyển thành một detector luôn tốt hơn dưới JPEG, resize và blur. Kết quả đánh giá prediction hỗ trợ tính bổ sung, không chứng minh independence của feature hoặc fingerprint invariant theo generator. [E06–E09]

## 12. Limitations và phần chưa hoàn tất để viết paper

| Vấn đề | Tình trạng | Hệ quả cho manuscript |
|---|---|---|
| Hai unseen sources, một fixed split | Đã biết | Không khái quát mọi generator; không gọi LOGO benchmark |
| Sinh hoàn toàn so với chỉnh sửa | Taxonomy ảnh/source chưa audit đầy đủ | Dùng scope AI-generated/edited faces; rà tên task trước khi nộp |
| Identity overlap hints | Có overlap; không có ID chính thức | Không claim identity-disjoint |
| Real-source confounding | Chỉ imdb_wiki | Cần real-source-controlled benchmark để cô lập nguyên nhân mạnh hơn |
| Near-duplicate/semantic overlap | Exact SHA audit không bao phủ | Không claim đã loại tất cả leakage |
| Pretraining exposure DINOv3 | Chưa audit | Unseen nghĩa là không nằm trong supervised branch training, không bảo đảm unseen với pretraining |
| Chronology/test reuse | Đã xem test trong quá trình phát triển | Pilot exploratory; selection code validation-only không đủ chứng minh blind evaluation |
| Hardware | Phase F 42/43 dùng MPS, 44 dùng CUDA | SD có thể chứa cả khác biệt số học/backend, không chỉ initialization |
| Bootstrap | Image-level, fixed split/checkpoints | Không phản ánh identity-cluster hoặc generator-population uncertainty |
| Calibration/weighted fusion | Chưa làm | Không diễn giải probability là confidence đã calibrated |
| G+F, LoRA, nhiều split, external test | Chưa làm | Không báo như kết quả đã có; không bắt buộc mở rộng để mô tả pilot hiện tại |
| Related work/citation | Plan nhắc 9 paper, chưa xác minh lại bibliography trong báo cáo này | Cần nguồn chính thức cho AI-Face, DINOv3 và frequency detection khi viết related work |
| Ethics/license/availability | Chưa xác nhận đủ để viết declarations | Kiểm tra điều kiện dùng face data, quyền tái phân phối và đóng góp tác giả |
| Qualitative figures và compute cost | Chưa có tổng hợp đầy đủ trong báo cáo này | Chỉ thêm montage/thời gian/VRAM khi có artifact hỗ trợ |

Các mục chưa làm không làm mất các metric đã có. Chúng xác định ranh giới kết luận và phần cần bổ sung cho bản paper, không phải lý do tự thay đổi dataset hoặc retrain trong lần tổng hợp này.

## 13. Tái lập và evidence map

Notebook 09 audit dữ liệu; 10 train G/L và G+L; 11 train spectral F; 12 ghép L+F và G+L+F; 13 phục vụ robustness Phase F. Bootstrap đọc predictions, không cần chạy lại backbone. Config và metadata đi kèm artifact là nguồn ưu tiên khi khác config mặc định hiện tại. [E04–E08]

| Evidence ID | Nguồn local | Vai trò |
|---|---|---|
| E01 | [plan.md](plan.md), mục 1–6 | RQ và dự định ban đầu; H1–H5 là cách diễn đạt vận hành trong báo cáo |
| E02 | `data/AI_Face/audit_outputs/audit_summary.json` | Quy mô annotation, source counts, split gốc |
| E03 | `data/AI_Face/aiface_pilot_32k/subset_metadata.json`, `audit/quality_gate.json` | Pilot counts, mapping, hash, identity hints |
| E04 | [aiface-phase-abc-report.md](aiface-phase-abc-report.md), `audit/repair_report.json`, `outputs/aiface_smoke_test/dinov3_smoke.json` | Repair và smoke |
| E05 | `configs/aiface_phase_e.yaml`, `configs/spectral.yaml`, `src/models/dinov3.py`, `src/models/spectral.py`, `scripts/run_aiface_phase_e.py`, resolved configs trong downloads | Kiến trúc/training thực tế |
| E06 | `downloads/aiface-seed{42,43,44}-fusion/`, `outputs/aiface_phase_f_seed42_complete/`, `outputs/aiface_phase_f_seed43/`, `outputs/aiface_phase_f_seed44/` | Predictions, clean và rescue/harm |
| E07 | `src/evaluation/aiface_phase_f.py`, `corruption_config.yaml` và metric CSV trong các output Phase F | Corruption và robustness |
| E08 | `outputs/aiface_phase_f_uncertainty/{seed_summary_mean_sd,ci_by_model_condition,paired_delta_ci}.csv`, `outputs/aiface_phase_f_model_selection_v2/` | Mean/SD, CI và validation selection |
| E09 | `downloads/aiface-seed42-baseline/aiface_phase_e/shortcut/metrics.csv` | Diagnostic shortcut |

Các evidence IDs thuộc phạm vi tài liệu này. Đây là đối chiếu local bằng công cụ; không thay cho xác nhận nguồn và quyết định khoa học của tác giả. Không tự tạo citation hoặc điền notebook version còn thiếu. `docs/plan.md` đang là file chưa được track ở thời điểm tổng hợp; cần giữ file này cùng báo cáo khi version-control để liên kết nguồn RQ không bị mất.

### 13.1. Cách dùng để viết manuscript

Introduction lấy mục 2 và related work đã xác minh; Data lấy mục 3–4; Methods lấy mục 5–7; Results lấy mục 8–10 và phụ lục số liệu; Discussion lấy mục 11–12. Trước khi nộp, kiểm tra taxonomy source, chronology, nguồn citation, figure captions và declarations. Không chuyển các hypothesis suy ra hồi cứu thành hypothesis preregistered.

### 13.2. Quy ước khi số liệu giữa report lịch sử khác nhau

Các bảng mean/SD và robustness trong báo cáo này dùng Phase F uncertainty canonical. Một số con số clean và threshold có thể khác chữ số cuối so với report Phase E do artifact/reconstruction/backend hoặc precision; phải truy về đúng run trước khi trộn bảng. Không coi mọi sai khác là rounding nếu chưa kiểm tra. Giữ full-precision CSV cho tái lập, dùng sáu chữ số cho bảng hiển thị; threshold rất nhỏ cần dùng scientific notation khi chép sang paper.
