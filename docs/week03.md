# Implementation plan Tuần 3: Residual và tính bổ sung giữa các nhánh

Ngày lập: 2026-10-01. Trạng thái: core Week 3 đã triển khai và có kết quả; residual multi-seed 43/44 và unseen-source evaluation còn thiếu.

Tài liệu này dùng để giao việc triển khai và chạy notebook trên Kaggle. Các tên file, config và artifact ở phần công việc là đề xuất cần tạo sau; không phải các thành phần đã tồn tại. Giữ nguyên plan/outline cũ để truy vết.

## 1. Những quyết định đã chốt cho tuần này

1. Tiếp tục với Who Is AI hiện tại: 2.000 ảnh có nhãn, split 1.600 train / 400 validation của Tuần 2. Thiếu unseen dataset không chặn việc xây pipeline.
2. Kết quả tuần này là **in-domain validation phục vụ phát triển**, chưa trả lời được giả thuyết generalization sang generator chưa thấy.
3. Tập trung một representation mới: image residual bằng bộ lọc thông cao cố định, kết hợp một CNN nhỏ học từ đầu. Không mở đồng thời FFT, DCT, SRM và nhiều backbone.
4. Đánh giá Residual-only trước; sau đó so sánh late fusion với cả Global-only và Local-only. Local-only là mốc mạnh nhất hiện có.
5. Sanitized dataset là control tùy chọn khi có lý do cụ thể, không là điều kiện hoàn thành Week 3. EXIF hiện không có trong các ảnh đã kiểm tra; đó là quan sát, không phải bằng chứng rằng code audit từng xóa EXIF.
6. Huấn luyện và inference ảnh dùng Kaggle GPU. Mac dùng để viết code, chạy unit test CPU nhỏ, đọc log và tổng hợp báo cáo.
7. LoRA/anchor-LoRA dành cho Week 4 sau khi residual/fusion ổn định. Đây là lựa chọn thu hẹp phạm vi so với mục fine-tuning tùy điều kiện trong plan gốc.
8. Chuẩn bị giao diện nhận manifest mới để đánh giá external dataset về sau. Chưa tìm được source của train thì external evaluation chỉ được gọi là cross-dataset/OOD; muốn gọi generator-unseen phải chứng minh generator đó không xuất hiện trong detector training.

Các quyết định này cập nhật thứ tự ưu tiên làm việc: những đoạn trong tài liệu cũ yêu cầu có unseen split trước mới triển khai residual không còn là cổng chặn của Week 3. Chúng vẫn đúng về điều kiện để kết luận generalization.

## 2. Điểm xuất phát đã đối chiếu trong repo

### 2.1. Dữ liệu và kết quả Tuần 2

Nguồn: [week02.md](week02.md), [progress.md](progress.md), output đã lưu của notebook `03_local_experiment.ipynb` và `04_error_robustness.ipynb`. Đây là đối chiếu artifact/notebook, không phải lần train mới.

| Thành phần | Hiện trạng | Cách dùng trong Week 3 |
| --- | --- | --- |
| Manifest gốc | `label`, `class_name`, `path`; 1.000 Real và 1.000 Fake | Giữ nhãn `0=Real`, `1=Fake`; không suy ra nhãn từ tên file |
| Source/generator | Chưa có mapping đáng tin cậy | Giữ `source` rỗng, `source_status=unavailable` |
| Public/private test | Mỗi tập 200 ảnh, chưa có nhãn local | Không dùng để chọn model hoặc tính metric |
| Split | Stratified theo label, seed 42; train 800/800, val 200/200 | Dùng đúng danh sách ID đã lưu trong Phase 4 |
| DINOv3 | Frozen `facebook/dinov3-vitb16-pretrain-lvd1689m` | Giữ model revision từ cache, không dùng revision mới mặc định |
| Model revision đã lưu | `5931719e67bbdb9737e363e781fb0c67687896bc` | Kiểm tra đối chiếu khi nạp weights |
| Feature | CLS `(2000,768)`, patch `(2000,196,768)`, float16 | Cast float32 rồi mean-pool patch; không đưa CLS/register vào Local |
| Local hiện tại | Mean của toàn bộ spatial patch tokens | Không gọi là top-k, attention, face-part crop hay concept branch |
| Artifact | Tạo trên Kaggle; `outputs/` local chưa chứa các run đó | Bắt buộc attach/download đúng artifact trước khi chạy tiếp |

Mốc Phase 5/6 trên 400 ảnh validation:

| Representation | AUROC | Balanced Accuracy tại threshold 0.5 |
| --- | ---: | ---: |
| Global-only + MLP | 0.930025 | 0.8400 |
| Local-only + MLP | 0.952550 | 0.8825 |
| Global+Local concat + MLP | 0.945550 | 0.8725 |

Global+MLP của Phase 3 có AUROC 0.932275 / BAcc 0.8700, thuộc run khác. Không thay nó vào bảng Phase 5 để ghép thành một comparison được cho là cùng điều kiện. Lưu riêng cột `run_origin` cho kết quả lịch sử và kết quả chạy mới.

### 2.2. Thành phần có thể dùng lại và chỗ còn thiếu

| File/module hiện có | Cách sử dụng | Lưu ý khi triển khai |
| --- | --- | --- |
| `scripts/create_splits.py` | Tái tạo split khi thiếu manifest artifact | Đối chiếu ID với cache; `source` đang được điền rỗng, script chưa hỗ trợ source-disjoint |
| `src/models/dinov3.py` | Trích CLS và patch của backbone frozen | Backbone luôn `eval()`, không tích gradient |
| `src/input_data/feature_dataset.py` | Train lại probe trên cache | Giữ mapping index/ID/label |
| `src/models/classifier.py` | MLP Global/Local | Hidden 256, GELU, dropout 0, hai logits |
| `src/training/feature_trainer.py` | Replay/train probe theo protocol cũ | Không sửa trainer lịch sử âm thầm trong lúc đối chiếu |
| `src/models/local.py::high_pass_residual` | Tham khảo phép trừ low-pass | Đang dùng `avg_pool2d(..., padding=1)` zero-padding; chưa có residual detector hoàn chỉnh |
| `src/evaluation/phase6.py` | Artifact validation, corruption, metric, error grouping | Một số hàm khóa 2.000 ảnh/400 val/ba nhánh; không dùng nguyên cho external dataset |
| `scripts/train_local.py`, `scripts/evaluate.py` | Chưa là entry point hoàn chỉnh | Còn `NotImplementedError`; plan không giả định chúng chạy được |
| `notebooks/00` đến `04` | Mốc kiểm chứng và ví dụ notebook độc lập | Không phụ thuộc biến còn trong kernel cũ |

## 3. Câu hỏi tuần này và căn cứ lựa chọn

### 3.1. Những câu hỏi thực sự có thể trả lời

- **W3-Q1:** Residual-only có học được tín hiệu Real/Fake trên split hiện tại không? So với cùng CNN nhận RGB thì ra sao?
- **W3-Q2:** Residual sửa được bao nhiêu lỗi của Global và Local; đồng thời làm sai thêm bao nhiêu mẫu khi fusion?
- **W3-Q3:** Late fusion cố định có cải thiện AUROC/BAcc so với nhánh riêng mạnh nhất hay chỉ so với Global yếu hơn?
- **W3-Q4:** Giá trị bổ sung còn xuất hiện dưới JPEG q70, resize 112→224 và blur r=1 không?

Các câu hỏi này khảo sát RQ2/RQ3 trong outline ở phạm vi in-domain. H1/H2/H3 về unseen generalization vẫn chờ dữ liệu đánh giá thích hợp. Không viết lại kết quả đã thấy ở Week 2 thành giả thuyết được đặt trước.

### 3.2. Paper evidence → quyết định triển khai

| Paper ID | Evidence dùng làm motivation | Quyết định Week 3 | Điều chưa được paper chứng minh cho project |
| --- | --- | --- | --- |
| [P01 AIDE](../research/cards/P01_AIDE.md) | Kết hợp global semantic và low-level noise/frequency; Chameleon bộc lộ giới hạn benchmark | Tách G/L/R, kiểm tra fusion và robustness | Box-filter residual + TinyCNN ở đây không phải tái hiện AIDE/SRM |
| [P02 UnivFD](../research/cards/P02_UniversalFakeDetectors.md) | Frozen pretrained features là baseline có giá trị | Giữ Frozen DINOv3 và probe controls | CLIP và DINOv3 không có kết quả mặc nhiên tương đương |
| [P03 SPAI](../research/cards/P03_SPAI.md) | Phổ/tần số và post-processing ảnh hưởng detection | Theo dõi residual dưới corruption | Không triển khai masked spectral learning hoặc one-class SPAI |
| [P04 B-Free](../research/cards/P04_BFree.md) | Data/content/format bias là confounder | Thêm shortcut diagnostic rẻ, giữ re-encode control tùy chọn | Re-encode không tương đương semantic alignment của B-Free |
| [P07 OmniAID](../research/cards/P07_OmniAID.md) | Phân tách semantic và artifact experts | Late fusion xác suất để thử complementarity | Không đồng nhất mean-pool DINO với semantic expert hoặc residual với artifact thuần túy |
| [P08 ForensicConcept](../research/cards/P08_ForensicConcept.md) | Local evidence cần kiểm tra khả năng chuyển giao | Phân tích lỗi và giới hạn diễn giải Local | Mean-pool patch hiện tại không tái hiện attribution/codebook/CleanDIFT |
| [P09 ARA](../research/cards/P09_ARA.md) | Adaptation có thể làm giảm transfer | Giữ frozen control, chuyển adaptation sang tuần sau | ARA không mặc định tốt hơn frozen trên DINOv3-B; xem Table 11/Appendix A.4 |

P05 VIB-Net và P06 BLADES vẫn thuộc literature base; không thêm bottleneck hay EXIF pretraining vào phạm vi implementation tuần này. Các lựa chọn CNN, kernel, epoch và seed bên dưới là thiết kế của nhóm để kiểm tra, không phải hyperparameter đã được các paper xác nhận trên Who Is AI.

## 4. Protocol chung và ranh giới đánh giá

### 4.1. Split, đơn vị mẫu và mức độ độc lập

- `split_seed=42` bất biến. `train_seed` thay đổi riêng khi lặp huấn luyện; tuyệt đối không tạo lại train/val mỗi lần đổi seed model.
- Ưu tiên `train.csv`/`val.csv` trong cache Phase 4. Đường dẫn có thể remap theo mount Kaggle mới, nhưng ID, label, split và original relative path phải giữ nguyên.
- Kiểm tra không trùng ID/path; kiểm tra hash nội dung nếu chưa có bằng chứng tương thích với bản dữ liệu được attach. Không có duplicate byte không loại trừ ảnh gần trùng hay cùng danh tính.
- Nếu phát hiện leakage/near-duplicate đáng tin cậy, dừng comparison chính để ghi protocol revision. Không âm thầm đổi split rồi so score với bảng cũ.
- Đơn vị đánh giá là ảnh. Các corruption của cùng ảnh, patch của cùng ảnh và nhiều seed model không phải những ảnh độc lập mới.
- Không tạo một “test mới” từ 400 val đã được dùng chọn model rồi gọi nó là untouched test.

### 4.2. Chọn checkpoint và metric

| Quy tắc | Giá trị đề xuất |
| --- | --- |
| Score dự đoán | `prob_fake`, cột softmax ứng với label 1 |
| Primary comparison | Validation AUROC; luôn báo BAcc kèm theo |
| Quyết định nhãn | `prob_fake >= 0.5` |
| Chọn checkpoint | AUROC clean val cao nhất; hòa thì BAcc cao hơn; vẫn hòa giữ epoch sớm hơn |
| Thời lượng train | 20 epoch cho run chính; không dùng corruption để chọn epoch |
| Model selection | Dùng clean validation, ghi rõ đây là validation đã dùng lựa chọn |
| Fusion chính | Trọng số cố định từ plan; không tìm trọng số trên 400 val |
| Test/corruption inference | Checkpoint và threshold giữ nguyên |

Kết quả trên validation này mang tính exploratory và có thể lạc quan vì vừa chọn checkpoint vừa báo cáo. Nhiều seed hoặc bootstrap không sửa được hạn chế đó. Chưa đặt mục tiêu kiểu “phải đạt 0.97 AUROC”.

### 4.3. Seed và baseline

- Mục tiêu đầy đủ: train seeds **42, 43, 44**, cùng split seed 42, cho G, L, R và RGB-control.
- G/L: dùng cache Phase 4, giữ MLP/training recipe Phase 5. Chạy lại seed 42 để kiểm tra notebook mới; sai khác phải giải thích trước khi trộn vào tổng hợp.
- G+L concat Phase 5: replay seed 42 làm mốc lịch sử. Không đưa nó vào bảng mean±SD ba seed nếu chưa chạy đủ ba seed.
- Nếu quota hạn chế: hoàn thành toàn bộ comparison seed 42 trước, rồi bổ sung seed 43/44. Gắn trạng thái `single_seed` khi chưa chạy đủ.
- Ghi GPU, precision, package versions và determinism settings; cùng seed không bảo đảm bitwise giống giữa mọi GPU/PyTorch.

## 5. Thiết kế Residual-only cần triển khai

### 5.1. Chọn representation chính

Tên version đề xuất: `residual_box3_reflect_rgb224_v1`.

Chuỗi xử lý cho ảnh clean:

`file ảnh → EXIF orientation → RGB → resize/crop như processor Week 2 → float32 [0,1] → residual → TinyCNN → 2 logits`.

- Kích thước chính 224×224 để cùng spatial preprocessing với DINOv3. Giữ ảnh ba kênh kể cả khi nội dung gần grayscale.
- Dùng bản cấu hình processor đã lưu; tạo processor riêng cho residual/RGB-control với normalize tắt, giữ resize/crop/rescale. Không mutate processor DINO đang dùng chung.
- Kiểm tra tensor trước normalize của nhánh mới tương ứng tensor DINO sau hoàn nguyên mean/std, trong tolerance số học đã ghi nhận.
- Phép residual theo từng kênh: `R = X - box_blur_3x3(X)`.
- `box_blur`: reflect-pad một pixel rồi average pool kernel 3, stride 1, padding 0. Output cùng shape input.
- `residual_gain=1.0`. Không min-max từng ảnh, không lấy trị tuyệt đối, không clip âm về 0, không chuyển residual sang JPEG/uint8.
- Phép trừ/lọc thực hiện float32 ngoài autocast. AMP chỉ dùng cho CNN.
- Residual có cả giá trị âm/dương. Với input [0,1], cần hữu hạn và nằm xấp xỉ [-1,1]. Hình visualization được scale riêng; tuyệt đối không đưa bản visualization vào train.

**Vì sao không dùng nguyên helper hiện tại?** Zero-padding trong `src/models/local.py` khiến một ảnh hằng số vẫn sinh residual ở viền. Filter mới phải có test ảnh hằng số để tránh CNN học viền nhân tạo.

Residual sau resize chỉ kiểm tra một cách biểu diễn cụ thể. Resize có thể làm mất low-level traces trước khi lọc; kết quả âm không chứng minh mọi residual đều vô ích. Native-resolution residual là extension có kiểm soát ở mục 13, không tự thay đổi giữa chừng.

### 5.2. CNN nhỏ và RGB-control

Tên kiến trúc đề xuất: `TinyResidualCNN_v1`, không pretrained.

| Bước | Cấu hình | Shape dự kiến |
| --- | --- | --- |
| Input | Residual RGB | B×3×224×224 |
| Block 1 | Conv 3×3, 3→32, stride 1, padding 1, bias false; GroupNorm 8 groups; GELU | B×32×224×224 |
| Block 2 | Conv 3×3, 32→64, stride 2, padding 1, bias false; GroupNorm 8 groups; GELU | B×64×112×112 |
| Block 3 | Conv 3×3, 64→128, stride 2, padding 1, bias false; GroupNorm 8 groups; GELU | B×128×56×56 |
| Pool | Adaptive average pooling 1×1, flatten | B×128 |
| Classifier | Linear 128→2, bias true | B×2 |

GroupNorm giảm phụ thuộc batch statistics khi batch nhỏ. Không đưa DINOv3 vào nhánh R và không feed residual đã chuẩn hóa tùy tiện vào pretrained DINOv3. CNN này là baseline rẻ để kiểm tra residual, chưa phải detector SOTA.

**RGB-control:** cùng CNN, khởi tạo cùng seed, optimizer, data order và preprocessing; chỉ bỏ bước residual, dùng X∈[0,1] trực tiếp. Cặp R vs RGB kiểm soát ảnh hưởng của kiến trúc train mới. So R với G/L vẫn khác encoder, pretraining và số tham số, nên không được diễn giải thành phép cô lập hoàn toàn representation.

### 5.3. Training recipe mặc định

| Tham số | R và RGB-control | G/L trên feature cache |
| --- | --- | --- |
| Trainable | Toàn bộ TinyCNN; filter cố định | MLP head; DINO frozen |
| Epoch | 20 | 20 |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | CrossEntropyLoss với logits và int64 labels | Như Week 2 |
| Scheduler / early stopping | Không trong run chính | Giữ protocol Week 2 |
| Augmentation | Không trong run chính | Không |
| AMP | CNN FP16 + GradScaler trên CUDA; residual FP32 | Giữ recipe probe cũ |
| Validation | `eval()` + inference mode; score float32 | Như Week 2 |
| Workers | 0 mặc định | 0 |

Không softmax trước CrossEntropyLoss. Trainer ảnh mới giữ một GradScaler xuyên suốt run; save state khi resume. Không thay đổi trainer cache cũ chỉ để đồng nhất implementation nếu việc đó làm mất mốc tái hiện.

Smoke test overfit 16–32 ảnh train nhằm kiểm tra gradient/nhãn, tách khỏi run chính. Nếu loss không giảm, ưu tiên kiểm tra input, gradient, optimizer và precision. Không liên tục đổi kernel/backbone để đuổi theo val score.

## 6. Ma trận thí nghiệm và fusion

### 6.1. Runs phải có

| ID | Cấu hình | Vai trò | Seed mục tiêu |
| --- | --- | --- | --- |
| W3-G | Frozen DINO CLS + MLP | Global control | 42,43,44 |
| W3-L | Frozen DINO patch mean + MLP | Strong in-domain control | 42,43,44 |
| W3-R | Box3 residual + TinyCNN | Nhánh mới chính | 42,43,44 |
| W3-C | RGB + cùng TinyCNN | Control cho phép residual | 42,43,44 |
| W3-GL-mean | Mean xác suất G và L | Control ensemble hai semantic-feature probes | 42,43,44 |
| W3-GR | Mean xác suất G và R | So với Global | 42,43,44 |
| W3-LR | Mean xác suất L và R | Comparison chính cho giá trị bổ sung | 42,43,44 |
| W3-GLR | Mean xác suất G, L, R | Full fusion đơn giản | 42,43,44 |
| W3-LC | Mean xác suất L và RGB-control | Kiểm tra lợi ích ensemble chung | 42,43,44 |
| HIST-GL-concat | G+L concat MLP của Phase 5 | Mốc lịch sử, khác late fusion | 42 |

Có 12 lượt fit chính: 4 model × 3 seeds. Năm cấu hình mean fusion không cần train thêm. Replay historical không phải lượt fit mới.

### 6.2. Công thức và cách merge

- `p_GR = (p_G + p_R)/2`; `p_LR = (p_L + p_R)/2`.
- `p_GLR = (p_G + p_L + p_R)/3`; GL-mean và LC tính tương tự.
- Dùng xác suất Fake, không dùng trung bình hard labels. Không trộn logits nhánh này với probability nhánh kia.
- Ghép theo `(train_seed, split, condition, image_id)`. Assert one-to-one và cùng nhãn; không dựa vào vị trí dòng hoặc thứ tự DataLoader.
- Ghép các model cùng seed. Seed 42 của L không được ghép với seed 44 của R rồi gọi là seed 44 comparison.
- Không dùng weight phụ thuộc nhãn, corruption hoặc từng sample. Không train stacking trên dự đoán in-sample của 1.600 train.
- Report toàn bộ mean fusions đã định trước dù chúng không cải thiện. Không chỉ giữ bảng của cấu hình thắng.

Weighted fusion, temperature scaling hoặc learned gating chỉ là extension: cần inner validation/OOF trong 1.600 train để fit và bản protocol riêng. Không chọn alpha trên 400 val rồi coi score ở đó là unbiased test.

### 6.3. Những comparison quan trọng

1. R vs RGB-control: high-pass input có hữu ích hơn cùng CNN nhận RGB không?
2. LR vs L: residual có thêm giá trị so với ứng viên mạnh nhất đang có không?
3. LR vs LC: gain có đặc thù với residual hay một CNN RGB thứ hai cũng đạt tương tự?
4. GR vs G: cải thiện so với baseline global, nhưng chưa đủ để chọn model nếu vẫn thua L.
5. GLR vs LR và GL-mean: thêm nhánh thứ ba có đáng chi phí không?

Khác biệt error sets không phải điều kiện toán học bắt buộc để AUROC tăng: thứ hạng score có thể cải thiện dù nhãn tại 0.5 giữ nguyên. Vì vậy không dùng error-overlap làm bộ lọc tự động bỏ hết fusion trước khi đo score.

## 7. Error analysis, shortcut diagnostic và robustness

### 7.1. Error overlap trên cùng 400 ảnh

Với B lần lượt là G và L, lập bảng bốn nhóm: cả B/R đúng; B sai R đúng; B đúng R sai; cả B/R sai. Báo riêng Real và Fake.

- `rescue_count = count(B sai, R đúng)`.
- `harm_count = count(B đúng, R sai)`.
- `rescue_rate = rescue_count / count(B sai)`; nếu mẫu số 0 ghi N/A.
- Khi phân tích **fusion**, tính lại rescue/harm của fusion so với B. Không coi mọi ảnh R đúng khi B sai đều được mean fusion sửa.
- Báo cả AUROC, BAcc, confusion matrix và score distributions; một nhánh sai threshold có thể vẫn có ranking tốt.
- Chọn tối đa 8 ảnh/nhóm bằng seed hiển thị cố định để làm montage RGB, residual và probabilities. Không cherry-pick các hình dễ kể chuyện.
- Tách color/near-grayscale theo đúng audit Week 2: tỷ lệ pixel có `max(R,G,B)-min(R,G,B)<=2` trên uint8 đạt ít nhất 0.99. Dùng nhóm của ảnh clean để theo dõi cùng mẫu dưới corruption; không giả định mỗi split có đúng 50/50.
- Nếu subgroup chỉ có một lớp, AUROC ghi N/A kèm counts. Không infer generator từ màu ảnh, filename hoặc cụm feature.

### 7.2. Shortcut diagnostic nhẹ, không yêu cầu sanitization

Tạo một control CPU để xem thống kê đơn giản đã dự đoán được label đến đâu:

- Logistic regression trên `log1p(file_size_bytes)` riêng.
- Logistic regression trên bộ cố định: log file size, mean/std luminance, mean saturation, gray-pixel fraction, edge mean. Dùng định nghĩa của notebook audit; edge mean chỉ là proxy thô, có thể chịu ảnh hưởng biên filter.
- StandardScaler fit chỉ 1.600 train; LogisticRegression `C=1`, solver lbfgs, max_iter 1000, không grid search. Lưu hội tụ, config, coefficient và predictions.
- Không dùng path, ID, class_name, split, hash hay source_status làm feature.
- Chạy trên dữ liệu clean; đây là diagnostic, không là model forensic cuối cùng và không chứng minh CNN đang dùng shortcut đó.
- Nếu diagnostic mạnh, ghi rõ khả năng confounding; quyết định control bổ sung theo mục 13. Nếu yếu, cũng không kết luận dataset hết bias.

### 7.3. Giữ nguyên bốn condition của Week 2

| Condition | Thao tác trước nhánh model | Ghi chú |
| --- | --- | --- |
| `clean` | Decode RGB gốc | Không re-encode |
| `jpeg_q70` | Encode/decode JPEG quality 70, subsampling 2 | Áp dụng trên RGB gốc trước processor |
| `resize_112_224` | Bilinear xuống 112×112 rồi lên 224×224 | Không thay bằng crop hoặc bicubic |
| `blur_r1` | PIL GaussianBlur radius 1.0 | Bán kính tính trên ảnh đầu vào trước resize |

Mỗi condition bắt đầu từ RGB clean; không nối JPEG rồi blur rồi resize. Tái sử dụng định nghĩa `corrupt_image` của Phase 6 hoặc port có test tương đương.

Đối với R: **corruption RGB → spatial preprocessing → tính residual mới → CNN**. Không blur/nén residual đã tính sẵn. Đối với G/L: corruption RGB → processor DINO → feature → checkpoint probe.

Tính DINO features cho 400 ảnh × 4 condition một lần, dùng lại cho tất cả probe seeds và fusions. Không cần cache toàn bộ patch tokens cho robustness: chỉ lưu CLS và mean patch 768 chiều cùng provenance. Clean phải replay khớp cache Phase 4 trước khi tin corrupted results.

Đánh giá tất cả 9 cấu hình W3 chính ở mỗi seed, cùng threshold 0.5. Khi đủ ba seed, bảng prediction phải có `400×4×9×3 = 43.200` dòng, không tính history/shortcut/historical runs. Lưu riêng prediction bị thiếu thay vì bỏ mẫu im lặng.

Report score clean, score corruption và `delta = corrupted - clean`. Một nhánh có score cao nhất không nhất thiết có mức giảm nhỏ nhất. Chỉ kết luận robustness tại những mức đã thử.

## 8. Tổng hợp nhiều seed và uncertainty

- Bảng chính gồm từng seed và mean±sample SD của ba seeds. SD này đo biến thiên training trên cùng split, không đại diện biến thiên generator hoặc dataset.
- Báo chênh lệch ghép cặp từng seed cho R−C, LR−L, LR−LC, GR−G, GLR−LR. Không lấy run tốt nhất trong ba seed làm headline.
- Phân tích uncertainty tùy đủ thời gian: 2.000 bootstrap samples, seed riêng 2026, resample ảnh theo nhãn để giữ cả hai lớp; cùng bootstrap indices cho tất cả model trong một comparison.
- Nếu tổng hợp delta ba seed, mỗi bootstrap resample cùng ảnh cho ba seeds rồi lấy mean delta; không xem 1.200 predictions như 1.200 ảnh độc lập.
- Report percentile interval 95% cho delta; ghi là conditional trên split và checkpoint đã chọn, không bao gồm selection uncertainty hoặc source uncertainty.
- Nếu biết nhóm ảnh cùng subject/near-duplicate về sau, cần group/cluster bootstrap; image-level CI hiện dựa vào giả định độc lập tương đối giữa ảnh.
- Không thêm hàng loạt p-value hoặc claim significance từ sweep nhiều model. Nếu cần kiểm định formal, khóa hypothesis/analysis riêng và xử lý multiple comparisons trước.

## 9. Tổ chức Kaggle và notebook độc lập

### 9.1. Môi trường, input và artifact cần chuẩn bị

Mặc định thiết kế cho một GPU Kaggle khả dụng, mục tiêu T4 16 GB hoặc tương đương; không giả định người dùng luôn có đúng loại GPU/quota. Nếu được cấp hai GPU, bản đầu dùng một GPU, chưa cần DDP/DataParallel.

Attach các input sau qua Add Input, ghi lại owner/slug/version thực tế khi chạy:

1. Who Is AI chứa `train/manifest.csv` và `train/images/`.
2. Output Phase 4: cache DINO + processor + train/val manifests + cache manifest.
3. Output Phase 5: checkpoint G/L/GL-concat, config và predictions.
4. Output Phase 6 nếu còn: clean replay và robustness predictions. Không bắt buộc có nếu sẽ tái tạo.
5. Snapshot code mới đúng commit hoặc repository clone được pin commit.

Không hardcode mount account của teammate. Cho phép đặt `DATA_ROOT`, `PHASE4_ROOT`, `PHASE5_ROOT`, `PHASE6_ROOT` thủ công. Auto-discovery phải báo lỗi khi tìm thấy nhiều candidate thay vì lấy candidate đầu.

`/kaggle/input` dùng đọc. Cache giải nén, checkpoint và output mới đặt dưới `/kaggle/working/week3/`. Không giả định file trong working còn tồn tại sau khi session kết thúc. Mỗi giai đoạn cần Save Version/run hoàn tất, xác nhận output lưu được và attach version đó cho notebook kế tiếp.

Weights DINO cần quyền truy cập checkpoint tương ứng và Hugging Face Secret như Week 2 nếu phải tải lại. Notebook chỉ train TinyCNN/probe từ cache không cần nạp DINO weights. Internet chỉ cần lúc tải code/dependency/weights chưa attach; nếu chạy offline thì chuẩn bị snapshot và dependency/weights trước.

Không reinstall toàn bộ PyTorch/CUDA khi môi trường đang tương thích. Dùng môi trường Week 2 làm mốc, kiểm tra Transformers hỗ trợ DINOv3; pin phiên bản đã smoke-test và lưu version thực chạy. Không suy ra `requirements.txt` với dấu `>=` là môi trường tái lập hoàn chỉnh.

### 9.2. Notebook 05: residual và controls

File đề xuất: `notebooks/05_residual_experiment.ipynb`.

| Thứ tự cell/section | Việc phải làm | Cổng kiểm tra |
| --- | --- | --- |
| 1 | Hướng dẫn input, GPU, output và protocol | Người chạy biết đây là in-domain |
| 2 | Bootstrap repo theo commit, dependency, import | Không phụ thuộc notebook trước; log code revision |
| 3 | Đọc config/path overrides, kiểm tra GPU/disk | Thiếu input thì dừng sớm |
| 4 | Nạp manifest/cache/provenance, kiểm tra ID-label | Đúng 1.600/400, cùng cache Week 2 |
| 5 | Replay G/L/GL-concat seed 42 | Đối chiếu predictions/metrics lịch sử |
| 6 | Kiểm tra residual bằng synthetic input + montage nhỏ | Hằng số không sinh viền; finite/shape đúng |
| 7 | TinyCNN smoke train trên train subset | Gradient/loss/checkpoint roundtrip hoạt động |
| 8 | Shortcut diagnostic clean | Fit statistics/scaler bằng train |
| 9 | Train G/L/R/C theo seed, mỗi run có directory riêng | Save best, last, history sau mỗi epoch |
| 10 | Xuất clean predictions/metrics, parameter count/runtime | Schema và sample counts hợp lệ |
| 11 | Export artifacts/checksums, hướng dẫn attach sang 06 | Có file thật trong output, không chỉ biến trong RAM |

Nếu phải chia session, chọn danh sách `TRAIN_SEEDS`/`RUN_IDS` rõ ràng. Chỉ skip run khi completion marker và manifest/hash/config đều khớp; thấy checkpoint tồn tại chưa đủ kết luận run hoàn tất.

### 9.3. Notebook 06: fusion, errors và robustness

File đề xuất: `notebooks/06_residual_fusion_robustness.ipynb`.

| Thứ tự cell/section | Việc phải làm | Cổng kiểm tra |
| --- | --- | --- |
| 1–3 | Bootstrap độc lập; attach data/Week 2 cache/Week 3 train outputs; HF nếu cần | Không dùng đường dẫn working từ session 05 |
| 4 | Verify hash/config/seed inventory; load checkpoints | Nhận diện missing seeds, không điền metric giả |
| 5 | Clean replay G/L/R/C | Kết quả khớp output notebook 05 |
| 6 | Mean fusions + clean comparison | Join one-to-one theo ID/seed |
| 7 | Error groups, subgroup metrics và montage | Giữ cả lỗi được sửa và lỗi mới tạo |
| 8 | Nạp/trích DINO condition features, R/C corruption inference | Cùng raw input và corruption recipe |
| 9 | Corrupted fusions + deltas | Không retune threshold/checkpoint |
| 10 | Aggregate seeds và paired deltas; CI nếu đã chọn | Phân biệt seed SD với image bootstrap |
| 11 | Export report tables/figures/run manifest/ZIP | Có thể tái lập từ artifact mà không train lại |

Kaggle notebook nên gọi hàm trong `src/` cho logic chính. Notebook giữ phần cấu hình, hướng dẫn, chạy và hiển thị; tránh sao chép trainer/filter trong nhiều cell với biến thể khác nhau.

### 9.4. Quota, tốc độ và resume

- Chạy thử một epoch R và C, một pass DINO/400 ảnh; đo wall time, peak VRAM, disk output để ước lượng phần còn lại. Plan chưa có runtime được đo cho nhánh mới nên không cam kết số giờ GPU.
- DINO clean/cache không cần extract lại mỗi seed. TinyCNN train độc lập, không giữ DINO trong GPU cùng lúc nếu không cần.
- Không load toàn bộ patch cache vào GPU. Với cache lớn, mean-pool theo chunk hoặc mmap khi được hỗ trợ; lưu descriptor L gọn và hash liên kết nguồn.
- Nếu OOM, inference giảm batch tự do và kiểm tra sai khác số học; training cần ghi config mới. Với R/C giảm microbatch 32→16 và gradient accumulation 2 để giữ effective batch 32, xử lý đúng batch cuối, áp dụng nhất quán cho cả cặp.
- Checkpoint resume gồm optimizer, scaler, epoch, RNG Python/NumPy/Torch CPU/CUDA và generator dùng shuffle. Resume từ đầu epoch kế tiếp; resume giữa batch là ngoài phạm vi.
- Best checkpoint phục vụ inference; last checkpoint phục vụ resume. Không chỉ lưu best rồi dùng nó như trạng thái cuối epoch.
- Mỗi seed có artifacts hoàn chỉnh trước khi chạy seed kế tiếp. Nếu Kaggle hết quota, vẫn có bộ seed 42 để báo cáo, với hạn chế single-seed rõ ràng.

## 10. Danh sách công việc code sẽ làm sau

Các file dưới đây **chưa được tạo bởi plan này**. Tên có thể điều chỉnh lúc implement nếu tránh trùng module, nhưng phải giữ chức năng và contract.

| File đề xuất | Trách nhiệm | Lưu ý interface |
| --- | --- | --- |
| `configs/week03.yaml` | Split seed, train seeds, paths, residual/CNN/training/fusion/corruption | Mỗi run lưu resolved config, không phụ thuộc default ẩn |
| `src/models/residual.py` | Filter Box3 reflect + TinyCNN | Filter nhận B×3×H×W float, giữ dấu; `forward` trả hai logits |
| `src/input_data/residual_dataset.py` | Đọc manifest, remap path, shared RGB preprocess, corruption | Trả input, image_id, label nếu có; không đổi data gốc |
| `src/training/residual_trainer.py` | Train/eval image model, best/last/resume | Không dùng feature trainer như thể input ảnh là feature 768D |
| `src/evaluation/fusion.py` | Merge predictions + mean probability | Kiểm tra keys, labels và probability range |
| `src/evaluation/week3.py` | Comparisons, errors, subgroups, corruption predictions, export | Gọi metric utilities hiện có; tránh hardcode đúng ba representation |
| `src/evaluation/shortcut_baseline.py` | Scalar diagnostics + train-only scaling | Không đưa vào main model inference |
| `scripts/train_residual.py` | Entry point gọi trainer từ config | Notebook và CLI dùng cùng logic |
| `scripts/evaluate_week3.py` | Manifest/checkpoints → predictions/metrics | Có chế độ prediction-only khi thiếu label |
| `notebooks/05_residual_experiment.ipynb` | Orchestrate training trên Kaggle | Notebook tự chạy được từ kernel sạch |
| `notebooks/06_residual_fusion_robustness.ipynb` | Orchestrate analysis trên Kaggle | Không train lại model mặc định |
| `tests/test_residual.py`, `tests/test_week3_protocol.py` | Test filter, mapping, metrics/fusion, reload | Dùng synthetic inputs CPU; không tải DINO để unit test |
| `docs/week03-results.md` | Báo cáo sau khi chạy | Chỉ tạo/điền kết quả khi có artifact thực |

Phần CLI Week 3 có thể dùng entry point mới để giữ replay Week 2. Việc hoàn thiện CLI cũ đang stub là task riêng nếu quyết định hợp nhất; tránh để notebook chạy được nhưng hướng dẫn CLI vẫn trỏ tới stub.

## 11. Contract artifact và prediction

### 11.1. Thư mục output đề xuất

```text
/kaggle/working/week3/<experiment_id>/
  protocol.json
  resolved_config.yaml
  environment.json
  input_checksums.json
  train.csv
  val.csv
  processor.json
  shortcut_diagnostic/
  runs/
    seed42/{global_only,local_only,residual_only,rgb_control}/
    seed43/{global_only,local_only,residual_only,rgb_control}/
    seed44/{global_only,local_only,residual_only,rgb_control}/
  condition_features/
  analysis/
    predictions.csv
    metrics_per_seed.csv
    metrics_summary.csv
    paired_deltas.csv
    error_overlap.csv
    subgroup_metrics.csv
    robustness_deltas.csv
    figures/
    summary.md
  artifact_manifest.json
```

Notebook 05 và 06 có `experiment_id`/parent artifact IDs liên kết nhau, không ghi đè output từ một config khác. Không copy lại toàn bộ cache DINO gốc vào mọi seed directory.

Mỗi run lưu `best_checkpoint.pt`, `last_checkpoint.pt`, `history.csv`, `validation_predictions.csv`, `run_config.json`, `completion.json`. Checkpoint có model type/version, state_dict, input preprocessing/residual spec, class mapping, train/split seed, selected epoch, threshold và hashes của input manifests. Với G/L ghi model revision/processor/cache hash; với R/C ghi filter và CNN config.

### 11.2. Prediction schema

| Cột | Ý nghĩa |
| --- | --- |
| `experiment_id`, `run_id`, `train_seed`, `split_seed` | Truy vết thí nghiệm |
| `image_id`, `split`, `source`, `source_status` | Metadata mẫu; source có thể chưa biết |
| `label` | 0/1; có thể null trong prediction-only |
| `representation`, `condition` | G/L/R/C hoặc fusion và corruption |
| `prob_fake`, `prediction`, `threshold` | Score, nhãn tại threshold khai báo |
| `checkpoint_id` | Hash checkpoint hoặc danh sách hash nhánh với fusion |
| `preprocess_id`, `condition_version` | Liên kết config thực sự dùng |

Unique key trong một experiment: `(train_seed, representation, condition, split, image_id)`. Score finite thuộc [0,1]. `source` rỗng không được biến thành một generator tên “unknown” rồi dùng làm group split.

### 11.3. Checklist export và tái lập

- Save config đã resolve, revision code và trạng thái code có sửa local; pin code snapshot cho lần replay.
- Hash manifest, processor, cache metadata, weights và checkpoint; ghi dataset/input version nếu có.
- Với ảnh, lưu hash inventory hoặc định danh version + kiểm tra manifest-to-file chặt chẽ; tránh chỉ hash tên file.
- ZIP nằm ngoài thư mục đang đóng gói để không tự bao gồm chính nó; đính kèm checksums.
- Có bước reload artifact trong process/kernel sạch rồi dự đoán lại sample cố định.
- Không lưu key/token trong notebook output, environment dump hoặc ZIP; chỉ log tên package/GPU/config liên quan.
- Không tự publish dữ liệu ảnh/checkpoint public; giữ phạm vi chia sẻ như dataset/input mà nhóm có quyền sử dụng.

## 12. Acceptance tests và lỗi cần tránh

### 12.1. Test CPU trên Mac trước khi đưa lên Kaggle

1. Ảnh hằng số (kể cả biên): residual xấp xỉ 0, tolerance khoảng 1e-6 cho float32.
2. Impulse/checkerboard: output đúng shape, có đáp ứng thông cao, không NaN/Inf; kênh này không bị trộn sang kênh khác.
3. Input có gradient bật: filter vẫn cho gradient đúng khi cần; hệ số filter không trainable. Trainable TinyCNN nhận được gradient và cập nhật weight.
4. RGB-control và R có cùng số tham số/khởi tạo khi reset cùng seed; chỉ đường input khác nhau.
5. Preprocess không normalize hai lần, không rescale uint8/[0,1] hai lần, không thêm crop ngẫu nhiên.
6. Join prediction vẫn đúng khi shuffle thứ tự rows; duplicate, thiếu ID hoặc lệch labels phải báo lỗi.
7. Fusion của hai probability cố định đúng công thức; threshold 0.5 và label Fake=1 được kiểm tra.
8. Metric AUROC nhận probability, không nhận hard label; dataset một lớp được đánh dấu N/A rõ ràng.
9. Checkpoint save/load giữ prediction trong tolerance; missing architecture/config gây lỗi rõ ràng.
10. Dataset root remap giữ nguyên relative path và ID; path không tồn tại bị chặn trước train.
11. Corruption áp dụng trước residual; `clean` không làm thay đổi ảnh; cache fingerprint thay đổi khi recipe thay đổi.
12. Smoke train vài batch synthetic CPU chạy được mà không gọi CUDA vô điều kiện. Full training trên Mac không là yêu cầu.

### 12.2. Kiểm tra tích hợp trên Kaggle

- Kernel sạch chạy tuần tự 05 rồi 06 qua artifact attach, không cần biến của kernel trước.
- Replay Phase 5 cho G/L/GL-concat trước. Đối chiếu max probability difference và metric; tolerance prediction đề xuất 1e-4 phải được kiểm tra thực tế, không nới âm thầm cho qua.
- Nếu hardware/precision làm threshold flips gần 0.5, lưu số mẫu và phân tích; không claim bitwise reproduction.
- Hoàn thành một seed end-to-end trước khi mở rộng ba seeds; prediction count và keys đúng.
- Resume thử ở ranh giới epoch trên một smoke run; kiểm tra history không lặp/mất epoch và state optimizer/scaler tồn tại.
- Inference notebook 06 load model ở `eval()`, không cập nhật weights/normalization từ validation.
- Hash/checksum output và download/attach replay hoạt động.

### 12.3. Những lỗi nghiên cứu dễ mắc

- Residual không mặc nhiên là sensor noise hoặc dấu vết generator; edges/texture/content vẫn có thể còn rất mạnh.
- CNN nhận RGB thua R chỉ hỗ trợ lợi ích phép biến đổi trong setup này, không chứng minh loại bỏ semantic bias.
- Cùng format JPEG, không EXIF hoặc cùng quantization table không chứng minh lịch sử nén giống nhau.
- DINO chỉ nhận pixel tensor thì không đọc trực tiếp header EXIF; audit metadata và loại bỏ shortcut pixel là hai việc khác nhau.
- G+L concat và mean probability G/L là hai model khác nhau; phải đặt tên tách biệt.
- AUROC tăng mà BAcc giảm có thể liên quan threshold/calibration; không chọn threshold bằng corrupted val để che vấn đề.
- Có ba seeds không có nghĩa đã có ba independent datasets hoặc đã đánh giá unseen.

## 13. Các extension có điều kiện

### 13.1. Sanitization/re-encode: hạ ưu tiên

Không tạo dataset sanitized trong path chính Week 3. Chỉ mở control này nếu phát hiện thêm compression lệch nhãn, shortcut diagnostic đáng nghi, hoặc muốn kiểm tra độ nhạy với việc re-encode sau khi core runs đã xong.

Nếu thực hiện, giữ raw; tạo derived version có cùng image IDs/split; declare JPEG Q95, subsampling 2 cho cả hai lớp và lưu file mới không metadata. Retrain các model được so sánh trên derived train và đánh giá derived val; cache DINO phải extract lại. Để phân biệt đổi train distribution và test degradation, thêm bảng train-original/test-derived khi có đủ compute và label chính xác từng ô.

Re-encode không xóa lịch sử JPEG và không “chứng minh dataset sạch bias”. Đổi JPEG thành PNG cũng không phục hồi thông tin đã mất. Nếu score giảm, có thể do mất tín hiệu forensic thật, do domain shift, do shortcut hoặc hỗn hợp; cần phân tích thêm.

### 13.2. Native-resolution residual

Nếu R224 học kém sau khi đã vượt qua smoke test và đủ quota, thử riêng R512/C512 với cùng CNN, original RGB geometry và recipe đã ghi. Không thay baseline R224. Chạy cả RGB-control cùng resolution để tránh nhầm lợi ích độ phân giải với lợi ích residual. Report chi phí và preprocessing khác DINO224; không tự kết luận R không có thông tin chỉ từ một baseline yếu.

### 13.3. Weighted fusion, SRM, top-k và LoRA

- Weighted fusion cần dữ liệu fit trọng số riêng bên trong training; không phải bước tuning mặc định của notebook 06.
- SRM/FFT/DCT/top-k local/concept branch tạo thêm trục thay đổi; chỉ xét sau khi core residual comparison có kết luận.
- LoRA/anchor-LoRA chuyển Week 4. Table 11 của ARA có kết quả phụ thuộc backbone; không đặt kỳ vọng ARA chắc chắn giúp DINOv3-B.
- Loss bảo toàn representation đơn giản phải được gọi là biến thể đề xuất; không dùng tên “reproduction ARA” nếu thiếu hybrid-data protocol/anchor objective tương ứng.

## 14. Chuẩn bị external evaluation sau này

### 14.1. Điều cần code ngay để tránh làm lại pipeline

- Evaluator nhận manifest, dataset root, checkpoint bundle và condition config; số ảnh bất kỳ, không cố định 400.
- Source/generator là metadata tùy chọn; labels có thể null ở prediction-only. Khi label thiếu, vẫn xuất scores nhưng không tính metric hoặc chọn model.
- Preprocessing có cấu hình serialize cùng checkpoint. External images dùng cùng resize, residual filter, class mapping và threshold đã chọn.
- Prediction IDs có namespace dataset để tránh đụng nhau khi hai dataset đều đánh số ảnh từ 00001.
- Data integrity checks giữa external và train: exact/near duplicates khi khả thi, provenance/generator mapping, license/access và phân bố Real/Fake. Không chỉ kiểm tra tên folder.
- Inference không fit scaler, classifier, fusion weight hoặc calibrator trên external set.

### 14.2. Khi có dataset mới

1. Chọn và đóng băng danh sách model/config/checkpoint trước khi xem metric external. Có thể đánh giá toàn bộ configurations đã định trước để so sánh; nếu chọn winner dựa trên external thì tập này trở thành development data cho lựa chọn đó.
2. Nếu chỉ external test: thường **không cần train lại**; chạy extraction/inference và metric cho model đã khóa.
3. Nếu tổ chức lại train/val/test theo source: cần train lại heads/CNN trên split mới; frozen features có thể tái dùng khi ảnh/processor/model revision không đổi, nhưng scaler/selection phải fit lại đúng training split.
4. Vì source của Who Is AI chưa biết, không thể khẳng định generator bên ngoài chắc chắn chưa xuất hiện trong train chỉ dựa vào việc tải dataset khác. Report external OOD/cross-dataset cho đến khi có bằng chứng provenance.
5. Chọn model bằng validation hiện tại không tự làm external test bị leakage; điều gây lệch là dùng external scores/labels để tuning hoặc có mẫu/nguồn không độc lập trái với claim.

Week 3 chỉ chuẩn bị khả năng nhận dữ liệu mới. Tìm/download external dataset và đánh giá thực tế là việc bổ sung khi nhóm có lựa chọn phù hợp.

## 15. Lịch triển khai và thứ tự ưu tiên

| Buổi/ngày dự kiến | Đầu việc | Đầu ra/cổng hoàn thành |
| --- | --- | --- |
| 1 | Khóa config và input contract; kiểm tra artifact; replay Week 2 | Split/ID/hash đúng, mốc G/L/GL tái hiện được |
| 2 | Implement filter/dataset/TinyCNN; CPU tests; Kaggle smoke | Input/residual/gradient/reload đúng; đo runtime/VRAM |
| 3 | Notebook 05 seed 42, shortcut diagnostic, clean predictions | G/L/R/C có checkpoint/history/predictions |
| 4 | Fusion và error analysis seed 42; debug notebook 06 | GR/LR/GLR/GL-mean/LC có bảng sạch, không lỗi join |
| 5 | Robustness seed 42, cache condition features | Clean/JPEG/resize/blur đủ, model cố định |
| 6 | Seeds 43/44 cho core models; reuse condition features | Ba seed complete hoặc ghi rõ phần còn thiếu |
| 7 | Tổng hợp metrics/deltas/uncertainty; export/replay; báo cáo | Bundle tái lập, decision và backlog Week 4 |

Đây là thứ tự làm việc, không phải cam kết số giờ GPU; có thể gộp buổi khi pipeline chạy nhanh. Nếu thiếu quota, ưu tiên: replay/input integrity → R/C seed42 → LR và controls → fixed robustness → seeds bổ sung → CI/extension. Không hy sinh RGB-control để dành compute cho nhiều loại filter chưa có căn cứ.

## 16. Definition of Done và bàn giao sang Week 4

### 16.1. Core hoàn tất ở một seed

- [ ] Dataset/split/provenance checks đạt; không sửa raw hoặc tự đổi split.
- [ ] G/L và historical GL replay được; sai khác có giải thích.
- [ ] R và RGB-control được train, có best/last checkpoints và history.
- [ ] Clean metrics/predictions đủ G/L/R/C và năm mean fusions.
- [ ] Có R vs C, LR vs L và LR vs LC; không chỉ so với Global.
- [x] Error rescue/harm được tính cho cả branch và fusion (Notebook 08; seed 42).
- [ ] Shortcut diagnostic và color/gray subgroup được báo với giới hạn.
- [ ] Bốn corruption conditions đủ trên cùng validation IDs.
- [ ] Notebook 05/06 độc lập, artifacts attach/reload được.
- [ ] Tất cả conclusions ghi đúng phạm vi validation in-domain.

### 16.2. Mục tiêu đầy đủ cuối tuần

- [ ] Đủ seeds 42/43/44 trên cùng split; bảng per-seed và mean±SD.
- [ ] Có paired deltas; CI mô tả nếu triển khai, không diễn giải quá mức.
- [ ] Có runtime/parameter count/peak VRAM và bundle có checksums.
- [ ] `docs/week03-results.md` ghi điều học được, lỗi còn mở và cấu hình đề xuất Week 4.

Không có sanitized dataset, unseen test hay LoRA vẫn có thể hoàn tất core Week 3 theo phạm vi đã thống nhất. Nếu chỉ có một seed, ghi rõ “core hoàn tất; độ ổn định nhiều seed còn thiếu”, không đánh dấu toàn bộ mục tiêu đầy đủ đã xong.

### 16.3. Cách ra quyết định từ kết quả

| Quan sát | Diễn giải được phép | Hành động Week 4 |
| --- | --- | --- |
| LR hơn L khá nhất quán, LR cũng hơn LC | Candidate evidence cho residual bổ sung trong setup in-domain | Mang L/R/LR sang external eval khi có dữ liệu |
| GR hơn G nhưng LR không hơn L | Residual hỗ trợ global yếu hơn, chưa thêm giá trị vượt local | Giữ L làm control mạnh; không mặc định full fusion |
| LR và LC tăng tương tự | Gain có thể là lợi ích ensemble/CNN thứ hai | Không claim residual đặc thù; cân nhắc model đơn giản |
| R yếu, CNN RGB cũng không học | Chưa đủ bằng chứng để bác bỏ residual | Kiểm tra trainer/capacity/preprocess, tránh overclaim negative result |
| R clean tốt nhưng suy giảm mạnh khi corruption | Tín hiệu nhạy post-processing tại mức đã thử | Đánh giá trade-off và control; chưa chứng minh shortcut |
| AUROC tăng nhưng BAcc không tăng | Ranking tốt hơn, threshold cố định chưa cải thiện | Report cả hai, nghiên cứu calibration riêng nếu cần |
| GLR thua L/LR | Thêm branch không có lợi trong protocol này | Không chọn fusion chỉ vì nhiều paper có nhiều nhánh |
| Results đổi dấu giữa seeds hoặc CI rộng | Evidence chưa ổn định | Giữ nhiều candidates, chưa tuyên bố thắng chắc |

Chọn candidate bằng evidence và chi phí, không theo số nhánh. Kết quả null/âm vẫn hoàn thành mục tiêu nghiên cứu nếu protocol, kiểm tra và artifacts đầy đủ.

## 17. Thông tin còn mở nhưng không chặn lập plan

- GPU/quota Kaggle thực tế: xử lý bằng smoke timing và chế độ một/ba seed.
- Artifact Phase 4/5 còn được lưu trên tài khoản nào, version nào: cần xác định trước khi chạy; nếu mất thì tái tạo từ code/data và ghi đây là run mới.
- Source/generator mapping và external test: tiếp tục là việc chưa giải quyết, không giả định có trong manifest.
- Các giá trị mặc định của residual/CNN trong tài liệu là lựa chọn triển khai ban đầu. Nếu đổi sau smoke test, ghi lý do và version trước full run, không chỉnh dựa trên external results.

Hiện không cần người dùng chọn thêm architecture hoặc có GPU Mac để bắt đầu implementation về sau. Khi triển khai, hai đầu vào thực tế quan trọng nhất là artifact Week 2 và GPU session Kaggle khả dụng.

## 18. Nguồn đọc nhanh và ghi chú phương pháp

- Yêu cầu gốc: [plan.md](plan.md), mục Tuần 3 và Tuần 4; [research-outline.md](research-outline.md), RQ2/RQ3 và residual.
- Trạng thái thực nghiệm: [week02.md](week02.md), [progress.md](progress.md), output notebooks 03/04.
- Evidence và caveats: [paper-review.csv](../research/matrix/paper-review.csv), [evidence-matrix.csv](../research/matrix/evidence-matrix.csv), paper cards dẫn ở mục 3.
- Blueprint tham khảo: [paper-informed-blueprint.md](../research/outline/paper-informed-blueprint.md). Quyết định hiện tại về sanitization/source-disjoint lấy theo mục 1 tài liệu này.
- Hướng dẫn procedural: `experimental-design` ảnh hưởng cách giữ split/controls và lặp seed; `kaggle` ảnh hưởng thiết kế notebook độc lập và lưu artifact. Tài liệu không phải đăng ký trước thí nghiệm hoặc bản thảo đã được human verification để nộp hội nghị.
- Tham khảo công cụ hướng dẫn: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065. Metadata đã được đối chiếu ở lượt xây research base trước; đây là nguồn phương pháp làm việc, không là bằng chứng detector.
