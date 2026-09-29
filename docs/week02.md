# Kế hoạch triển khai Tuần 2

## Mục tiêu cuối tuần

Hoàn thiện một quy trình có thể tái lập để trả lời câu hỏi H1: **DINOv3 local patch feature có cải thiện khả năng generalization so với global-only hay không?** Kết luận chính phải dựa trên unseen-source test nếu xác định được nguồn/generator và tạo được split hợp lệ.

Phạm vi tuần này gồm audit dữ liệu, cố định split, baseline Frozen DINOv3, so sánh Global/Local/Global + Local, phân tích lỗi và sanity check robustness nhỏ. Chưa triển khai residual, fusion phức tạp, LoRA hoặc fine-tuning.

## Tình trạng dữ liệu đã biết

Dataset nằm tại `data/Who_is_AI/data/`:

- `train/`: 2.000 ảnh có nhãn, cân bằng 1.000 Real và 1.000 Fake. Manifest `train/manifest.csv` có cột `label`, `class_name`, `path`.
- `public_test/`: 200 ảnh không thấy nhãn đi kèm.
- `private_test/`: 200 ảnh không thấy nhãn đi kèm; ảnh nằm dưới `private_test/private_test/images/`.
- Các ảnh đã kiểm tra đều đọc được, là JPEG RGB kích thước 512×512.
- Hiện chưa thấy metadata source/generator. Do đó chưa thể khẳng định split theo ảnh là unseen-source evaluation. Cần truy tìm metadata trước khi cố định split chính.
- `baseline_TACVU1.ipynb` là notebook CNN để tạo submission cuộc thi; không phải baseline DINOv3 của nghiên cứu.

Thư mục `data/` bị loại khỏi Git qua `.gitignore`; không commit ảnh hoặc artifact dữ liệu.

## Phase 1 — Audit dữ liệu và chốt manifest

**Việc làm**

1. Hoàn thiện `notebooks/00_dataset_audit.ipynb` để đọc manifest và thống kê:
   - phân bố nhãn, file thiếu/hỏng và đường dẫn không tồn tại;
   - kích thước, định dạng, dung lượng file và EXIF;
   - EDA màu sắc theo lớp Real/Fake: histogram RGB chuẩn hóa chồng nhau, histogram độ sáng (grayscale/luminance) và saturation; thống kê theo từng ảnh về saturation trung bình, tỷ lệ pixel gần grayscale, độ sáng và độ tương phản. So sánh phân bố giữa hai lớp, không chỉ xem histogram gộp;
   - duplicate chính xác (hash) và các tên file có thể tiết lộ nhãn;
   - khác biệt giữa hai lớp về dung lượng, metadata ảnh, crop/background nếu có thể kiểm tra.
2. Tạo montage cân bằng số lượng ảnh Real/Fake, lấy mẫu cố định bằng seed; xem riêng ảnh màu và gần grayscale để nhận biết khác biệt về crop, background, ánh sáng và chất lượng ảnh. Chỉ dùng như bước phát hiện pattern cần kiểm tra, không làm bằng chứng duy nhất.
3. Tìm tài liệu gốc hoặc metadata dataset để xác định nguồn ảnh/generator của từng mẫu. Ghi rõ nguồn đã biết, nguồn thiếu và mức độ tin cậy.
4. Chuẩn hóa manifest dùng chung thành ít nhất các trường `image_path`, `label`, `source`, `split`; giữ `class_name` và metadata gốc nếu có.
5. Sửa cấu hình/loader cho định dạng thực tế: `path` hiện là đường dẫn tương đối từ thư mục split, còn split test chưa có nhãn.

**Cách đọc EDA màu sắc**

- Ảnh JPEG có thể được mở với mode `RGB` dù pixel gần như grayscale. Xác định ảnh grayscale bằng nội dung pixel (ví dụ tỷ lệ pixel có chênh lệch R/G/B rất nhỏ), không suy ra từ mode lưu/giải mã.
- Vẽ histogram RGB chuẩn hóa riêng cho mỗi lớp; bổ sung luminance và saturation để phân biệt ảnh tối/sáng hoặc nhạt màu. Vì ảnh hiện cùng kích thước, histogram pixel gộp có thể dùng để sàng lọc, nhưng thống kê image-level (mỗi ảnh một điểm) hữu ích hơn để tránh một số ảnh chi phối kết quả.
- Nếu tỷ lệ grayscale/saturation hoặc độ sáng khác rõ giữa Real và Fake, xem đây là shortcut tiềm năng; kiểm tra nguồn gốc, file size/metadata và kết quả trên các nhóm màu trước khi quyết định preprocessing. Không tự động chuyển toàn bộ ảnh sang grayscale vì có thể làm mất tín hiệu hữu ích.

**Đầu ra**

- Notebook audit hoàn chỉnh.
- `outputs/dataset_summary.csv` và một bản ghi ngắn về shortcut/metadata trong notebook hoặc `docs/`.
- Manifest đã chuẩn hóa, không làm thay đổi dữ liệu gốc.

**Cổng quyết định**

Nếu không tìm được source/generator cho các ảnh train, ghi nhận rõ giới hạn. Không gọi random split theo ảnh là unseen-generator test và không dùng public/private test không nhãn để tính metric.

## Phase 2 — Cố định split và protocol

**Trạng thái: hoàn tất theo nhánh không có source metadata (2026-09-28).** Audit xác nhận manifest không có source/generator và ảnh không có EXIF software để xác định nguồn. Vì vậy split dưới đây chỉ là in-domain/protocol sanity check; chưa có `test_unseen` và chưa thể dùng để kết luận H1.

Protocol đã khóa:

- Script tái tạo: `scripts/create_splits.py`; seed `42`; stratified theo nhãn với 20% validation.
- Số lượng: train 1.600 ảnh (800 Real, 800 Fake); validation 400 ảnh (200 Real, 200 Fake).
- Manifest: `outputs/splits/train.csv`, `val.csv`; bảng đếm tại `counts_by_split_label_source.csv`; protocol máy đọc tại `protocol.json`. Đây là artifact local bị Git ignore.
- `image_path` là đường dẫn tương đối từ repository root; `source` để trống và `source_status=unavailable`, không đại diện cho một source chung.
- Preprocessing tối thiểu: kích thước 224 theo config baseline, dùng processor chính thức của checkpoint DINOv3, không chuyển grayscale hoặc thêm augmentation. Threshold cố định `0.5`; chọn head/checkpoint theo validation AUROC.
- Kiểm tra khi tạo split: không giao nhau theo ID/path, đủ 2.000 mẫu và mọi đường dẫn tồn tại.

Không tạo `test_unseen.csv`; public/private test không nhãn vẫn bị loại khỏi metric. Cần bổ sung metadata nguồn hoặc dữ liệu có nguồn xác định trước khi có thể tạo split unseen-source.


**Việc làm**

1. Nếu có source/generator metadata: dành tối thiểu một nguồn Fake hoàn toàn khỏi train và validation làm `test_unseen`; tạo `train.csv`, `val.csv`, `test_unseen.csv` theo nhóm nguồn, đồng thời giữ cân bằng nhãn trong khả năng dữ liệu cho phép.
2. Tạo in-domain validation từ các nguồn có mặt trong train để kiểm tra pipeline và khả năng học bài toán.
3. Nếu không có source metadata: cố định stratified train/validation split theo ảnh chỉ làm **in-domain/protocol sanity check**; đánh dấu nghiên cứu chưa có bằng chứng unseen-source và tiếp tục tìm dữ liệu/metadata phù hợp.
4. Lưu seed, danh sách ID/đường dẫn mỗi split và kiểm tra không giao nhau. Không thay split giữa các model.
5. Chốt preprocessing tối thiểu, image size, quy tắc threshold và cách chọn checkpoint; không chọn cấu hình bằng test.

**Đầu ra**

- Các manifest split bất biến trong `outputs/splits/` (hoặc vị trí dữ liệu cục bộ được ignore bởi Git).
- Bảng số lượng theo split × nhãn × source và xác nhận không trùng mẫu giữa các split.

## Phase 3 — Hoàn thiện pipeline và baseline Global

**Trạng thái (2026-09-29): hoàn tất baseline Phase 3.** Đã nối dataset từ manifest, Frozen DINOv3 global/CLS feature, classifier và validation evaluator trong `src/`. Notebook `notebooks/01_global_baseline.ipynb` đã chạy thành công trên Kaggle GPU với split Phase 2, trích feature trong lúc chạy baseline, lưu history/config/môi trường/checkpoint/metrics và đóng gói artifact vào `/kaggle/working/phase3_baseline_artifacts.zip`. Đã nâng minimum Transformers lên 4.56.1 để hỗ trợ DINOv3.

Kết quả validation in-domain:

| Head | Best epoch | Val AUROC | Val Balanced Accuracy | Val Accuracy | Val F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global + Linear (B0) | 18 | 0.926725 | 0.865 | 0.865 | 0.860104 |
| Global + MLP (B1) | 3 | **0.932275** | **0.870** | **0.870** | 0.857143 |

Head được chọn là B1 MLP theo validation AUROC, sau đó mới xét Balanced Accuracy khi hòa điểm. Đây là baseline in-domain; chưa có test unseen-source và không được dùng để claim H1 về generalization. Run có một số cảnh báo dọn DataLoader worker và cảnh báo API AMP deprecated, nhưng không làm gián đoạn việc huấn luyện hay tạo metrics/checkpoint.

**Thứ tự notebook:** `00_dataset_audit.ipynb` → `01_global_baseline.ipynb` (Phase 3) → `02_dinov3_feature_extraction.ipynb` (Phase 4, cache dùng lại) → `03_local_experiment.ipynb` (Phase 5) → `04_error_robustness.ipynb` (Phase 6 và tổng hợp định lượng Phase 7). Mỗi notebook chạy trong một kernel Kaggle độc lập.

Kết quả chỉ là in-domain validation vì Phase 2 chưa có held-out source.

**Việc làm**

1. Cố định môi trường, config, seed, device và thư mục run. Chạy GPU trên Kaggle nếu môi trường local không có tài nguyên phù hợp.
2. Kiểm tra khả năng truy cập checkpoint DINOv3 và preprocessing đúng với model; ghi lại model ID/version.
3. Nối pipeline `Dataset → DINOv3 → Feature → Classifier → Evaluation`. Dataset trả ảnh, nhãn và metadata; evaluation cần gom kết quả theo source.
4. Đóng băng toàn bộ backbone. Chạy ít nhất:
   - **B0:** global feature + linear classifier.
   - **B1:** global feature + MLP nhỏ.
5. Chọn head theo validation, không theo test. Cố định classifier/training protocol dùng cho so sánh Local.

**Đầu ra**

- Code trong `src/` và entry point script chạy được từ config.
- Log config/seed, loss theo epoch, checkpoint classifier và metrics validation.
- Baseline in-domain; unseen-source metrics chỉ khi Phase 2 tạo được split hợp lệ.

## Phase 4 — Trích xuất và cache DINOv3 features

Notebook dùng cho phase này là `notebooks/02_dinov3_feature_extraction.ipynb`. Đây là notebook Kaggle độc lập: mỗi lần mở kernel mới, notebook tự cài dependency, clone repo, tìm dataset, xác thực Hugging Face, tái tạo split và load DINOv3; nó không dùng state từ notebook 01. Chạy sau khi đã có baseline Phase 3; mục tiêu là trích một lần và lưu cache để Phase 5 không phải chạy backbone lặp lại.

**Trạng thái: hoàn tất.** Notebook 02 đã trích global `(2000, 768)` và patch `(2000, 196, 768)` bằng AMP rồi lưu float16; patch không chứa CLS/register tokens. Model revision là `5931719e67bbdb9737e363e781fb0c67687896bc`. Notebook 03 đã nạp lại cache trong kernel mới và xác nhận thứ tự ID/nhãn khớp split manifests trước khi huấn luyện. ZIP cache: `/kaggle/working/phase4_dinov3_feature_cache.zip`.

**Việc làm**

1. Dùng backbone Frozen để trích xuất một lần cho toàn bộ ảnh thuộc train/validation/test đã định danh.
2. Lưu global CLS feature, patch tokens, labels và metadata kèm ID ảnh, split, source, preprocessing và model ID.
3. Đảm bảo cache được truy vết theo phiên bản; không dùng nhãn/feature test để fit hay chọn classifier.
4. Nạp cache và xác nhận số lượng/thứ tự ID khớp manifest trước khi huấn luyện các head.

**Đầu ra**

- `global_features.pt`, `patch_features.pt`, `labels.pt`, `metadata.csv`, `cache_manifest.json` và split manifests trong thư mục artifact bị ignore bởi Git; notebook cũng tạo ZIP để tải từ Kaggle.
- Notebook `02_dinov3_feature_extraction.ipynb` gọi lại code từ `src/`, không chứa implementation trùng lặp.

## Phase 5 — Thí nghiệm Local tối thiểu

Notebook `notebooks/03_local_experiment.ipynb` đã được chuẩn bị như một Kaggle workflow độc lập cho kernel mới. Notebook nhận cache Phase 4, kiểm tra lại thứ tự `image_id` với split manifests rồi chạy cùng MLP protocol của Phase 3 cho Global-only, Local-only và Global + Local. Local dùng mean pooling trên patch tokens; Global + Local dùng phép nối hai vector, không chuẩn hóa branch trong lần so sánh đầu tiên. Kết quả và prediction validation được lưu vào `/kaggle/working/phase5_local_outputs/`.

**Trạng thái: hoàn tất so sánh in-domain (2026-09-29).** Cả ba nhánh đã chạy đủ 20 epoch và tạo checkpoint, history, validation predictions, metrics và ZIP artifact tại `/kaggle/working/phase5_local_experiment_artifacts.zip`. Nhánh được chọn theo validation AUROC là Local-only ở epoch 7.

| Representation | Best epoch | Val AUROC | Val Balanced Accuracy | Val Accuracy | Val F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global-only | 7 | 0.930025 | 0.8400 | 0.8400 | 0.842365 |
| Local-only (mean patch) | 7 | **0.952550** | **0.8825** | **0.8825** | **0.885645** |
| Global + Local (concat) | 14 | 0.945550 | 0.8725 | 0.8725 | 0.873449 |

Trong validation này, Local-only cao hơn Global-only (+0.022525 AUROC, +0.0425 Balanced Accuracy). Global + Local cũng cao hơn Global-only nhưng thấp hơn Local-only. Đây chưa phải bằng chứng H1 vì split không có held-out source/generator. Global-only của Phase 5 (AUROC 0.930025, BAcc 0.8400) cũng thấp hơn kết quả Global + MLP của Phase 3 (AUROC 0.932275, BAcc 0.8700).

Rà soát code thấy Phase 3 đánh giá backbone bằng float32, còn Phase 4 trích bằng AMP và lưu float16; Phase 5 chuyển cache sang float32 cho classifier. Notebook 04 đo ảnh hưởng precision bằng cách giữ nguyên từng checkpoint và đổi giữa cache, clean AMP và clean float32. Hai run cũng huấn luyện/chọn checkpoint riêng, nên chưa thể quy toàn bộ chênh lệch metrics cho precision.

**Đối chiếu đã chạy ở Phase 6:** cùng head Phase 5, đổi precision chỉ tăng AUROC 0.000025–0.000050, BAcc/Accuracy/F1 giữ nguyên. Head MLP Phase 3 trên cache đạt AUROC 0.932225, BAcc 0.8700; trên clean float32 đạt 0.932275/0.8700, tái hiện kết quả gốc. Vì vậy precision ở bước đánh giá không giải thích được chênh lệch BAcc 0.8700 → 0.8400 giữa hai head. Nguyên nhân cụ thể ở quá trình huấn luyện/chọn checkpoint chưa được tách riêng; so sánh representation chính vẫn dùng ba head cùng pipeline Phase 5.

**Việc làm**

1. Chọn trước một cách gộp patch đơn giản, ưu tiên mean pooling trong lần so sánh chính.
2. Chạy ba cấu hình:
   - Global-only;
   - Local-only;
   - Global + Local.
3. Giữ nguyên backbone/checkpoint, preprocessing, split, seed, classifier protocol và quy trình chọn model. Với Global + Local, chuẩn hóa/ghép feature theo quy tắc ghi trong config.
4. Báo cáo AUROC và Balanced Accuracy là metric chính; Accuracy và F1 là bổ sung. Báo cáo validation và từng held-out source riêng nếu có.

**Đầu ra**

- Notebook `03_local_experiment.ipynb` và bảng kết quả so sánh.
- Kết luận dựa trên chênh lệch ở unseen-source test. Cải thiện validation đơn thuần không đủ để xác nhận H1.

## Phase 6 — Phân tích lỗi và robustness sanity check

**Trạng thái: hoàn tất phân tích in-domain (2026-09-29).** `notebooks/04_error_robustness.ipynb` chạy hết 11 code cells trên Kaggle Tesla T4, PyTorch 2.10.0+cu128, không có error output. Đã attach cả artifact Phase 3, xem sáu montage và đối chiếu metrics. Các nhận xét review được ghi dưới đây; trong run gốc `OBSERVATION_NOTES` còn trống nên JSON artifact vẫn ghi trạng thái chờ visual review.

**Input và cách chạy**

- Push source mới lên `main` hoặc attach snapshot repo có module Phase 6 trước khi chạy notebook.
- Kernel mới: bật GPU/Internet, cấp Kaggle Secret `HF_TOKEN`; attach dataset Who Is AI gốc, cache/ZIP Phase 4 và artifact/ZIP Phase 5.
- Nên attach thêm artifact/ZIP Phase 3 để đánh giá lại checkpoint MLP cũ. Nếu thiếu, diagnostic ghi rõ chưa đối chiếu được baseline cũ.
- Notebook tự tìm thư mục hoặc ZIP; khi có nhiều bản artifact, chọn rõ `PHASE4_ROOT`, `PHASE5_ROOT`, `PHASE3_ROOT`. Đường dẫn ảnh được dựng lại từ `original_path` và dataset hiện tại, không phụ thuộc mount của kernel trước.

**Việc làm**

1. Xác nhận ID/nhãn/thứ tự cache; load ba checkpoint Phase 5 và tái hiện probabilities/metrics đã lưu trước khi phân tích.
2. Xuất dự đoán và nhóm lỗi Global-only so với Global + Local, đồng thời so với Local-only vì nhánh này đang tốt nhất in-domain. Giữ confidence, nhãn, source và nhóm cả hai cùng sai; montage lấy tối đa 8 ảnh/nhóm bằng seed 42. Ghi nhận xét sau khi xem ảnh vào `OBSERVATION_NOTES` trong notebook. Vùng mắt, tóc, da, biên mặt và texture chỉ là gợi ý quan sát, không phải bằng chứng vùng model chú ý.
3. Trích lại clean validation bằng đúng model revision/processor đã lưu với AMP float16 và float32; đối chiếu feature/metrics với cache bằng cùng checkpoint. Nếu có artifact Phase 3, đánh giá thêm MLP cũ trên cache và clean float32 để đối chiếu kết quả gốc.
4. Chạy cả ba nhánh trên cùng 400 ảnh validation, giữ checkpoint/threshold `0.5`, dùng clean AMP làm mốc và ba corruption cố định:
   - JPEG encode/decode: quality `70`, subsampling `2`;
   - resize bilinear: giảm xuống `112×112`, tăng lên `224×224`;
   - Gaussian blur: radius `1.0` pixel trên ảnh gốc.
   Mỗi corruption áp dụng riêng lên ảnh RGB gốc sau EXIF transpose, trước processor; không ghép các corruption. Feature tính theo AMP/float16 rồi chuyển float32 cho classifier như Phase 5.
5. Lưu AUROC/BAcc/Accuracy/F1 và delta so với clean cho từng nhánh/condition; delta âm là giảm metric. Không chọn lại checkpoint, threshold hoặc mức corruption theo kết quả.

**Đầu ra**

- Thư mục `/kaggle/working/phase6_outputs/dinov3_seed42/`: prediction/error CSV, montage PNG, kiểm tra precision, bảng robustness, biểu đồ và môi trường/protocol kèm hash input.
- `robustness_predictions.csv`: 4.800 hàng = 400 ảnh × 3 nhánh × 4 điều kiện; `robustness_metrics.csv`: metrics và delta so với clean.
- `week02_summary.json` và bảng tổng hợp hiển thị cuối notebook; ZIP `/kaggle/working/phase6_analysis_artifacts.zip`.

**Kiểm tra đã qua:** cả ba checkpoint tái hiện metrics Phase 5, sai lệch probability tối đa khoảng `2.97e-8`. Bảng cache so với clean AMP báo mean/max absolute difference `0.000000`, cosine `1.000000` ở cả global/local. Cell cuối kiểm tra đủ 4.800 prediction rows và đúng thứ tự ID/nhãn trong từng condition/representation trước khi tạo ZIP.

### Phân tích lỗi trên clean validation

| So với Global-only | Cả hai đúng | Cả hai sai | Global sai, nhánh so sánh đúng | Global đúng, nhánh so sánh sai | Số ảnh đúng tăng ròng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global + Local | 327 | 42 | 22 | 9 | +13 |
| Local-only | 316 | 27 | 37 | 20 | +17 |

Global-only sai 64/400 ảnh; Local-only sai 47/400; fusion sai 51/400. Local-only sửa được nhiều lỗi của Global hơn fusion, đồng thời cũng mất 20 trường hợp Global đã đúng. Các nhánh có nhóm lỗi khác nhau; concat trong lần chạy này chưa vượt Local-only.

**Nhận xét từ montage seed 42:** ảnh màu và gần grayscale đều xuất hiện trong các nhóm sai/bất đồng; có nhiều kiểu crop, background, ánh sáng và kính. Chưa thấy một pattern đủ rõ để quy lỗi chung cho riêng mắt, tóc hay da. Một số lỗi có confidence cao: `train/00592.jpg` có nhãn Fake nhưng Global/Local cho probability Fake 0.001/0.016; `train/00819.jpg` có nhãn Real, Global đúng với probability Fake 0.219 nhưng fusion sai với 0.890. Đây là ví dụ quan sát, chưa phải phân tích toàn bộ nhóm hoặc attribution vùng ảnh. Tỷ lệ lỗi theo nhóm màu/crop chưa được tính trong notebook 04.

### Robustness với checkpoint và threshold cố định

Mỗi ô là **AUROC / Balanced Accuracy**, cùng 400 ảnh và threshold 0.5:

| Điều kiện | Global-only | Local-only | Global + Local |
| --- | ---: | ---: | ---: |
| Clean | 0.930025 / 0.8400 | 0.952550 / 0.8825 | 0.945550 / 0.8725 |
| JPEG q70 | 0.928600 / 0.8300 | 0.950525 / 0.8825 | 0.944700 / 0.8825 |
| Resize 112→224 | 0.903000 / 0.8100 | 0.925675 / 0.8300 | 0.918025 / 0.8400 |
| Blur radius 1.0 | 0.927700 / 0.8450 | 0.948275 / 0.8800 | 0.943550 / 0.8725 |

- Resize gây giảm lớn nhất: AUROC giảm khoảng 0.027 ở cả ba nhánh. BAcc Local-only giảm 0.0525 (5.25 điểm phần trăm, 21 ảnh đúng ít hơn), Global giảm 0.0300 và fusion giảm 0.0325. Fusion có BAcc cao hơn Local-only trong điều kiện này dù AUROC thấp hơn.
- JPEG q70 và blur radius 1.0 gây giảm AUROC nhỏ ở mức đã thử. BAcc có thể giữ nguyên hoặc tăng nhẹ: fusion tăng 0.0100 khi JPEG, Global tăng 0.0050 khi blur. Đây là thay đổi quyết định ở threshold cố định trên tập này; chưa chứng minh corruption làm model tốt hơn.
- Local-only giữ AUROC cao nhất ở cả bốn điều kiện, nhưng mức giảm so với clean không luôn nhỏ nhất. Kết quả chưa đủ để kết luận Local bền vững hơn với mọi dạng corruption hoặc source mới.
- Đối chiếu float32 giữ nguyên BAcc/Accuracy/F1 cho cả ba head Phase 5; AUROC Global/Local/fusion lần lượt là 0.930075/0.952575/0.945575. Diagnostic baseline cũ và giới hạn nguyên nhân chênh lệch đã được ghi ở Phase 5.

**Điều kiện hoàn tất đã đáp ứng ở phạm vi in-domain:** run thành công, checkpoint/cache đã đối chiếu, đủ metrics/predictions, montage đã review và có diễn giải precision/robustness. JSON nhận xét của artifact gốc còn trống; báo cáo review chính là doc này. Không có unseen-source evaluation trong run này.

## Phase 7 — Tổng hợp và quyết định cuối tuần

**Trạng thái: hoàn tất tổng hợp Tuần 2 theo nhánh không có source metadata (2026-09-29).** Báo cáo là chính `docs/week02.md`, đối chiếu với output notebook 00–04. Notebook 04 đã tạo bảng tổng hợp và `week02_summary.json`; nhận xét montage cùng kết luận review cuối nằm trong doc này. Nội dung đã tổng hợp gồm:

- phiên bản dữ liệu, audit và shortcut risks;
- nguồn/generator đã biết và giới hạn metadata;
- định nghĩa split, seed, model/preprocessing và cấu hình;
- bảng Global-only, Local-only, Global + Local với Val AUROC/BAcc và Unseen AUROC/BAcc (nếu hợp lệ);
- phân tích lỗi và robustness sanity check;
- đối chiếu precision/cache và chênh lệch baseline Phase 3/5, phân biệt kết quả đo với nguyên nhân chưa xác định;
- kết luận H1 cùng hạn chế.

**Quy tắc quyết định:** giữ Local cho Tuần 3 nếu có bằng chứng cải thiện rõ trên held-out source và không chỉ tăng in-domain validation. Nếu không có source metadata hoặc unseen-source test hợp lệ, kết luận H1 là **chưa xác định được**, không phải Local không hiệu quả; cần ưu tiên bổ sung metadata/dữ liệu đánh giá.

**Kết luận tuần:** Local-only là ứng viên cho thí nghiệm tiếp theo: cao nhất về AUROC trên clean và ba corruption đã thử, sửa ròng 17 lỗi so với Global. Fusion cải thiện so với Global nhưng chưa vượt Local-only về AUROC; ở resize, fusion có BAcc cao hơn. H1 về unseen-source generalization vẫn **chưa xác định được**.

**Giới hạn của báo cáo:** một seed và một split in-domain; checkpoint được chọn và phân tích trên cùng validation; concat tăng input MLP từ 768 lên 1536 nên tăng số tham số; chỉ ba mức corruption cố định; chưa có đánh giá theo source hoặc kiểm định nhiều run. Precision khi đánh giá đã được đối chiếu, nhưng ảnh hưởng của khác biệt quá trình huấn luyện/chọn head chưa được cô lập. Audit có 500/1.000 ảnh gần grayscale ở mỗi lớp; file size, contrast và edge statistics khác nhau vẫn là shortcut tiềm năng, chưa có thí nghiệm kiểm soát để xác định ảnh hưởng.

**Ưu tiên Tuần 3:** bổ sung source/generator metadata hoặc dataset có nguồn xác định, khóa unseen-source split rồi đánh giá Global/Local/fusion trong cùng pipeline. Local-only được giữ làm ứng viên; quyết định về generalization chờ bằng chứng từ nguồn chưa thấy. Robustness hiện tại là sanity check bổ sung cho kết quả in-domain.

## Thứ tự ưu tiên khi thiếu thời gian hoặc GPU

1. Audit + truy tìm source/generator metadata.
2. Split cố định và kiểm tra không leakage.
3. Global + Linear baseline.
4. Cache global/patch features.
5. So sánh Local-only và Global + Local bằng classifier trên cache.
6. Error analysis.
7. Robustness sanity check và MLP baseline.
