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

**Trạng thái: đã triển khai, chờ chạy Kaggle GPU và xem output.** Entry point là `notebooks/04_error_robustness.ipynb`, gọi module `src/evaluation/phase6.py`. Không chạy DINOv3 trên local.

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

**Điều kiện hoàn tất:** notebook chạy hết và tái hiện checkpoint Phase 5; đủ prediction/metrics cho mọi nhánh/condition; xem montage và ghi nhận xét; diễn giải precision/robustness cùng giới hạn in-domain. Chưa có kết quả GPU để xác nhận các điều kiện này.

## Phase 7 — Tổng hợp và quyết định cuối tuần

**Trạng thái: đã chuẩn bị tổng hợp, chờ output Phase 6 và review cuối.** Dùng chính `docs/week02.md` làm báo cáo tuần; notebook 04 hiển thị bảng và xuất `week02_summary.json` để đối chiếu. Cập nhật kết quả đã đo vào doc này sau khi chạy, gồm:

- phiên bản dữ liệu, audit và shortcut risks;
- nguồn/generator đã biết và giới hạn metadata;
- định nghĩa split, seed, model/preprocessing và cấu hình;
- bảng Global-only, Local-only, Global + Local với Val AUROC/BAcc và Unseen AUROC/BAcc (nếu hợp lệ);
- phân tích lỗi và robustness sanity check;
- đối chiếu precision/cache và chênh lệch baseline Phase 3/5, phân biệt kết quả đo với nguyên nhân chưa xác định;
- kết luận H1 cùng hạn chế.

**Quy tắc quyết định:** giữ Local cho Tuần 3 nếu có bằng chứng cải thiện rõ trên held-out source và không chỉ tăng in-domain validation. Nếu không có source metadata hoặc unseen-source test hợp lệ, kết luận H1 là **chưa xác định được**, không phải Local không hiệu quả; cần ưu tiên bổ sung metadata/dữ liệu đánh giá.

**Nhận định hiện tại:** Local-only là ứng viên cho thí nghiệm tiếp theo nhờ kết quả validation Phase 5. Chưa xác nhận H1. Ưu tiên Tuần 3 là bổ sung source/generator metadata hoặc dataset có nguồn xác định để đánh giá unseen-source; robustness chỉ là sanity check bổ sung. Báo cáo cuối cần ghi rõ một seed, một split in-domain, checkpoint được chọn và phân tích trên cùng validation, cùng việc concat tăng số tham số classifier.

## Thứ tự ưu tiên khi thiếu thời gian hoặc GPU

1. Audit + truy tìm source/generator metadata.
2. Split cố định và kiểm tra không leakage.
3. Global + Linear baseline.
4. Cache global/patch features.
5. So sánh Local-only và Global + Local bằng classifier trên cache.
6. Error analysis.
7. Robustness sanity check và MLP baseline.
