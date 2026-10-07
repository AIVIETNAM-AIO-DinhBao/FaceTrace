# Chạy phần forensic AI-Face trên Kaggle

Notebook: `notebooks/11_aiface_phase_e_spectral.ipynb`.
Dataset ảnh: `nguyentrann0703/aiface-pilot-32k-single-bin`.
Không cần Hugging Face token hay tải DINOv3.

## 1. Đưa source code lên Kaggle

Cách mặc định giống notebook 10: clone GitHub `main`.

1. Tạo Notebook Python và import `notebooks/11_aiface_phase_e_spectral.ipynb`.
2. Add Input dataset ảnh `nguyentrann0703/aiface-pilot-32k-single-bin`.
3. Settings: bật Internet, chọn GPU trong Accelerator.
4. Giữ `SOURCE_MODE='git'` và `REPO_REF='main'`; notebook tự clone source. Không cần dataset code hay Hugging Face token.

Cách ZIP dự phòng khi muốn dùng một snapshot local:

1. Lấy `outputs/kaggle_code/facetrace_spectral_source.zip` được tạo từ repo local.
2. Trên Kaggle, tạo một dataset **private** chứa file ZIP code này. Đây là dataset code riêng; dataset ảnh AI-Face vẫn giữ nguyên.
3. Tạo Notebook Python rồi import/upload `11_aiface_phase_e_spectral.ipynb` bằng chức năng import notebook.
4. Add Input cả dataset ảnh AI-Face và dataset ZIP code vừa tạo.
5. Settings: bật Internet để cài dependencies, chọn GPU trong Accelerator.

Nếu đã push source lên GitHub, có thể đổi `SOURCE_MODE='git'` và `REPO_REF` thành branch/tag chứa code spectral. Notebook sẽ clone source từ branch đó. Không push/merge nào được thực hiện tự động bởi implementation local này.

## 2. Chạy thử

Giữ `SOURCE_MODE='git'` (hoặc `'zip'` nếu dùng package), `RUN_FULL=False`, `BASELINE_DIR=None`.
Chạy các cell lần lượt. Notebook kiểm tra:

- unit tests, bao gồm test số học FFT so với NumPy;
- train/checkpoint/predict trên dữ liệu giả nhỏ;
- giải nén archive `.bin` và xác minh manifest, generator, số lượng của dataset thật;
- smoke train một epoch trên ảnh real và mỗi generator của cả ba split;
- prediction giữ đúng ID và thứ tự manifest.

Phải thấy `Forensic-only SMOKE PASS`. Smoke artifacts nằm trong `forensic_only_smoke/`; metric smoke không dùng trong báo cáo nghiên cứu.

Dataset được giải nén vào `/kaggle/temp` để không lưu thêm 32k ảnh vào Output. Code không random split; nó đọc đúng train/val/test đã có. Không có checkpoint/threshold nào được chọn từ test.

Bản single-bin hiện chứa `train/manifest.csv`, `val/manifest.csv`, `test/manifest.csv` và ảnh, nhưng không kèm `subset_metadata.json`. Runner nhận diện dataset từ manifest/ảnh và vẫn kiểm tra schema, đủ 32k ảnh, class balance, generator mapping/disjointness, đường dẫn và ID. Artifact ghi `original_subset_metadata_available=false`, checksum archive/manifest và `manifest_derived_evidence`; không giả tạo metadata gốc hay claim đã đọc quality gate gốc. Nếu metadata được bổ sung ở root hoặc thư mục con, runner sẽ đọc file đó.

## 3. Chạy đầy đủ

1. Điền `DATASET_VERSION` bằng số version của dataset ảnh đang attach (xem metadata/input trên Kaggle).
2. Đổi `RUN_FULL=True`.
3. Save Version và chọn **Save & Run All**. Phiên này chạy lại từ đầu với source/input/config đã lưu, gồm unit test, smoke và full training.
4. Khi thành công, lưu URL của notebook version và tải `facetrace_spectral_artifacts.zip` từ Output.

Nếu chạy tương tác nhiều lần, source hoặc checkpoint cũ có thể đã tồn tại; code dừng để bảo vệ artifact. Restart session trước một run mới, hoặc dùng output directory mới khi chạy script thủ công.

Notebook mặc định train MLP 20 epoch, seed 42. Cấu hình FFT: Hann đối xứng, FFT2 float32, `log1p(abs(spectrum))`, 64 radial bins phủ tâm đến góc phổ trên từng kênh RGB. Vector 192 chiều; MLP 192→128→2 có 24.962 tham số. Chuẩn hóa feature bằng mean/std **chỉ fit trên train**, lưu trong checkpoint.

Radial bins được chia bằng so sánh bình phương khoảng cách và biên bằng số nguyên; pixel đúng trên biên thuộc bin cao hơn, góc phổ thuộc bin cuối. Cách này tránh sai lệch làm tròn sqrt/division giữa NumPy/PyTorch trên Windows/Kaggle. Version quy tắc nằm trong config/metadata; checkpoint theo quy tắc cũ cần train lại trước khi dùng predictor mới.

Resize square bilinear antialias 224×224, EXIF transpose, RGB; không thêm crop/augmentation. Geometry theo runner DINOv3 AI-Face hiện tại; pixel FFT ở [0,1], không ImageNet normalization. Khi baseline đổi processor/geometry, cần đồng bộ cấu hình và kiểm tra lại trước khi so sánh. Mã nguồn processor chính thức: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py

## 4. Artifact cần gửi cho bạn kia

`forensic_only/` chứa:

```text
config_resolved.yaml
train_log.csv
val_metrics.json
predictions_val.csv
predictions_test.csv
test_metrics_overall.json
test_metrics_by_generator.csv
confusion_matrix.csv
checkpoint.pt
environment.json
run_metadata.json
features_train.pt, features_val.pt, features_test.pt
manifests/train.csv, val.csv, test.csv
```

ZIP còn chứa unit-test log, smoke log, full-train log và snapshot code. `run_metadata.json` ghi checksum manifest/source, seed, split mapping, preprocessing, threshold, limitations và thời gian chạy. Khi upload bằng ZIP không có `.git`, source checksum thay cho git commit.

Dataset generator-disjoint nhưng chưa identity-disjoint; real chỉ từ imdb_wiki. Full train thành công cũng chưa chứng minh hiệu quả forensic: phải xem unseen-generator metric và shortcut diagnostic của thí nghiệm chung.

## 5. Fusion sau khi nhận baseline

Cách tiện nhất khi đã hoàn thành notebook 11: chạy riêng **`notebooks/12_aiface_phase_e_fusion.ipynb`**.

1. Tạo một dataset **private** chứa ZIP kết quả `facetrace_spectral_artifacts.zip` vừa tải. Có thể để Kaggle giải nén ZIP hoặc giữ ZIP; runner hỗ trợ cả hai layout. Đây là dataset artifact, không phải dataset code hay dataset 32k ảnh.
2. Import notebook 12 vào một Kaggle notebook mới. Add Input dataset artifact vừa tạo.
3. Add Input **Output của Saved Version full-run** notebook Global/Local của bạn kia. Nếu không truy cập được Output, nhờ bạn kia ZIP **toàn bộ** `outputs/aiface_phase_e` rồi upload thành dataset private thứ hai và attach. Output cần có `feature_cache.pt` và checkpoint Global/Local, hoặc CSV validation tương ứng; chỉ tải metrics/CSV test thì chưa đủ.
4. Settings: Internet bật, Accelerator **None / CPU**. Không cần attach dataset ảnh, HF_TOKEN hay train lại.
5. Giữ `FORENSIC_INPUT=None`, `BASELINE_INPUT=None` để tự tìm artifact trong `/kaggle/input`. Nếu attach nhiều full run, điền đường dẫn chính xác tới directory hoặc ZIP tương ứng, ví dụ `FORENSIC_INPUT='/kaggle/input/<artifact-dataset>/facetrace_spectral_artifacts.zip'`. Cell đầu in các Input thực tế; không dùng đường dẫn `D:/...` trên Kaggle.
6. Run All. Notebook kiểm tra alignment/protocol, tạo hai fusion, hiển thị bảng so sánh sáu model và rescue/harm. Sau khi kiểm tra, Save Version → Save & Run All rồi tải `facetrace_fusion_artifacts.zip`. Giữ URL saved version và version của cả hai Input.

Notebook 12 chỉ chọn threshold trên validation; không học lại MLP hay tối ưu trọng số fusion trên test. `comparison.csv` và `comparison_by_generator.csv` ở `outputs/aiface_phase_e_fusion/`. Artifact fusion ghi environment CPU hiện tại, checksum source/metadata đầu vào; không dùng environment GPU của lần train forensic làm environment của fusion. Fusion không được đảm bảo cải thiện mọi metric.

Nhờ bạn kia gửi **artifact AI-Face mới, seed 42, cùng version dataset**, không dùng notebook 01–08 / Who Is AI. Cần:

```text
run_metadata.json
feature_cache.pt
global_only/checkpoint.pt
global_only/predictions_test.csv
global_only/val_metrics.json
global_only/test_metrics_overall.json
global_only/test_metrics_by_generator.csv
local_only/checkpoint.pt
local_only/predictions_test.csv
local_only/val_metrics.json
local_only/test_metrics_overall.json
local_only/test_metrics_by_generator.csv
global_local/predictions_test.csv
global_local/val_metrics.json
global_local/test_metrics_overall.json
global_local/test_metrics_by_generator.csv
```

Nếu có `global_only/predictions_val.csv` và `local_only/predictions_val.csv` thì không cần feature_cache/checkpoint để khôi phục validation prediction. Runner baseline hiện tại không xuất hai CSV validation này, nên hãy yêu cầu toàn bộ artifact có feature cache và checkpoint.

Giải nén baseline vào một thư mục đọc được trên Kaggle và gán `BASELINE_DIR` tới thư mục gốc có `global_only/`, `local_only/`, `global_local/` cùng `run_metadata.json`. Chạy cell optional fusion sau full forensic run. Hoặc gọi:

```bash
python scripts/fuse_aiface_spectral.py \
  --forensic-dir /kaggle/working/outputs/aiface_phase_e/forensic_only \
  --baseline-dir /path/to/aiface-baseline \
  --output-dir /kaggle/working/outputs/aiface_phase_e
```

Script kiểm tra checksum manifest, generator split, seed, image size, các ID/label/generator/split và xác suất. Nó khôi phục validation prediction từ feature cache/checkpoint nếu thiếu; chọn threshold fusion trên validation rồi đánh giá test. Fusion ba nhánh là `(p_global + p_local + p_forensic)/3`.

Hai output là `local_forensic/` và `global_local_forensic/`. Mỗi output có prediction, metric per-generator/macro, artifact và rescue/harm. Rescue/harm chính dùng threshold validation đã khóa của từng model, đúng với metric chính; thêm `rescue_harm_fixed_0.5.csv` cho so sánh dùng cùng threshold 0.5. Real test pool được dùng chung cho mỗi generator theo policy metric của baseline.

Bạn có thể fusion ở local sau khi tải artifact, không cần train lại hay dùng GPU. Forensic-only hoàn tất phần độc lập; task đầy đủ còn hai fusion và rescue/harm khi baseline đến.

## 6. Lệnh local

```bash
python -m unittest discover -s tests -p test_spectral.py -v
python scripts/train_spectral.py --data-root /path/to/aiface_pilot_32k --smoke
python scripts/train_spectral.py --data-root /path/to/aiface_pilot_32k
python scripts/predict_spectral.py --checkpoint /path/to/checkpoint.pt --manifest /path/to/test/manifest.csv --image-path-base /path/to/test --output-dir outputs/spectral_predictions
```

Hướng dẫn chính thức về Input, Accelerator và Save & Run All: https://www.kaggle.com/docs/notebooks
