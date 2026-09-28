# FaceTrace

Nghiên cứu khả năng bổ sung giữa đặc trưng toàn ảnh, đặc trưng cục bộ và dấu vết forensic trong phát hiện khuôn mặt thật và khuôn mặt do AI tạo hoặc chỉnh sửa.

Trọng tâm đánh giá là **generalization sang nguồn/generator chưa xuất hiện trong huấn luyện**. Repo triển khai theo thứ tự từ baseline Frozen DINOv3 với đặc trưng global, đến local patch feature và image residual. Mỗi thành phần được đánh giá riêng; chỉ kết hợp khi có bằng chứng thực nghiệm cho thấy thông tin bổ sung.

## Cấu trúc

- `configs/`: cấu hình baseline và thí nghiệm local.
- `notebooks/`: audit dữ liệu, trích xuất feature và các thí nghiệm theo kế hoạch.
- `src/`: module dữ liệu, model, training, evaluation và tiện ích.
- `scripts/`: entry point chạy train/evaluate.
- `outputs/`: nơi lưu checkpoint, feature và kết quả (không commit dữ liệu sinh ra).
- `docs/week02.md`: mục tiêu và đầu ra tuần 2.

## Bắt đầu

1. Tạo môi trường Python 3.10+ và cài dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Cập nhật `data.root` trong config tương ứng để trỏ tới dữ liệu cục bộ. Dataset cần có nhãn real/fake và metadata nguồn/generator để tạo split source-disjoint. Định dạng manifest sẽ được thống nhất sau khi audit dữ liệu.

3. Khảo sát dữ liệu trong `notebooks/00_dataset_audit.ipynb`. Không dùng random split theo ảnh làm đánh giá chính nếu có thể xác định nguồn.

4. Chạy baseline sau khi dataset loader và manifest đã được cấu hình:

   ```bash
   python scripts/train_baseline.py --config configs/baseline.yaml
   ```

   Các script/module hiện là scaffold ban đầu; một số bước cần hoàn thiện sau khi chốt định dạng dữ liệu và checkpoint DINOv3 được phép truy cập.

## Thiết kế thí nghiệm

Các chỉ số chính: AUROC và Balanced Accuracy; Accuracy và F1 là chỉ số bổ sung. Báo cáo kết quả trên nguồn giữ lại khỏi train. Robustness với JPEG, resize, blur và crop là trục đánh giá riêng. Khi so sánh các nhánh, giữ nguyên split, seed và quy trình huấn luyện để quy kết thay đổi cho representation.

Thứ tự ưu tiên:

1. Audit dữ liệu và kiểm tra shortcut (nguồn, cân bằng lớp, resolution, nén, crop).
2. Frozen DINOv3 + global feature + linear/MLP classifier.
3. Global + local patch feature từ cùng backbone.
4. Residual branch độc lập; xem xét kết hợp khi có bằng chứng bổ sung.
5. LoRA/fine-tuning và robustness khi các thí nghiệm chính đã ổn định.

## Lưu ý dữ liệu và tái lập

Không đưa dữ liệu khuôn mặt, checkpoint, feature cache hoặc kết quả lớn vào Git. Ghi lại seed, danh sách nguồn train/validation/test, phiên bản dữ liệu, model ID/checkpoint và cấu hình cho mỗi lần chạy. Không dùng nguồn test để chọn threshold hay tinh chỉnh cấu hình.

