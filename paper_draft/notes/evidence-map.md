# Bản đồ nguồn và phạm vi kết luận

## Quyết định đã chốt

Báo cáo bằng tiếng Việt, nộp cuối kỳ AIO 2026, không giới hạn trang/từ. Tác giả: Đinh Bảo Bảo, Nguyễn Thị Khánh Linh, Trần Tất Nguyên. Phạm vi nhãn: “ảnh khuôn mặt AI theo nhãn của dataset”.

## Nguồn dữ kiện

| Nội dung | Nguồn trong repo | Ghi chú |
|---|---|---|
| Research question, phương pháp, diễn giải và giới hạn | docs/aiface-end-to-end-report.md | Kết luận chỉ trong fixed pilot |
| Dataset, audit/identity hints | docs/aiface-phase-abc-report.md; end-to-end report | Hash không chứng minh identity/near-duplicate disjoint |
| Clean/robustness/CI/rescue-harm | docs/aiface-complete-results.md | Dùng số canonical Phase F khi report lịch sử khác chữ số cuối |
| Thông số mặc định và thuật toán | configs/aiface_phase_e.yaml; configs/spectral.yaml; src/models/dinov3.py; src/models/spectral.py | Resolved config của run ưu tiên hơn mặc định |
| Clean theo seed | Bảng Phase F clean trong complete-results | 18 dòng clean macro |
| Clean mean theo generator | Bảng per-generator Phase F clean | Tính mean từ 36 dòng (3 seed × 6 cấu hình × 2 nguồn), số hiển thị sáu chữ số |
| Related work | references.bib, nguồn paper/proceedings gốc | Không so sánh trực tiếp metric giữa benchmark khác |

Những đường dẫn docs/configs/src trên là tương đối với root repo FaceTrace.

## Mức kiểm chứng của bản thảo

Các báo cáo ghi artifact đã được kiểm tra ở giai đoạn nghiên cứu. Checkout hiện không có đủ prediction CSV/checkpoint Phase E/F. Khi viết lần này, số liệu được đọc và tổng hợp từ báo cáo; chưa chạy lại model hoặc tái tính từ prediction gốc. Hình robustness dùng cùng mean trong bảng, không có CI hoặc error bars.

Mỗi source test dùng chung 4.000 real; seed thay đổi lần huấn luyện trên cùng manifest. Bootstrap là image-level trên checkpoint/test cố định. Không mô tả nghiên cứu là kiểm chứng nhiều dataset, identity-disjoint, LOGO nhiều fold, blind prospective evaluation hoặc generator-population CI.

## Cần thông tin tác giả

Xem notes/author-confirmations.md. Hướng dẫn venue, giới hạn từ, tên tác giả và lựa chọn thuật ngữ không còn là mục cần xác nhận.
