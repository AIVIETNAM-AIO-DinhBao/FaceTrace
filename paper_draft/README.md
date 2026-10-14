# FaceTrace: Báo cáo bài cuối kỳ AIO2026 Module 4

Olympic AI - AI là AI.

Báo cáo bằng tiếng Việt, dùng style và logo từ AIO2026_Template. File gốc là main.tex. Các phần nội dung nằm trong sections, bảng ở tables, hình ở figures và phụ lục ở appendices.

## Nội dung

Báo cáo trình bày việc nhóm chọn AI-Face v2, dựng và kiểm tra pilot 32.000 ảnh, train Global/Local/Forensic, fusion, rồi đánh giá ba seed và bốn phép biến đổi ảnh. Lời văn đã được chỉnh theo cách kể công việc và kết quả trong daily report ngày 08/10/2026.

Đã bỏ phần Thông tin nhóm và khả năng tái lập. Báo cáo giữ một TODO ở appendices/protocol.tex để thành viên giữ artifact Phase F đối chiếu số liệu cuối.

## Số liệu

Bảng kết quả lấy từ docs/aiface-complete-results.md. Các bảng đã được đối chiếu với báo cáo này. Mean theo generator được tính từ số hiển thị sáu chữ số. Hai ZIP seed 42 cũ đã được kiểm tra riêng và có một số khác biệt nhỏ với Phase F. Chi tiết bằng chứng nằm trong notes/evidence-map.md.

## Biên dịch

Từ paper_draft, chạy:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error "-outdir=../outputs/paper_draft" main.tex
```

PDF được tạo ở outputs/paper_draft/main.pdf. Bản để đọc local được lưu thêm thành paper_draft/main.pdf.
