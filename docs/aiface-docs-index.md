# AI-Face Documentation Index

Đây là nhóm tài liệu đang được dùng cho nghiên cứu AI-Face pilot và Phase E.

## Tài liệu chính

- `aiface-pilot-plan.md`: kế hoạch dữ liệu, fixed generator split và Phase E/F.
- `aiface-method-design.md`: candidate forensic, Radial FFT + MLP và fusion policy đã khóa.
- `aiface-phase-f-plan.md`: protocol seed 42/43/44, robustness, bootstrap CI và model selection.
- `aiface-async-status.md`: trạng thái live của các run Kaggle và artifact đang chờ xác minh.
- `aiface-audit.md`: audit annotation và chất lượng dữ liệu AI-Face.
- `aiface-phase-abc-report.md`: báo cáo audit, subset và quality gate.
- `aiface-phase-e-report.md`: kết quả Phase E seed 42/43/44, sáu cấu hình và rescue/harm.
- `aiface-complete-results.md`: bảng số liệu đầy đủ Phase E/F, robustness, bootstrap CI và model selection để viết paper.
- `aiface-spectral-kaggle-guide.md`: hướng dẫn chạy forensic spectral trên Kaggle.

## Quy ước sử dụng

- Khi cập nhật protocol hoặc báo cáo hiện tại, chỉ tham chiếu các file `aiface-*.md`
  trong thư mục `docs/`.
- Tài liệu lịch sử của bài Olympic/Who Is AI và các bản forensic/fusion cũ không
  thuộc protocol hiện tại; không dùng các tài liệu `week*.md` hoặc task cũ làm
  nguồn phương pháp chính.
