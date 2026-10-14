# FaceTrace — Bản thảo báo cáo cuối kỳ AIO 2026

Bản thảo bằng tiếng Việt, dùng style/logo từ AIO2026_Template. Không áp dụng giới hạn trang hoặc từ của venue. File gốc là main.tex; ba tác giả là Đinh Bảo Bảo, Nguyễn Thị Khánh Linh và Trần Tất Nguyên.

## Nội dung đã viết

Abstract, giới thiệu/động lực, nghiên cứu liên quan, bài toán/dữ liệu, phương pháp, protocol, kết quả, thảo luận, giới hạn và kết luận đã có nội dung. Thuật ngữ phạm vi là “ảnh khuôn mặt AI theo nhãn của dataset”; lớp AI không mặc định chỉ gồm ảnh sinh hoàn toàn từ đầu.

Các bảng bao gồm clean mean/SD, paired CI, robustness AUROC/BAcc, rescue/harm, shortcut, clean theo seed và clean theo generator. Sơ đồ phương pháp dùng TikZ, biểu đồ robustness dùng PGFPlots và được vẽ trực tiếp khi biên dịch. Không cần file hình ngoài cho hai hình này.

## Làm việc trong Prism

Nhập toàn bộ thư mục paper_draft làm project và dùng main.tex làm file gốc. Các file LaTeX, thư viện style, bibliography và logo nằm trong thư mục này. Chỉnh từng phần trong sections; các bảng nằm ở tables; hình ở figures; phụ lục ở appendices.

Để tìm phần cần xác nhận, tìm lệnh todo trong các file tex. TODO được in đỏ trong bản PDF. notes/author-confirmations.md liệt kê các nhóm thông tin còn cần tác giả cung cấp.

## Cơ sở số liệu

Bản thảo dùng docs/aiface-complete-results.md và docs/aiface-end-to-end-report.md của repo làm nguồn số liệu. Bảng generator mean được tính từ các số sáu chữ số hiển thị trong báo cáo; sơ đồ/biểu đồ không bổ sung lần chạy mới. notes/evidence-map.md ghi rõ phạm vi này.

Các bảng đã được đối chiếu với báo cáo tổng hợp Phase F. Hai ZIP seed 42 cũ ở thư mục cha của repo cũng đã được kiểm tra từ prediction gốc. Các ZIP này khác nhẹ số Phase F và không thay thế bộ artifact cuối của ba seed. Chi tiết kiểm tra và nguồn của từng phần được ghi trong notes/evidence-map.md.

Đã bổ sung AI-Face v2 và annotation, version gói pilot, điều kiện sử dụng dữ liệu, đóng góp theo lịch sử commit, môi trường seed 42, các mốc hồ sơ và liên kết mã/notebook đã được ghi. Bốn TODO còn lại cần file môi trường các run khác, liên kết/version artifact cuối, đối chiếu prediction Phase F và thông tin lớp/giảng viên nếu được yêu cầu. Không điền thời gian chạy, bộ nhớ hay mốc khóa phương pháp khi chưa có hồ sơ.

## Biên dịch

Template dùng pdfLaTeX và biblatex với backend BibTeX. Khi build local, chạy pdfLaTeX, BibTeX, rồi pdfLaTeX hai lần; nếu dùng latexmk có thể để công cụ điều phối các lượt. Các package tiếng Việt và package đồ họa của template cần có trong TeX distribution.

Lần rà ngày 14/10/2026 dùng TeX Live 2026 để biên dịch bằng pdfLaTeX và BibTeX. Đã sửa font T5 cho tiếng Việt trong phần tài liệu tham khảo và header, cùng chỗ ngắt dòng tên generator dài. Chạy từ paper_draft:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error "-outdir=../outputs/paper_draft" main.tex
```

PDF được tạo ở outputs/paper_draft/main.pdf, ngoài thư mục nguồn và không được Git track. Bản này vẫn hiển thị bốn TODO cần hồ sơ nhóm, chưa phải bản đã xóa hết đánh dấu để nộp.
