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

Checkout hiện chưa chứa toàn bộ prediction CSV/checkpoint Phase E/F. Việc viết bản thảo không đồng nghĩa đã tính lại mọi metric từ prediction gốc. Các chỗ cần bổ sung artifact và provenance được đánh dấu đỏ.

## Biên dịch

Template dùng pdfLaTeX và biblatex với backend BibTeX. Khi build local, chạy pdfLaTeX, BibTeX, rồi pdfLaTeX hai lần; nếu dùng latexmk có thể để công cụ điều phối các lượt. Các package tiếng Việt và package đồ họa của template cần có trong TeX distribution.

Môi trường soạn hiện chưa có pdflatex/latexmk/bibtex/tectonic nên chưa tạo PDF để kiểm tra dàn trang. Đường dẫn input, citation keys và dữ liệu bảng được kiểm tra ở mức source.
