# Các thông tin còn cần nhóm xác nhận

Đã bổ sung thông tin có bằng chứng từ repo và hai ZIP seed 42 đang có trên local. Bản thảo còn bốn TODO đỏ:

| Thông tin | File |
|---|---|
| File môi trường của baseline seed 42, các lần train seed 43/44 và Phase F. Cần model revision DINOv3, transformers, thời gian chạy và bộ nhớ đo được | sections/experimental_setup.tex |
| Liên kết lưu artifact Phase F và version của các lần chạy chưa được ghi. URL notebook không xác định một lần chạy cụ thể | sections/disclosures.tex |
| Mã nhóm, lớp và giảng viên, chỉ khi hướng dẫn nộp yêu cầu | sections/disclosures.tex |
| Bộ prediction/checkpoint Phase F seed 42/43/44, output bootstrap và model selection để đối chiếu bảng cuối từ dữ liệu gốc | appendices/protocol.tex |

Những mục này cần hồ sơ của nhóm, không thể điền bằng cấu hình mặc định hoặc suy từ backend. G+F, LoRA và đánh giá nhiều dataset vẫn thuộc hướng phát triển, không phải thí nghiệm đã thực hiện.

Đóng góp thành viên đã được ghi theo lịch sử commit, không suy thêm vai trò ngoài repo. AI-Face v2 và annotation được xác định từ báo cáo audit. Dataset pilot trên Kaggle là version 1 theo metadata seed 42, khác với version của AI-Face gốc.

Nguồn chính thức đã được kiểm tra ngày 14/10/2026: repo AI-Face công bố CC BY-NC-ND 4.0, còn IMDB-WIKI giới hạn mục đích nghiên cứu học thuật. Hồ sơ chưa lưu hash annotation gốc hoặc xác nhận quyền tái phân phối pilot. Bản hiện tại không có ảnh mặt minh họa và không tuyên bố nhóm đã được cấp quyền phân phối.

Chronology được ghi theo timestamp metadata và ngày commit. Không có bằng chứng về thời điểm khóa toàn bộ phương pháp trước lần đầu xem test, nên bản thảo giữ cách diễn giải pilot khám phá.
