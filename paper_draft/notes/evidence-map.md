# Bản đồ nguồn và phạm vi kết luận

Sau lần chỉnh lời văn, báo cáo không còn phần Thông tin nhóm và khả năng tái lập. Một TODO được giữ ở appendices/protocol.tex để thành viên có artifact Phase F đối chiếu số liệu cuối. Các ghi chép nguồn bên dưới được giữ làm hồ sơ nội bộ, không phải toàn bộ nội dung xuất hiện trong PDF.

## Quyết định đã chốt

Báo cáo bằng tiếng Việt, nộp cuối kỳ AIO 2026, không giới hạn trang/từ. Tác giả: Đinh Bảo Bảo, Nguyễn Thị Khánh Linh, Trần Tất Nguyên. Phạm vi nhãn là ảnh khuôn mặt AI theo nhãn của dataset.

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
| AI-Face v2 và annotation | docs/aiface-audit.md; docs/aiface-phase-abc-report.md | Ghi rõ tên hai CSV và archive đã audit, chưa có hash annotation gốc |
| Dataset pilot version 1 | forensic_only/run_metadata.json trong ZIP spectral seed 42 | Không nhầm version gói Kaggle với AI-Face v2 |
| Thiết bị và thư viện seed 42 | environment.json của forensic và fusion trong hai ZIP local; baseline_run_metadata.json trong ZIP fusion | Forensic Tesla T4/CUDA, fusion CPU, baseline CUDA. Chưa có môi trường đầy đủ các run khác |
| Backend Phase F | docs/aiface-complete-results.md, mục 2.1 | MPS cho seed 42/43, CUDA cho seed 44. Đây là backend suy luận |
| Link mã và run đã được ghi | git remote origin; docs/aiface-phase-e-report.md mục 7.3; docs/aiface-async-status.md | Baseline seed 44 version 6, fusion seed 42 version 1 theo hồ sơ repo. Không gán phiên bản này cho ZIP cũ |
| Điều kiện dữ liệu | Repo chính thức AI-Face và trang IMDB-WIKI, truy cập 14/10/2026 | Thêm citation riêng, không suy ra quyền tái phân phối của nhóm |

Những đường dẫn docs/configs/src trên là tương đối với root repo FaceTrace.

## Mức kiểm chứng của bản thảo

Các bảng clean mean/SD, clean theo seed, robustness AUROC/BAcc và paired clean AUROC CI đã được đối chiếu với docs/aiface-complete-results.md. Mean theo generator được tính lại từ các dòng trong báo cáo. Các con số khớp ở độ chính xác hiển thị, với sai số tối đa 0.000001 khi lấy mean từ số đã làm tròn. Hình robustness dùng cùng mean trong bảng, không có CI hoặc error bars.

Checkout hiện không có đầy đủ prediction/checkpoint Phase F. Hai ZIP local ở thư mục cha của repo chỉ xác minh được bộ seed 42 cũ, không phải toàn bộ kết quả cuối. Không có lần train hoặc inference mới trong lần chỉnh bản thảo này.

### Hai ZIP đã kiểm tra trực tiếp

| Gói local | SHA-256 | Phạm vi |
|---|---|---|
| facetrace_spectral_artifacts.zip | 62d02ff5b202b5679a60bbf7fb068e5d01a198a62d1cb5dd83fc4527b726f93b | Forensic seed 42, source commit 187374c, timestamp metadata 2026-10-07T16:05:08.878650+00:00 |
| facetrace_fusion_artifacts.zip | 4b399ca468566a62adb180a55af805e69fd58f9a05ec5248d299e3482810eca0 | Fusion seed 42, source commit eda43be, timestamp metadata 2026-10-07T16:52:21.627061+00:00 |

Tính lại AUROC/BAcc/Accuracy/F1 overall và theo generator của sáu cấu hình từ prediction test khớp JSON/CSV trong ZIP. ID, nhãn, generator và split trùng giữa sáu cấu hình. Ba công thức fusion khớp mean xác suất của đúng các nhánh. Rescue/harm overall và theo generator của hai fusion khớp CSV. Mỗi fusion có 8.000 validation ID riêng và không giao với test, AUROC validation khớp JSON.

Forensic seed 42 trong ZIP cũ đạt AUROC 0.958515375, BAcc 0.898750 và threshold 0.4730623960494995. Bảng canonical Phase F ghi 0.958504, 0.898875 và 0.472834. Local + Forensic trong ZIP cũ có AUROC 0.9818959375, còn Phase F ghi 0.981894. Giữ số canonical trong bảng cuối, không thay bằng số ZIP cũ hoặc tuyên bố hai bộ hoàn toàn giống nhau.

SHA-256 ZIP fusion local khác SHA-256 8b62cec... được ghi ở mục 7.3 báo cáo Phase E. Metadata local cũng ghi commit eda43be, trong khi hồ sơ phiên bản Kaggle về sau ghi 0d67f1a. Không ghép nguồn commit, notebook version hoặc threshold của hai gói này. Chưa đủ artifact Phase F để xác định toàn bộ nguyên nhân chênh lệch.

### Đóng góp và chronology

Commit f717877 của nguyentrann0703 thêm subset, audit và smoke. Commit 82a9a48 thêm baseline Global/Local. Các commit effd168, 63cf427 và 9f5893d thêm Phase F, khôi phục threshold từ validation và hoàn tất uncertainty/reporting. Profile GitHub nguyentrann0703 ghi tên Tran Tat Nguyen.

Các commit 5910a4e, 7085064, eda43be và 2b0fa33 của khanhlinh-jnf thêm forensic, notebook fusion và đối chiếu báo cáo Phase E. Commit f82a116 của AIVIETNAM-AIO-DinhBao thêm bản thảo, bảng và hình LaTeX. Các mốc được giữ để đối chiếu công việc, không suy tên người vận hành mọi run từ người viết mã.

Timestamp metadata seed 42 được chuyển sang UTC+7 là 23:05 và 23:52 ngày 07/10. Báo cáo đủ sáu cấu hình được commit 00:09 ngày 08/10. Các cập nhật Phase F là ngày 13/10. Đây là mốc hồ sơ, không phải timestamp người dùng mở test hoặc bằng chứng đăng ký trước phương pháp.

### Nguồn bên ngoài đã kiểm tra

- [AI-Face, README chính thức](https://github.com/Purdue-M2/AI-Face-FairnessBench/blob/main/README.md): liên kết ảnh/annotation, mục Update Notes cho v2 và giấy phép CC BY-NC-ND 4.0.
- [IMDB-WIKI, trang ETH Zurich](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/): mục License ghi mục đích nghiên cứu học thuật và chủ sở hữu bản quyền ảnh.
- [Profile nguyentrann0703](https://github.com/nguyentrann0703): đối chiếu tên tài khoản với Trần Tất Nguyên.

Không truy cập được trang Kaggle pilot bằng công cụ web trong lần kiểm tra này. Version 1 lấy từ metadata ZIP, không từ phiên bản mới nhất trên Kaggle.

Mỗi source test dùng chung 4.000 real; seed thay đổi lần huấn luyện trên cùng manifest. Bootstrap là image-level trên checkpoint/test cố định. Không mô tả nghiên cứu là kiểm chứng nhiều dataset, identity-disjoint, LOGO nhiều fold, blind prospective evaluation hoặc generator-population CI.

## Cần thông tin tác giả

Xem notes/author-confirmations.md. Hướng dẫn venue, giới hạn từ, tên tác giả và lựa chọn thuật ngữ không còn là mục cần xác nhận.
