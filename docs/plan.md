KẾ HOẠCH NGHIÊN CỨU

1. Mục tiêu chung

Nhóm lựa chọn hướng nghiên cứu bài toán phân biệt khuôn mặt thật và khuôn mặt do AI
tạo hoặc chỉnh sửa. Trọng tâm của nghiên cứu không chỉ là đạt accuracy cao trên tập dữ
liệu có cùng phân phối với tập train, mà là đánh giá khả năng generalization sang những
nguồn ảnh AI chưa từng xuất hiện trong quá trình huấn luyện.

Qua quá trình tổng hợp các nghiên cứu liên quan, nhóm nhận thấy đặc trưng toàn ảnh, đặc
trưng cục bộ và các dấu vết forensic mức thấp có thể phản ánh những khía cạnh khác nhau
của ảnh. Tuy nhiên, thay vì mặc định kết hợp tất cả các thành phần thành một model,
nghiên cứu sẽ kiểm tra lần lượt giá trị của từng loại representation và chỉ thực hiện kết hợp
khi có bằng chứng thực nghiệm cho thấy chúng bổ sung thông tin cho nhau.

DINOv3 được lựa chọn làm backbone chính vì có thể cung cấp đồng thời representation
toàn ảnh và representation ở mức patch từ cùng một model. Điều này giúp nhóm kiểm soát
tốt hơn các thí nghiệm khi so sánh giữa đặc trưng toàn ảnh và đặc trưng cục bộ. Bên cạnh
đó, nhóm lựa chọn image residual hoặc high-frequency residual làm representation mức
thấp ban đầu nhằm kiểm tra khả năng bổ sung các dấu vết về texture, noise và quá trình
hình thành ảnh mà DINOv3 có thể chưa biểu diễn đầy đủ.

Quy trình đánh giá chính của nghiên cứu là chia dữ liệu theo nguồn hoặc generator, trong
đó một hoặc nhiều nguồn được giữ hoàn toàn khỏi quá trình huấn luyện để phục vụ đánh
giá. Ngoài ra, nhóm sẽ kiểm tra độ bền của model trước các biến đổi như JPEG
compression, resize, blur và crop.

2. Tuần 1 - Xác định hướng nghiên cứu và xây dựng kế hoạch

Tuần 1 đã được nhóm hoàn thành ở mức cơ bản và chủ yếu tập trung vào việc tìm hiểu bài
toán, tổng hợp tài liệu và xác định hướng nghiên cứu.

Nhóm đã đọc và tổng hợp 9 paper liên quan đến phát hiện ảnh do AI tạo. Các nghiên cứu
bao phủ một số hướng chính như sử dụng model pretrained, khai thác representation cục
bộ, phân tích dấu vết tần số hoặc residual, hạn chế bias của dữ liệu và cải thiện khả năng
generalization sang generator chưa từng thấy.

Từ quá trình tổng hợp, nhóm xác định vấn đề chính cần nghiên cứu là khả năng
generalization dưới điều kiện thay đổi nguồn sinh. Một model có thể đạt kết quả tốt trên
validation thông thường nhưng vẫn phụ thuộc vào dấu vết riêng của generator, resolution,
chất lượng JPEG hoặc các đặc điểm của dataset. Vì vậy, đánh giá theo nguồn sinh chưa
từng thấy được lựa chọn làm quy trình đánh giá chính thay vì chỉ sử dụng random split theo
ảnh.

Câu hỏi nghiên cứu chính được điều chỉnh thành:

Các đặc trưng cục bộ và dấu vết forensic mức thấp có cung cấp thông tin bổ sung
cho đặc trưng toàn ảnh của DINOv3 trong việc generalization sang các nguồn khuôn
mặt AI chưa từng xuất hiện trong quá trình huấn luyện hay không?

Từ câu hỏi chính, nghiên cứu được chia thành các câu hỏi nhỏ: đặc trưng cục bộ có cải
thiện so với đặc trưng toàn ảnh hay không; residual có bổ sung thông tin ngoài
representation của DINOv3 hay không; việc kết hợp các thành phần có thực sự mang lại lợi
ích hay không; và việc giữ nguyên DINOv3 so với fine-tuning nhẹ ảnh hưởng như thế nào
trong điều kiện dữ liệu train nhỏ.

Kết quả của Tuần 1 bao gồm Paper Tracker, Research Outline, câu hỏi nghiên cứu, các giả
thuyết cần kiểm chứng và kế hoạch thực nghiệm cho ba tuần tiếp theo.

3. Tuần 2 - Khảo sát dữ liệu, xây dựng baseline và kiểm tra đặc trưng
cục bộ

Tuần 2 là giai đoạn bắt đầu triển khai thực nghiệm. Mục tiêu của tuần này là xây dựng một
baseline đủ đơn giản và đáng tin cậy, sau đó kiểm tra riêng giả thuyết về giá trị của đặc
trưng cục bộ.

Trước tiên, nhóm sẽ thực hiện khảo sát dữ liệu để xác định nguồn ảnh, phân bố hai lớp,
resolution, mức độ nén, cách crop khuôn mặt và các yếu tố có thể tạo shortcut. Nếu có
thông tin về nguồn hoặc generator, dữ liệu sẽ được chia theo hướng source-disjoint ngay từ
đầu để đảm bảo tập đánh giá chính chứa các nguồn chưa từng xuất hiện trong tập train.

Sau bước khảo sát, nhóm xây dựng baseline tối thiểu gồm Frozen DINOv3, đặc trưng toàn
ảnh và một Linear hoặc MLP classifier nhỏ. Model này được sử dụng làm mốc so sánh cho
toàn bộ nghiên cứu.

Khi baseline đã ổn định, nhóm bắt đầu kiểm tra giả thuyết về đặc trưng cục bộ. Local
representation được xác định cụ thể là các patch feature lấy từ cùng backbone DINOv3.
Nhóm sẽ chọn một phương pháp tổng hợp patch cố định để tránh thay đổi nhiều cơ chế
cùng lúc. Kết quả của phiên bản Global-only và Global + Local sẽ được so sánh trên cùng
tập đánh giá, đặc biệt là tập chứa nguồn sinh chưa từng thấy.

Các chỉ số chính trong giai đoạn này là AUROC và Balanced Accuracy. Accuracy và F1
được sử dụng bổ sung. Bên cạnh kết quả tổng thể, nhóm cần kiểm tra xem các cải thiện có
duy trì trên nguồn chưa từng thấy hay chỉ xuất hiện trên dữ liệu cùng phân phối với tập train.

Kết thúc Tuần 2, nhóm cần có một baseline DINOv3 hoàn chỉnh, một phiên bản sử dụng
thêm local representation và kết quả đủ rõ để trả lời câu hỏi: đặc trưng cục bộ có thực sự
cải thiện khả năng generalization so với đặc trưng toàn ảnh hay không?

4. Tuần 3 - Kiểm tra dấu vết mức thấp và khả năng bổ sung giữa các
representation

Tuần 3 tiếp tục phát triển từ kết quả của Tuần 2. Mục tiêu chính của tuần này là kiểm tra giả
thuyết về dấu vết forensic mức thấp và xác định liệu loại thông tin này có thực sự bổ sung
cho representation của DINOv3 hay không.

Nhóm ưu tiên sử dụng image residual hoặc high-frequency residual làm representation mức
thấp chính. Việc lựa chọn một hướng duy nhất giúp thí nghiệm dễ kiểm soát hơn so với việc
đồng thời thử residual, FFT, DCT và nhiều loại biến đổi khác.

Trước hết, residual branch sẽ được đánh giá độc lập để xác định khả năng phân biệt
Real/Fake của loại thông tin này. Sau đó, kết quả của residual branch được so sánh với
DINOv3 không chỉ thông qua accuracy mà còn thông qua các trường hợp dự đoán sai. Nếu
residual branch có thể nhận đúng một phần đáng kể các mẫu mà DINOv3 dự đoán sai, đây
là bằng chứng cho thấy hai representation có tính bổ sung.

Tiếp theo, nhóm sẽ thử các tổ hợp đơn giản như Global + Residual và Global + Local +
Residual. Tuy nhiên, full fusion không được mặc định là model cuối cùng. Thành phần nào
không có bằng chứng cải thiện rõ ràng trên tập đánh giá nguồn mới sẽ không được giữ chỉ
vì làm kiến trúc phức tạp hơn.

Nếu các nhánh cho thấy tính bổ sung, nhóm có thể bắt đầu với cách kết hợp đơn giản như
trung bình xác suất hoặc weighted average. Các cơ chế fusion phức tạp hơn chỉ được xem
xét khi cách kết hợp đơn giản đã cho thấy xu hướng tích cực.

Sau khi cấu trúc representation đã được kiểm chứng, nhóm có thể tiến hành thêm thí
nghiệm so sánh Frozen DINOv3 với fine-tuning LoRA. Fine-tuning được xem là một yếu tố
thực nghiệm nhằm đánh giá sự đánh đổi giữa khả năng thích nghi với tập train và khả năng
generalization, không phải đóng góp chính của nghiên cứu. Full fine-tuning chỉ được thực
hiện nếu dữ liệu, thời gian và tài nguyên cho phép.

Kết thúc Tuần 3, nhóm cần trả lời được ba vấn đề chính: Local representation có đóng góp
hay không; residual có chứa thông tin bổ sung ngoài DINOv3 hay không; và khi kết hợp các
representation đã có bằng chứng, hiệu quả generalization thay đổi như thế nào.

Model được lựa chọn để bước sang Tuần 4 sẽ là model có bằng chứng thực nghiệm tốt
nhất, không nhất thiết phải là model chứa đầy đủ tất cả các thành phần.

5. Tuần 4 - Bổ sung thí nghiệm, kiểm chứng và hoàn thiện kết quả

Tuần 4 tập trung vào việc kiểm chứng và hoàn thiện kết quả thay vì mở thêm nhiều hướng
nghiên cứu mới.

Trước tiên, nhóm sẽ rà soát các thí nghiệm của Tuần 2 và Tuần 3 để xác định những kết
quả chưa đủ rõ hoặc cần chạy lại. Các cấu hình quan trọng có thể được thử với nhiều seed
để giảm khả năng kết luận dựa trên một lần huấn luyện duy nhất.

Một phần quan trọng của Tuần 4 là hoàn thiện các thí nghiệm so sánh thành phần. Tùy theo
kết quả của hai tuần trước, nhóm sẽ so sánh các cấu hình như Global-only, Global + Local,
Residual-only, Global + Residual, Global + Local và Global + Local + Residual. Mục đích của
việc so sánh này là xác định chính xác nguồn gốc của sự cải thiện thay vì chỉ so baseline
với model cuối cùng.

Nhóm cũng thực hiện trục đánh giá thứ hai về độ bền của model trước JPEG compression,
resize, blur và crop. Các thí nghiệm này giúp kiểm tra xem các representation forensic có
duy trì giá trị khi ảnh đã trải qua quá trình xử lý hay không.

Nếu thí nghiệm fine-tuning trong Tuần 3 chưa đủ hoàn chỉnh, Tuần 4 sẽ bổ sung so sánh
giữa Frozen DINOv3 và phương pháp fine-tuning tiết kiệm tham số. Full fine-tuning chỉ được
bổ sung khi điều kiện cho phép và không làm ảnh hưởng tới các thí nghiệm chính.

Ngoài các chỉ số định lượng, nhóm sẽ phân tích các trường hợp dự đoán sai và mức độ
trùng lặp lỗi giữa các nhánh. Phần phân tích này được sử dụng để giải thích rõ hơn local
representation hoặc residual thực sự bổ sung loại thông tin nào cho model.

Cuối tuần, nhóm tổng hợp kết quả thành báo cáo hoàn chỉnh, bao gồm bài toán nghiên cứu,
quy trình đánh giá, baseline, các thí nghiệm kiểm chứng từng giả thuyết, kết quả so sánh,
phân tích lỗi, hạn chế và hướng phát triển tiếp theo.

6. Kết quả dự kiến sau 4 tuần

Sau bốn tuần, nhóm dự kiến hoàn thành một quy trình thực nghiệm có khả năng kiểm tra
một cách rõ ràng vai trò của các mức representation khác nhau trong bài toán phát hiện
khuôn mặt do AI tạo.

Output cuối cùng bao gồm mã nguồn huấn luyện và inference, baseline DINOv3 có thể tái
lập, quy trình đánh giá theo nguồn sinh chưa từng thấy, kết quả so sánh giữa Global, Local
và Residual, các thí nghiệm về fine-tuning nếu điều kiện cho phép, kiểm tra độ bền trước
các biến đổi ảnh và phân tích các trường hợp model dự đoán sai.

Kết quả nghiên cứu không nhất thiết phải là một model kết hợp toàn bộ Global, Local và
Residual. Nếu thực nghiệm cho thấy một thành phần không mang lại giá trị bổ sung, nhóm
sẽ loại bỏ thành phần đó và xem đây là một kết quả của nghiên cứu.

Mục tiêu quan trọng nhất sau bốn tuần là có đủ bằng chứng để trả lời câu hỏi:

Đặc trưng cục bộ và dấu vết forensic mức thấp có thực sự cung cấp thông tin bổ
sung cho đặc trưng toàn ảnh của DINOv3 khi model phải nhận biết khuôn mặt AI từ
những nguồn sinh chưa từng thấy hay không?

Như vậy, Tuần 1 tập trung hình thành câu hỏi và giả thuyết; Tuần 2 xây dựng baseline và
kiểm tra local representation; Tuần 3 kiểm tra residual và tính bổ sung giữa các
representation; Tuần 4 tập trung kiểm chứng, bổ sung thí nghiệm và hoàn thiện báo cáo.
Cách triển khai này giúp đảm bảo mỗi thay đổi trong model đều được kiểm tra riêng trước
khi đưa vào cấu hình cuối cùng.

