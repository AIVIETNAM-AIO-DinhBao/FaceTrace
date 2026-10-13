# AI-Face Method Design — Forensic Branch and Fusion

This document records the design decisions for the AI-Face pilot. It is the canonical method-design reference; the execution protocol lives in `aiface-pilot-plan.md`, results live in `aiface-phase-e-report.md`, and Phase F extensions live in `aiface-phase-f-plan.md`.

## Part I — Forensic branch candidates


## **1\. Research question**

**Các đặc trưng forensic/low-level có cung cấp thông tin bổ sung cho DINOv3 khi gặp unseen generator hay không?**

Bài toán chính sẽ được đánh giá theo **Leave-One-Generator-Out (LOGO)** hoặc source-disjoint protocol. Vì vậy, tiêu chí chọn forensic branch không chỉ là accuracy trên pilot hiện tại mà còn gồm:

* khả năng generalize sang unseen generator;  
* mức độ giữ được forensic/low-level information;  
* compute/VRAM;  
* mức độ phụ thuộc vào domain/checkpoint gốc;  
* độ phức tạp khi triển khai và đánh giá độc lập.

Ba candidate hiện tại:

1. Radial FFT \+ MLP  
2. F3-Net  
3. AIDE PFE

---

# **2\. Radial FFT \+ MLP**

**Pipeline**

RGB  
→ FFT2 \+ fftshift  
→ log magnitude / power spectrum  
→ radial averaging  
→ 1D spectral profile  
→ small MLP  
→ real / AI-generated

Ý tưởng dựa trên các nghiên cứu cho thấy ảnh sinh có thể để lại bất thường trong miền tần số (Frank et al., 2020; Durall et al., 2020).

### **Ưu điểm**

* Rất nhẹ, chỉ cần một MLP nhỏ.  
* Train/inference nhanh.  
* Là **pure frequency branch**, dễ kiểm tra riêng giá trị của spectral signal.  
* Dễ ablation và ít nguy cơ model capacity quá lớn che mất bản chất của forensic cue.  
* Phù hợp làm baseline để xác định xem global spectral statistics có transferable signal hay không.

### **Vướng**

* Magnitude/power spectrum làm mất **phase information**.  
* Radial averaging làm mất **orientation** và gần như toàn bộ **spatial location**.  
* Representation bị nén mạnh:

2D image  
→ 2D spectrum  
→ 1D radial profile

* Có thể quá đơn giản nếu artifact của generator mới mang tính cục bộ hoặc không tạo spectral pattern toàn cục rõ ràng.  
* Cần kiểm soát preprocessing như resize, windowing, DC component và normalization để tránh spectral shortcut.

### **Vai trò**

**Lightweight frequency-only baseline.**

> Lưu ý: `Radial FFT + MLP` là baseline của nhóm lấy cảm hứng từ các nghiên cứu frequency-domain trước đó, không phải reproduction trực tiếp của Frank et al. hay Durall et al.

---

# **3\. F3-Net**

**Nguồn:** Qian et al., *Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues*, ECCV 2020\.

## **Pipeline**

                   ┌→ FAD → Xception ─┐  
RGB → DCT-aware ────┤                  ├→ fusion → classifier  
                    └→ LFS → Xception ─┘

### **FAD — Frequency-aware Decomposition**

DCT được sử dụng để phân tách các thành phần theo vùng tần số. Các thành phần sau đó được IDCT trở lại dạng spatial representation và đưa vào CNN.

### **LFS — Local Frequency Statistics**

Tính thống kê DCT trên các vùng/cửa sổ cục bộ để khai thác forensic pattern ở mức local frequency.

Hai stream được kết hợp thông qua cơ chế collaborative/fusion của F3-Net.

### **Ưu điểm**

* Giữ được **frequency \+ spatial/local information** tốt hơn Radial FFT.  
* Frequency representation không bị collapse thành vector 1D.  
* Có khả năng học các pattern forensic phức tạp hơn.  
* Kết hợp cả global frequency decomposition và local frequency statistics.

### **Vướng**

* Hai Xception backbone \+ frequency modules → **compute và VRAM cao**.  
* Vẫn cần task-train/fine-tune hai stream cho dataset của nhóm.  
* Không phải forensic encoder pretrained có thể đơn giản freeze rồi sử dụng trực tiếp.  
* Paper gốc tập trung vào **face forgery/deepfake**, trong khi bài toán hiện tại là general objects/scenes.  
* Do đó có **domain mismatch** giữa setting gốc và bài toán AI-generated images tổng quát.

### **Vai trò**

**Frequency-native deep model**, dùng để kiểm tra xem richer frequency–spatial representation có generalize tốt hơn global spectral statistics hay không.

---

# **4\. AIDE PFE**

**Nguồn:** Yan et al., *A Sanity Check for AI-generated Image Detection*, ICLR 2025\.

AIDE đầy đủ gồm semantic branch dùng OpenCLIP và **Patchwise Feature Extraction (PFE)** để khai thác low-level forensic clues.

Trong bài toán hiện tại, candidate được xem xét là **AIDE PFE**, không phải toàn bộ AIDE.

## **Pipeline PFE**

RGB  
→ 32×32 patches  
→ DCT scoring  
→ select low/high-frequency-score patches  
→ SRM filters  
→ ResNet50-low / ResNet50-high  
→ aggregate  
→ 2048-D forensic feature

Trong official implementation, các patch có frequency score cực thấp/cực cao được chọn. Sau đó:

selected patches  
→ 30 SRM filters  
→ two ResNet-50 branches  
→ forensic representation

DCT ở đây chủ yếu đóng vai trò **patch selection**. Feature forensic chính được học từ **SRM residuals \+ ResNet-50**.

### **Ưu điểm**

* Được thiết kế trực tiếp cho **general AI-generated image detection**.  
* Khai thác local noise/artifact thay vì chỉ global spectrum.  
* Có official implementation và pretrained checkpoint.  
* Có thể tận dụng pretrained forensic representation thay vì train toàn bộ từ đầu.  
* Gần với setting unseen-generator hiện tại hơn F3-Net về domain bài toán.

### **Vướng**

* Hai ResNet-50 → compute/VRAM vẫn tương đối cao.  
* PFE trong AIDE gốc được tối ưu **cùng OpenCLIP semantic branch**, không phải như một detector độc lập.  
* Hai ResNet PFE **không có classifier real/fake riêng**.

Nếu muốn đánh giá PFE như một forensic detector độc lập:

PFE  
→ 2048-D forensic feature  
→ new Linear/MLP classifier  
→ train

* Performance của **full AIDE không thể mặc định xem là performance của riêng PFE**, vì OpenCLIP cũng tham gia vào quyết định cuối.  
* Cần kiểm tra generator overlap giữa pretrained checkpoint và unseen-generator benchmark để tránh leakage hoặc đánh giá không công bằng.

### **Vai trò**

**Pretrained/heavy forensic candidate** dành cho general AI-generated images, nhưng cần adaptation khi tách PFE khỏi kiến trúc AIDE đầy đủ.

---

# **5\. So sánh nhanh**

| Tiêu chí | Radial FFT \+ MLP | F3-Net | AIDE PFE |
| ----- | ----- | ----- | ----- |
| Forensic signal | Global spectrum | Frequency \+ spatial | Local noise/artifact |
| Spatial information | Rất ít | Cao | Local selected patches |
| Frequency role | Feature trực tiếp | Representation chính | Chủ yếu patch selection |
| Backbone | MLP nhỏ | 2× Xception | 2× ResNet-50 |
| Compute | **Thấp** | **Cao** | **Cao** |
| Pretrained forensic representation | Không | Hạn chế | **Có** |
| Domain gốc | General frequency analysis | Face forgery | General AI-generated images |
| Dùng độc lập | Dễ | Có classifier sau training | Cần thêm classifier head |
| Rủi ro chính | Representation quá đơn giản | Compute \+ domain mismatch | Dependence on original AIDE training setup |

---

# **6\. Cách đánh giá các candidate**

Pilot 2.000 ảnh hiện tại chỉ nên được dùng để:

* kiểm tra implementation;  
* kiểm tra convergence;  
* ước lượng compute/VRAM;  
* phát hiện các vấn đề preprocessing;  
* xác nhận branch có học được signal cơ bản hay không.

**Không nên chọn forensic architecture cuối cùng chỉ dựa trên pilot này.**

Việc chọn branch nên dựa trên benchmark có generator/source labels theo LOGO.

Với mỗi held-out generator `G`:

Train:  
all generators except G

Test:  
generator G

Trước tiên cần đánh giá:

Forensic-only

để trả lời:

> Forensic representation này có tự generalize sang unseen generator hay không?

Sau đó mới đánh giá contribution của forensic branch khi kết hợp với DINOv3.

---

# **7\. Điểm cần thảo luận**

Ba candidate đại diện cho ba mức độ forensic modeling khác nhau:

Radial FFT \+ MLP  
↓  
global spectral statistics  
↓  
nhẹ nhất / mất nhiều thông tin nhất

F3-Net  
↓  
frequency \+ spatial/local structure  
↓  
representation giàu hơn nhưng compute lớn và có face-domain bias

AIDE PFE  
↓  
patchwise SRM/noise forensic representation  
↓  
general AI-image domain \+ pretrained,  
nhưng architecture gốc phụ thuộc semantic branch

**Câu hỏi cần chốt ở bước này:**

> Với mục tiêu unseen-generator generalization, candidate nào hợp lý nhất để đưa vào main LOGO benchmark với vai trò forensic branch?

Hiện tại chưa cần chốt cách kết hợp forensic branch với DINOv3. **Late probability fusion vs latent feature fusion sẽ được xem xét riêng sau khi forensic architecture đã được chọn hoặc shortlist.**

---

## **References**

* Frank et al., **Leveraging Frequency Analysis for Deep Fake Image Recognition**, ICML 2020\.  
* Durall et al., **Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions**, CVPR 2020\.  
* Qian et al., **Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues**, ECCV 2020\.  
* Yan et al., **A Sanity Check for AI-generated Image Detection**, ICLR 2025\.



## Part II — Fusion strategy


## **1\. Câu hỏi cần chốt**

Sau khi đã có ba nguồn thông tin:

Global   \= DINOv3 CLS/global feature  
Local    \= DINOv3 patch/local feature  
Forensic \= low-level forensic representation

nên kết hợp chúng bằng:

1. **Late probability/logit fusion**  
2. **Latent feature fusion**

Mục tiêu không chỉ là đạt accuracy cao nhất, mà còn phải trả lời được:

> **Forensic branch có thực sự cung cấp thông tin bổ sung cho DINOv3 trên unseen generator hay không?**

---

# **2\. Late Fusion**

## **Pipeline**

Mỗi branch được train như một detector độc lập:

Global   → classifier → p\_G  
Local    → classifier → p\_L  
Forensic → classifier → p\_F

Sau đó mới fusion:

G \+ L  
L \+ F  
G \+ L \+ F

Ví dụ đơn giản:

p\_GL  \= (p\_G \+ p\_L) / 2  
p\_LF  \= (p\_L \+ p\_F) / 2  
p\_GLF \= (p\_G \+ p\_L \+ p\_F) / 3

Có thể thử weighted fusion sau nếu cần.

## **Ưu điểm**

* Mỗi branch phải chứng minh khả năng phân loại **độc lập**.  
* Dễ đo contribution của từng branch.  
* Dễ làm ablation:

G  
L  
F  
G \+ L  
L \+ F  
G \+ L \+ F

* Nếu `DINO + F > DINO`, dễ diễn giải rằng forensic branch cung cấp complementary signal.  
* Ít phụ thuộc vào việc latent feature của các backbone có tương thích hay không.

## **Vướng**

* Mỗi branch cần classifier riêng.  
* Probability của các branch có thể có calibration khác nhau.  
* Average probability 1:1 có thể chưa tối ưu nếu một branch mạnh/yếu hơn rõ rệt.

### **Implementation note**

Có thể cân nhắc fusion **logits** thay vì raw probabilities nếu calibration giữa các branch khác nhau quá nhiều.

---

# **3\. Latent Fusion**

## **Pipeline**

Không đưa từng branch ra probability trước mà nối feature:

f\_global ───┐  
f\_local  ───┼→ projection / normalization  
f\_forensic ─┘  
                 ↓  
               concat  
                 ↓  
          fusion classifier  
                 ↓  
             real / fake

Ví dụ:

f\_G → projection ─┐  
f\_L → projection ─┼→ concat → MLP → prediction  
f\_F → projection ─┘

## **Ưu điểm**

* Cho phép classifier học interaction trực tiếp giữa semantic và forensic features.  
* Có thể khai thác thông tin bổ sung ở mức representation, thay vì chỉ ở mức probability.  
* Có khả năng đạt performance cao hơn nếu các feature thực sự complementary.

## **Vướng**

* Các branch có thể có:  
  * dimension khác nhau;  
  * scale khác nhau;  
  * distribution khác nhau;  
  * semantic meaning khác nhau.

Do đó thường cần:

LayerNorm / normalization  
\+ projection layer  
\+ fusion head

* Khó attribution hơn.

Nếu performance tăng:

Latent(G \+ L \+ F) \> DINO

thì chưa chắc gain đến trực tiếp từ forensic signal, vì có thể đến từ:

* projection layer;  
* fusion classifier;  
* interaction giữa representations;  
* additional model capacity.

Do đó latent fusion khó trả lời research question về **additive forensic value** sạch bằng late fusion.

---

# **4\. Trường hợp riêng của AIDE PFE**

AIDE gốc:

OpenCLIP semantic feature ─┐  
                          ├→ concat → shared classifier  
PFE forensic feature ─────┘

PFE không có classifier real/fake riêng.

## **Nếu dùng Late Fusion**

Phải tạo:

AIDE PFE  
→ 2048-D forensic feature  
→ new Linear/MLP head  
→ p\_F

Head này được train trên training generators của nhóm.

Ưu điểm là sau đó PFE trở thành một forensic detector độc lập, nên có thể đánh giá:

F-only  
DINO-only  
DINO \+ F

## **Nếu dùng Latent Fusion**

Có thể lấy trực tiếp:

DINO feature \+ AIDE PFE feature

nhưng không nên concat thẳng mà không adaptation.

Nên có:

DINO feature  
→ normalization/projection ─┐

                              ├→ fusion classifier

PFE feature  
→ normalization/projection ─┘

Lý do: PFE checkpoint ban đầu được tối ưu trong context **PFE \+ OpenCLIP**, nên feature space của PFE không được đảm bảo sẵn là compatible với DINOv3.

Đây không có nghĩa latent fusion không làm được, mà chỉ có nghĩa nó cần **representation adaptation**.

---

# **5\. So sánh nhanh**

| Tiêu chí | Late Fusion | Latent Fusion |
| ----- | ----- | ----- |
| Mỗi branch độc lập | **Có** | Không bắt buộc |
| Đo additive value | **Dễ** | Khó hơn |
| Feature compatibility | Ít vấn đề | **Quan trọng** |
| Cần classifier riêng cho F | Có | Không nhất thiết |
| Cần projection/alignment | Không bắt buộc | **Thường cần** |
| Khả năng học interaction | Hạn chế | **Cao** |
| Interpretability | **Cao hơn** | Thấp hơn |
| Complexity | Thấp hơn | Cao hơn |
| Phù hợp ablation | **Rất tốt** | Trung bình |

---

# **6\. Experimental order đề xuất**

Không nhất thiết phải chọn một trong hai ngay từ đầu.

Có thể tách thành hai stage.

## **Stage 1 — Late Fusion**

Trước tiên train và đánh giá từng branch:

Global-only  
Local-only  
Forensic-only

Sau đó:

Global \+ Local  
Local \+ Forensic  
Global \+ Local \+ Forensic

Mục tiêu:

> Kiểm tra forensic branch có signal độc lập và có thực sự rescue/add value cho DINO trên unseen generator hay không.

---

## **Stage 2 — Latent Fusion**

Chỉ thực hiện sau khi forensic branch đã chứng minh có giá trị ở Stage 1\.

f\_global  
f\_local  
f\_forensic  
    ↓  
projection / normalization  
    ↓  
latent fusion  
    ↓  
classifier

Mục tiêu lúc này chuyển từ:

> “Forensic có bổ sung thông tin hay không?”

sang:

> “Khi đã biết forensic có giá trị, có thể khai thác interaction giữa các representation tốt hơn nữa hay không?”

---

# **7\. Điểm cần thảo luận**

Câu hỏi chính ở bước này là:

> **Với research question nhấn mạnh additive forensic value trên unseen generator, nên dùng late fusion làm main experiment và latent fusion làm supplementary experiment, hay có lý do đủ mạnh để dùng latent fusion ngay từ đầu?**

Đặc biệt với AIDE PFE cần cân nhắc:

Late fusion:  
PFE → new classifier head → p\_F

Latent fusion:  
PFE feature → projection  
             \+  
DINO feature → projection  
             → fusion classifier

Hai hướng đều cần train thêm module mới; khác biệt chính nằm ở **khả năng isolate contribution** và **mức độ interaction giữa các representations**.



## Locked Phase E choice

The main AI-Face pilot uses **Radial FFT + MLP** as the forensic branch and **equal-weight late probability fusion** as the primary integration protocol. F3-Net/AIDE-inspired alternatives remain design references, not reported Phase E models.

The forensic input is: RGB → fixed resize → Hann window → 2D FFT → log magnitude → radial averaging → MLP. The fusion outputs are: `mean(p_global, p_local)`, `mean(p_local, p_forensic)`, and `mean(p_global, p_local, p_forensic)`. Thresholds are selected on validation only.
