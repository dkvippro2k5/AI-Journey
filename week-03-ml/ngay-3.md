# Ngày 3 · Học không giám sát, scikit-learn Pipeline và tuning

> **Loại tài liệu: Lý thuyết (bản nhẹ)** cho bài 08–09 · **Kỹ thuật** cho bài 10–11 (API, quy trình, quyết định thiết kế).
> **Hôm nay gồm 4 bài:**
> - **08** Phân cụm: K-Means và DBSCAN
> - **09** Trực quan hóa dữ liệu nhiều chiều: PCA, t-SNE, UMAP
> - **10** scikit-learn API và Pipeline
> - **11** Tuning hyperparameter
>
> **Thời gian:** đọc ~2h45 · quiz 45' · bài tập ~2h

---

## 08 · Phân cụm: K-Means và DBSCAN

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Cần biết trước:** khoảng cách L2 (Tuần 2, bài 01); chuẩn hóa đặc trưng (bài 05).

> **📖 Từ mới**
> - **Phân cụm (clustering):** chia dữ liệu **không có nhãn** thành các nhóm, sao cho điểm trong cùng nhóm giống nhau.
> - **Tâm cụm (centroid):** điểm trung bình của một cụm.
> - **Inertia:** tổng bình phương khoảng cách từ mỗi điểm tới tâm cụm của nó; càng nhỏ thì cụm càng "chặt".
> - **Silhouette:** điểm từ −1 tới 1 cho biết một điểm nằm đúng cụm tới mức nào.
> - **Mật độ (density):** số điểm trong một vùng lân cận nhỏ.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.**
- **K-Means:** đặt $k$ thùng rác trong công viên sao cho tổng quãng đường mọi người phải đi tới thùng gần nhất là nhỏ nhất. Lặp lại hai bước: (1) mỗi người chọn thùng gần nhất; (2) dời mỗi thùng tới **chính giữa** nhóm người đã chọn nó.
- **DBSCAN:** tìm các "đám đông": một người có nhiều hàng xóm ở gần thì thuộc một đám đông; những đám đông chạm nhau được nối lại thành một; người đứng lẻ loi là **nhiễu**. Không cần biết trước có bao nhiêu đám đông.

**Ba ý cốt lõi:**
1. K-Means cần chọn $k$, giả định các cụm **tròn, cỡ tương đương**, và rất nhạy với outlier.
2. DBSCAN tìm được cụm **hình dạng bất kỳ** và tự đánh dấu nhiễu, nhưng cần chọn bán kính `eps`.
3. Phân cụm **không có đáp án đúng**: phải kiểm tra bằng metric nội tại (silhouette) **và** bằng ý nghĩa với bài toán.

---

### TẦNG 2 · Kiến thức chuẩn *(bắt buộc · đề thi chấm theo tầng này)*

#### 8.1 Thuật ngữ chuẩn

| Thuật ngữ | Định nghĩa | Nguồn |
|---|---|---|
| **K-Means objective** | $\min\sum_{i}\|x_i - \mu_{c(i)}\|^2$ (inertia) | MacQueen (1967) |
| **Thuật toán Lloyd** | Lặp: gán mỗi điểm vào tâm gần nhất → cập nhật tâm bằng trung bình của cụm. **Inertia không bao giờ tăng** sau mỗi bước, nên thuật toán dừng sau hữu hạn bước, nhưng chỉ tới một **cực tiểu địa phương** | Lloyd (1982) |
| **k-means++** | Chọn tâm đầu tiên ngẫu nhiên; mỗi tâm tiếp theo chọn với xác suất $\propto D(x)^2$ (bình phương khoảng cách tới tâm gần nhất). Có bảo đảm inertia kỳ vọng $\le O(\log k)$ lần tối ưu | Arthur & Vassilvitskii, SODA 2007 |
| **Silhouette** | $s(i) = \dfrac{b(i) - a(i)}{\max(a(i), b(i))}$, với $a$ = khoảng cách trung bình tới các điểm **cùng cụm**, $b$ = khoảng cách trung bình tới **cụm gần nhất khác** | Rousseeuw (1987) |
| **DBSCAN** | Điểm **lõi** (core) có ≥ `min_samples` điểm trong bán kính `eps`; cụm là tập các điểm lõi nối được với nhau, cộng các điểm **biên** (border) nằm cạnh chúng; còn lại là **nhiễu** (nhãn −1) | Ester et al., KDD 1996 |
| **ARI (Adjusted Rand Index)** | So sánh cách phân cụm với nhãn thật (nếu có): 1 là khớp hoàn toàn, khoảng 0 là ngẫu nhiên | Hubert & Arabie (1985) |

*Vì sao Lloyd không làm tăng inertia (phác thảo):* bước gán chọn tâm **gần nhất** cho mỗi điểm, nên không làm tăng tổng. Bước cập nhật dùng **trung bình**, mà trung bình là điểm làm tổng bình phương khoảng cách nhỏ nhất (đạo hàm bằng 0), nên cũng không làm tăng tổng.

#### 8.2 Ví dụ tính tay

**(a) Lloyd trên 1 chiều.** Dữ liệu $1, 2, 3, 8, 9, 10, 25$, với $k = 2$ và tâm ban đầu $(2, 3)$:

| Vòng | Gán | Tâm mới | Inertia |
|---|---|---|---|
| 1 | {1, 2} · {3, 8, 9, 10, 25} | 1.5 · 11 | $0.5 + 274 = 274.5$ |
| 2 | {1, 2, 3} · {8, 9, 10, 25} | 2 · 13 | $2 + 194 = $ **196** |
| 3 | không đổi | — | dừng |

Outlier 25 **kéo tâm** của cụm thứ hai lên 13, xa khỏi nhóm 8–10. Với $k = 3$, tâm là 2, 9, 25 và inertia chỉ còn **4**.

**(b) Silhouette với $k = 2$ ở trên** (đã chạy kiểm chứng):
- Điểm 9: $a = (1 + 1 + 16)/3 = 6$, $b = (8 + 7 + 6)/3 = 7$, nên $s = 1/7 = 0.143$.
- Điểm 8: $a = (1 + 2 + 17)/3 = 6.67$, $b = (7 + 6 + 5)/3 = 6$, nên $s = $ **−0.10**. Âm nghĩa là điểm 8 "gần" cụm kia hơn cụm của chính nó.
- Silhouette trung bình là 0.46.

**(c) Chọn $k$** trên 600 điểm sinh từ 4 cụm (đo trên máy bạn):

| $k$ | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|
| Inertia | 2 922 | 1 682 | **1 044** | 919 | 810 |
| Silhouette | 0.463 | 0.467 | **0.500** | 0.436 | 0.402 |

Inertia **luôn giảm** khi tăng $k$, nên chỉ nhìn được chỗ "gãy khúc" (*elbow*). Silhouette có **cực đại**, ở đây đúng tại $k = 4$.

**(d) Khi K-Means thất bại.** Dữ liệu hình hai nửa vầng trăng lồng nhau (`make_moons`, 500 điểm):
- K-Means cho ARI **0.26**: nó cắt dữ liệu bằng một đường thẳng.
- DBSCAN (`eps=0.2`, `min_samples=5`) cho ARI **1.00**.

#### 8.3 So sánh

| | K-Means | DBSCAN |
|---|---|---|
| Cần chọn | $k$ | `eps`, `min_samples` |
| Hình dạng cụm | Tròn (lồi), cỡ tương đương | Bất kỳ |
| Outlier | Kéo lệch tâm | Tự đánh dấu là nhiễu |
| Mật độ các cụm khác nhau nhiều | Ổn | Kém, vì chỉ có một `eps` chung (HDBSCAN khắc phục) |
| Chi phí | $O(n \cdot k \cdot d)$ mỗi vòng, rất nhanh | $O(n\log n)$ nếu có chỉ mục không gian; dữ liệu nhiều chiều thì chậm |

**Trong hệ thống AI:** phân cụm embedding của câu hỏi người dùng để khám phá các chủ đề mới; phân cụm các ca model trả lời sai để phân tích lỗi; và K-Means chính là bước đầu của chỉ mục **IVF** trong vector database (Tuần 8): chia không gian thành các cụm, khi tìm kiếm chỉ duyệt vài cụm gần nhất.

#### 8.4 Code chuẩn

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

Z = StandardScaler().fit_transform(X_num)        # PHẢI chuẩn hóa: K-Means dùng khoảng cách
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    print(k, km.inertia_, silhouette_score(Z, km.labels_))
labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(Z)    # nhãn −1 là nhiễu
```

#### 8.5 Cạm bẫy

1. **Không chuẩn hóa:** đặc trưng có thang lớn (ví dụ `answer_tokens` cỡ hàng trăm) chiếm toàn bộ khoảng cách.
2. **Chạy K-Means một lần với khởi tạo ngẫu nhiên:** dễ kẹt ở cực tiểu địa phương tệ. Dùng k-means++ và `n_init` > 1.
3. **Chọn $k$ chỉ bằng inertia:** inertia luôn giảm khi $k$ tăng.
4. **Dùng K-Means cho biến phân loại đã one-hot:** "trung bình" của các cột one-hot không có nhiều ý nghĩa; cân nhắc k-modes hoặc phân cụm trên embedding.
5. **Coi kết quả phân cụm là "sự thật".** Đổi cách chuẩn hóa hay đổi $k$ là đổi kết quả.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"K-Means có luôn hội tụ không? Có tìm ra tối ưu toàn cục không?"* Luôn hội tụ (inertia không tăng, và số cách gán là hữu hạn), nhưng **chỉ tới cực tiểu địa phương**; tìm tối ưu toàn cục là bài toán NP-khó.

#### 8.6 Đọc thêm

- ISLR §12.4; ESL §14.3.
- scikit-learn User Guide, *2.3 Clustering* (có bảng so sánh rất hữu ích giữa các thuật toán).
- D. Arthur & S. Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, SODA 2007.

---

## 09 · Trực quan hóa dữ liệu nhiều chiều: PCA, t-SNE, UMAP

> ⏱ Tầng 1: 10' · Tầng 2: 20'
> **Cần biết trước:** PCA (Tuần 2, bài 03); KL divergence (Tuần 2, bài 11).

> **📖 Từ mới**
> - **Nhúng xuống 2D (2D embedding):** gán cho mỗi điểm nhiều chiều một tọa độ 2D để vẽ lên màn hình.
> - **Cấu trúc cục bộ / toàn cục:** ai là hàng xóm của ai / các nhóm lớn nằm cách nhau bao xa.
> - **Perplexity (của t-SNE):** gần đúng là "số hàng xóm mà mỗi điểm quan tâm". Khác với perplexity của language model (Tuần 2, bài 11), dù cùng công thức gốc.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Vẽ bản đồ thế giới lên tờ giấy phẳng: kiểu chiếu nào cũng phải **hy sinh** một thứ (diện tích, hình dạng hoặc khoảng cách).
- **PCA** giống như chụp ảnh bóng từ góc nhìn "rộng nhất": giữ được khoảng cách xa, nhưng các nhóm có thể bị chồng lên nhau.
- **t-SNE** và **UMAP** giống như một bản đồ "ai đứng cạnh ai": hàng xóm vẫn là hàng xóm, nhưng **khoảng cách giữa các nhóm và kích thước của từng nhóm trên hình không mang ý nghĩa**.

**Ba ý cốt lõi:**
1. PCA tuyến tính, nhanh, giữ cấu trúc toàn cục. t-SNE/UMAP phi tuyến, giữ cấu trúc cục bộ.
2. t-SNE/UMAP dùng để **nhìn**, không dùng để **đo**.
3. Hình đẹp không chứng minh được gì; luôn kiểm tra lại bằng số liệu.

---

### TẦNG 2 · Kiến thức chuẩn

#### 9.1 So sánh

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Ý tưởng | Chiếu lên các hướng có phương sai lớn nhất | Ghép phân phối "hàng xóm" ở không gian gốc (Gauss) với phân phối ở 2D (Student-t), cực tiểu $D_{\mathrm{KL}}$ | Dựng đồ thị láng giềng (fuzzy) rồi tối ưu bố cục 2D cho giống đồ thị đó |
| Giữ | Khoảng cách toàn cục | Hàng xóm gần | Hàng xóm gần, và phần nào cấu trúc toàn cục |
| Tham số chính | $k$ | `perplexity` (5–50) | `n_neighbors`, `min_dist` |
| Áp dụng cho điểm mới | Có (`transform`) | **Không** (scikit-learn) | Có |
| Tốc độ | Rất nhanh | Chậm ($O(n\log n)$ với Barnes–Hut) | Nhanh hơn t-SNE |
| Nguồn | Pearson (1901) | van der Maaten & Hinton, JMLR 2008 | McInnes et al., arXiv:1802.03426 |

UMAP nằm trong gói riêng `umap-learn` (không có sẵn trong scikit-learn). Tuần này không bắt buộc cài.

#### 9.2 Đo trên máy bạn

Bộ `digits` (1 797 ảnh chữ số 8×8, tức 64 chiều). Đo xem sau khi giảm xuống 2D, các lớp còn tách nhau tới đâu, bằng accuracy của kNN ($k = 5$, CV 5-fold) **trên chính tọa độ 2D**:

| Không gian | Thời gian | Accuracy kNN |
|---|---|---|
| Gốc, 64 chiều | — | 0.963 |
| PCA → 2D | < 0.1 s | 0.603 |
| t-SNE → 2D (`perplexity=30`) | 2.4 s | **0.976** |

PCA 2D chỉ giữ được 2 hướng có phương sai lớn nhất, nên nhiều chữ số chồng lên nhau. t-SNE giữ được gần như toàn bộ cấu trúc hàng xóm.

Lưu ý: con số 0.976 **không** có nghĩa là "nên dùng t-SNE làm đặc trưng". t-SNE được fit trên **toàn bộ** dữ liệu, kể cả các điểm sẽ nằm trong fold test, và không biến đổi được điểm mới. Đây chính là một dạng rò rỉ (bài 01).

#### 9.3 Cạm bẫy khi đọc hình t-SNE/UMAP

Theo Wattenberg, Viégas, Johnson, *How to Use t-SNE Effectively*, Distill 2016:
1. **Kích thước** của một cụm trên hình không phản ánh độ phân tán thật của nó.
2. **Khoảng cách giữa các cụm** trên hình có thể vô nghĩa.
3. **Perplexity khác nhau** cho ra hình rất khác nhau; luôn thử vài giá trị.
4. **Dữ liệu ngẫu nhiên** cũng có thể hiện ra thành "cụm".
5. Chạy với seed khác nhau thì ra hình khác nhau.

Quy trình hay dùng với embedding 768 chiều: PCA xuống khoảng 50 chiều (bỏ nhiễu, chạy nhanh hơn) → t-SNE hoặc UMAP xuống 2D → **tô màu theo nhãn hoặc metadata** để kiểm tra giả thuyết.

#### 9.4 Đọc thêm

- L. van der Maaten & G. Hinton, *Visualizing Data using t-SNE*, JMLR 2008.
- M. Wattenberg, F. Viégas, I. Johnson, *How to Use t-SNE Effectively*, Distill 2016.
- L. McInnes, J. Healy, J. Melville, *UMAP*, arXiv:1802.03426.

---

## 10 · scikit-learn API và Pipeline

> ⏱ Tầng 1: 10' · Tầng 2: 35'
> **Loại: Kỹ thuật.** Trọng tâm là API, quy trình và quyết định thiết kế.
> **Cần biết trước:** rò rỉ dữ liệu (bài 01); cross-validation (bài 03).

> **📖 Từ mới**
> - **Estimator:** mọi đối tượng có `fit`. **Transformer:** có thêm `transform` (biến đổi dữ liệu). **Predictor:** có thêm `predict` (và thường có `predict_proba`).
> - **Pipeline:** chuỗi các bước "transformer → … → model", dùng như **một** model duy nhất.
> - **ColumnTransformer:** áp dụng các phép xử lý khác nhau cho các nhóm cột khác nhau.
> - **Imputer:** điền giá trị thiếu (NaN).
> - **Serialize (lưu model):** ghi model đã huấn luyện ra file để tải lại sau.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Dây chuyền làm bánh có công thức cố định: rây bột → trộn → nướng. Pipeline đóng gói **cả dây chuyền** thành một chiếc máy. Có hai lợi ích:
1. Lúc "học công thức" (fit), mỗi công đoạn chỉ được nhìn **nguyên liệu huấn luyện**. Lượng đường trung bình chẳng hạn được đo trên dữ liệu train, không bao giờ trên dữ liệu test. Nhờ vậy không có rò rỉ.
2. Lúc phục vụ, đưa nguyên liệu thô vào là ra bánh, **đúng y công thức** lúc huấn luyện. Nhờ vậy không có training–serving skew.

**Ba ý cốt lõi:**
1. Mọi bước có `fit` (imputer, scaler, encoder, chọn đặc trưng) **phải** nằm trong Pipeline.
2. `fit` chỉ gọi trên train; `transform` và `predict` thì dùng cho mọi dữ liệu.
3. Lưu **cả Pipeline**, không chỉ lưu model.

---

### TẦNG 2 · Kiến thức chuẩn

#### 10.1 Quy ước API (Buitinck et al., *API design for machine learning software*, 2013)

| Quy ước | Ý nghĩa |
|---|---|
| Hyperparameter đặt trong `__init__` | `LogisticRegression(C=0.5)`; **không** học từ dữ liệu |
| Thuộc tính đã học kết thúc bằng `_` | `scaler.mean_`, `model.coef_`, `tree.feature_importances_`; chỉ có sau khi `fit` |
| `fit` trả về `self` | Nên viết được `model.fit(X, y).predict(X2)` |
| `fit_transform(X)` | Tương đương `fit(X).transform(X)` nhưng có thể nhanh hơn. **Chỉ dùng trên train** |
| `get_params` / `set_params` | Đặt tham số lồng nhau bằng `bước__tham_số`, ví dụ `clf__C`, `pre__num__impute__strategy` |
| `clone(estimator)` | Tạo bản sao **chưa fit** với cùng tham số; CV dùng nó cho mỗi fold |

#### 10.2 Các bộ tiền xử lý hay dùng

| Transformer | Công thức / hành vi | Khi nào dùng |
|---|---|---|
| `StandardScaler` | $(x - \mu)/\sigma$ ($\sigma$ chia $n$) | Model tuyến tính, SVM, kNN, mạng neural |
| `MinMaxScaler` | $(x - \min)/(\max - \min)$, đưa về $[0, 1]$ | Cần một khoảng cố định; rất nhạy với outlier |
| `RobustScaler` | $(x - \text{median})/\text{IQR}$ | Có outlier |
| `SimpleImputer(strategy="median", add_indicator=True)` | Điền trung vị và thêm cột 0/1 "có thiếu không" | Việc thiếu dữ liệu tự nó có thể mang thông tin |
| `OneHotEncoder(handle_unknown="ignore")` | Mỗi giá trị một cột; giá trị lạ lúc predict thành toàn 0 | Biến phân loại, số giá trị ít |
| `OrdinalEncoder` | Mỗi giá trị một số nguyên | Model cây, hoặc biến có thứ tự thật |

**Tính tay với $x = (1, 2, 3, 4, 100)$** (có một outlier), đã chạy kiểm chứng:

| Scaler | Tham số học được | Kết quả |
|---|---|---|
| Standard | $\mu = 22$, $\sigma = 39.01$ | $(-0.54,\ -0.51,\ -0.49,\ -0.46,\ 2.00)$ |
| MinMax | $\min = 1$, $\max = 100$ | $(0,\ 0.01,\ 0.02,\ 0.03,\ 1)$ |
| Robust | median $= 3$, IQR $= 4 - 2 = 2$ | $(-1,\ -0.5,\ 0,\ 0.5,\ 48.5)$ |

Outlier làm Standard và MinMax **ép bốn điểm bình thường dính vào nhau**. Robust giữ được khoảng cách giữa chúng và để outlier nằm riêng ở xa.

#### 10.3 Pipeline của project

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler())])
pre = ColumnTransformer([("num", num, FEATURES_NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])

model.set_params(clf__C=0.3)                       # tham số lồng nhau
cross_val_score(model, X_tr, y_tr, cv=cv)          # mỗi fold fit imputer/scaler RIÊNG → không rò rỉ
model.fit(X_tr, y_tr).predict_proba(X_new)         # X_new là dữ liệu THÔ, có NaN, có intent lạ
```

Ghi chú:
- Với model cây, bỏ bước `StandardScaler` (không cần).
- `HistGradientBoostingClassifier` tự xử lý được `NaN`.
- `model.set_output(transform="pandas")` làm cho các bước biến đổi trả về DataFrame có tên cột, rất tiện khi debug.

#### 10.4 Lưu và tải model

```python
import joblib
joblib.dump(model, "artifacts/model.joblib")       # lưu CẢ pipeline
model = joblib.load("artifacts/model.joblib")
```

| Quyết định | Lý do |
|---|---|
| Lưu cả pipeline | Tiền xử lý lúc phục vụ phải **giống hệt** lúc huấn luyện |
| Ghi kèm phiên bản scikit-learn, Python, và hash của dữ liệu | File pickle không đảm bảo tải được trên phiên bản khác (scikit-learn sẽ cảnh báo `InconsistentVersionWarning`) |
| **Chỉ tải file từ nguồn tin cậy** | `joblib` và `pickle` có thể **chạy code tùy ý** khi tải. Một file `.joblib` lạ chẳng khác gì một file `.exe` lạ. Cần định dạng an toàn hơn thì xem `skops.io` hoặc ONNX (Tuần 13) |

#### 10.5 Cạm bẫy

1. **`scaler.fit_transform(X_test)`** thay vì `scaler.transform(X_test)`: test được chuẩn hóa theo thống kê của **chính nó**.
2. **Tiền xử lý bằng pandas trước khi vào Pipeline** (ví dụ `df.fillna(df.mean())` trên toàn bộ dữ liệu): rò rỉ, và còn phải viết lại y hệt ở phía phục vụ.
3. **Quên `handle_unknown="ignore"`**: gặp một intent mới khi chạy thật là chương trình crash.
4. **Lưu mỗi model mà quên preprocessor.**
5. **Tên tham số lồng nhau sai** (`C` thay vì `clf__C`): `GridSearchCV` sẽ báo lỗi `Invalid parameter`.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Vì sao phải dùng Pipeline thay vì tự gọi từng bước?"* Ba lý do: chặn rò rỉ trong CV, tránh training–serving skew, và tune được tham số của **mọi** bước cùng lúc.

#### 10.6 Đọc thêm

- scikit-learn User Guide, *6.1 Pipelines and composite estimators*, *7.3 Preprocessing data*, *9 Model persistence*.
- L. Buitinck et al., *API design for machine learning software: experiences from the scikit-learn project*, 2013.

---

## 11 · Tuning hyperparameter

> ⏱ Tầng 1: 10' · Tầng 2: 30'
> **Loại: Kỹ thuật** (kèm ước lượng xác suất đơn giản).
> **Cần biết trước:** cross-validation và nested CV (bài 03).

> **📖 Từ mới**
> - **Hyperparameter:** tham số do **người** chọn trước khi huấn luyện (ví dụ `learning_rate`, `max_depth`, `C`), khác với tham số mà model **học** được.
> - **Không gian tìm kiếm (search space):** khoảng giá trị cho từng hyperparameter.
> - **Grid search / random search:** thử mọi tổ hợp trên lưới / thử các điểm lấy ngẫu nhiên.
> - **Bayesian optimization:** dùng kết quả các lần thử trước để **đoán** vùng nào đáng thử tiếp.
> - **Early stopping:** dừng huấn luyện khi điểm validation thôi cải thiện.

### TẦNG 1 · Trực giác

**Ví dụ đời thường.** Chỉnh nhiệt độ và thời gian nướng bánh.
- **Grid:** thử 3 mức nhiệt × 3 mức thời gian = 9 mẻ.
- **Random:** thử 9 mẻ với nhiệt độ và thời gian **ngẫu nhiên**.

Nếu thực ra chỉ **nhiệt độ** là quan trọng, grid chỉ thử được 3 mức nhiệt khác nhau, còn random thử được **9 mức**. Đó là lý do random search thường thắng grid search (Bergstra & Bengio, JMLR 2012).

**Ba ý cốt lõi:**
1. Random search hiệu quả hơn grid khi chỉ vài hyperparameter thật sự quan trọng, và gần như luôn là vậy.
2. Tham số có tính chất "tỉ lệ" (learning rate, `C`, `alpha`) phải lấy mẫu theo **thang log**.
3. Điểm CV tốt nhất sau khi tune luôn lạc quan; con số cuối cùng phải đo trên test.

---

### TẦNG 2 · Kiến thức chuẩn

#### 11.1 Các phương pháp

| Phương pháp | Cách làm | Khi nào dùng | scikit-learn / thư viện |
|---|---|---|---|
| **Grid search** | Mọi tổ hợp trên lưới | 1–2 tham số, khi đã biết vùng tốt | `GridSearchCV` |
| **Random search** | Lấy mẫu từ các phân phối | **Mặc định** khi có từ 3 tham số | `RandomizedSearchCV` |
| **Successive halving / Hyperband** | Chạy nhiều cấu hình với ngân sách nhỏ, giữ lại phần tốt nhất, tăng ngân sách cho chúng | Ngân sách hạn chế, model huấn luyện lâu | `HalvingRandomSearchCV`; Li et al., JMLR 2018 |
| **Bayesian optimization (TPE, GP)** | Dựng một mô hình xấp xỉ "cấu hình → điểm số", chọn điểm thử tiếp theo cân bằng giữa khai thác và khám phá | Mỗi lần thử rất đắt | Optuna (Akiba et al., KDD 2019); Bergstra et al., NeurIPS 2011 |
| **Early stopping** | Theo dõi điểm validation trong lúc huấn luyện, dừng khi không cải thiện sau `n_iter_no_change` vòng | Boosting, mạng neural: tự chọn số vòng | `HistGradientBoosting(early_stopping=...)` |

#### 11.2 Ước lượng: random search cần bao nhiêu lần thử?

Gọi "vùng tốt" là 5% không gian tìm kiếm có điểm số cao nhất. Mỗi lần thử ngẫu nhiên rơi vào vùng đó với xác suất 0.05. Sau $n$ lần thử, xác suất **ít nhất một lần** rơi vào vùng tốt là $1 - 0.95^n$:

| $n$ | 20 | 60 | 100 |
|---|---|---|---|
| $1 - 0.95^n$ | 0.64 | **0.95** | 0.994 |

Con số **60 lần thử** cho 95% (Bergstra & Bengio 2012) **không phụ thuộc vào số chiều** của không gian tìm kiếm. Grid search thì có số điểm tăng theo hàm mũ với số tham số: 5 tham số × 5 mức là 3 125 điểm.

**Chi phí:** số lần fit = $n_{\text{iter}} \times k_{\text{fold}}$. Trên dữ liệu project, `RandomizedSearchCV` với 20 lần thử × 5 fold mất khoảng **20 giây** (22 core). Cấu hình tốt nhất đạt AUC CV 0.811, gần bằng cấu hình mặc định đã chọn sẵn (0.812): với HGB, các mặc định hợp lý đã gần tối ưu.

#### 11.3 Thiết kế không gian tìm kiếm

```python
from scipy.stats import loguniform, randint

space = {
    "clf__learning_rate": loguniform(0.01, 0.3),     # thang LOG: 0.01→0.03 quan trọng ngang 0.1→0.3
    "clf__max_leaf_nodes": randint(4, 64),
    "clf__min_samples_leaf": randint(5, 200),
    "clf__max_iter": randint(50, 500),
}
search = RandomizedSearchCV(pipe, space, n_iter=20, cv=cv, scoring="roc_auc", random_state=0, n_jobs=-1)
search.fit(X_tr, y_tr)
print(search.best_params_, search.best_score_)
```

**Vì sao dùng thang log?** Với `uniform(0.01, 0.3)`, khoảng 97% số mẫu nằm trên 0.02, và vùng $[0.01, 0.03]$ chỉ được thử khoảng 7% số lần, dù vùng đó thường rất quan trọng.

#### 11.4 Thứ tự ưu tiên khi tune (kinh nghiệm thực tế)

| Model | Tham số quan trọng nhất |
|---|---|
| Logistic regression | `C` (thang log) |
| Random Forest | `max_features`, `min_samples_leaf`; số cây thì chỉ cần "đủ nhiều" |
| Gradient boosting | `learning_rate` và số cây (dùng early stopping), rồi tới kích thước cây (`max_leaf_nodes` hoặc `max_depth`), rồi `min_samples_leaf` |

#### 11.5 Cạm bẫy

1. **Tune trên test set.**
2. **Báo `best_score_` như kết quả cuối cùng.** Trên project: AUC CV tốt nhất 0.817, AUC test **0.796**.
3. **Không gian tìm kiếm quá hẹp**, hoặc dùng thang tuyến tính cho learning rate.
4. **Tune quá kỹ trên dữ liệu nhỏ**: overfit vào validation. Khi mức cải thiện nhỏ hơn độ dao động giữa các fold (bài 03) thì dừng tune.
5. **Tune mà quên Pipeline**: tiền xử lý không được fit lại cho mỗi fold.

**Sắc thái hay bị hỏi khi phỏng vấn:** *"Grid hay random?"* Random, và nêu được lập luận của Bergstra & Bengio (số giá trị phân biệt trên mỗi tham số quan trọng; công thức $1 - 0.95^n$).

#### 11.6 Đọc thêm

- J. Bergstra & Y. Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 2012.
- T. Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
- scikit-learn User Guide, *3.2 Tuning the hyper-parameters of an estimator*.

---

## Bài tập Ngày 3 · `quality/cluster.py` và `quality/pipeline.py` (~2h)

Test: `python -m pytest -q tests/test_cluster.py tests/test_pipeline.py` (11 test).

**`quality/cluster.py`** (NumPy thuần; không dùng `sklearn.cluster`):

| Hàm | Yêu cầu |
|---|---|
| `kmeans_pp_init(X, k, rng)` | k-means++ (bảng 8.1). Trả về `k` tâm, **lấy từ chính các điểm dữ liệu** |
| `kmeans(X, k, *, n_init=10, max_iter=300, tol=1e-6, seed=0, init=None)` | Thuật toán Lloyd. Chấp nhận `X` một chiều (coi là một cột). `init` cho trước thì chỉ chạy **một lần** từ đúng các tâm đó; nếu không thì chạy `n_init` lần với k-means++ (dùng **một** `rng = default_rng(seed)`) và giữ lần có inertia nhỏ nhất. Dừng khi tâm dịch chuyển tối đa ≤ `tol`. Cụm rỗng thì đặt lại tâm đó vào điểm xa nhất. Trả về `(centers, labels, inertia)`. Không thỏa $1 \le k \le n$ thì raise `ValueError` |

**`quality/pipeline.py`** (dùng scikit-learn):

| Hàm | Yêu cầu |
|---|---|
| `split_xy(df)` | `X` chỉ gồm `FEATURES_NUM + FEATURES_CAT` (trong `data.py`); `y = df["bad"]` |
| `build_preprocessor(kind)` | `"linear"`: số → imputer trung vị (`add_indicator=True`) rồi `StandardScaler`; `"tree"`: số → chỉ imputer. Phân loại → `OneHotEncoder(handle_unknown="ignore")`. Kind khác thì raise `ValueError` |
| `build_model(name, **params)` | `Pipeline([("pre", ...), ("clf", ...)])` với `"logreg"`, `"rf"`, `"hgb"` và các tham số mặc định ở [project.md](project.md) §3. `params` ghi đè mặc định. Tên khác thì raise `ValueError` |
