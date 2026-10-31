# Quiz · Ngày 3 · Phân cụm, trực quan hóa, Pipeline, tuning (bài 08–11)

> ⏱ **45 phút** · Quick test 15' + Tự luận 30' · **30 điểm** · Đạt khi ≥ 21
> Làm trên giấy, được dùng máy tính cầm tay. **Không chạy code.** Đáp án ở cuối file.

## Phần 1 — Quick test (14 câu, câu 14 được 2đ, các câu khác 1đ)

1. Thuật toán Lloyd đảm bảo điều gì về inertia? Nó có đảm bảo tìm được tối ưu toàn cục không?
2. Dữ liệu một chiều $0, 2, 10, 12$, $k = 2$, tâm ban đầu $(0, 2)$. Lloyd hội tụ về các tâm nào, và inertia cuối cùng bằng bao nhiêu?
3. Tính silhouette của một điểm có $a = 2$, $b = 6$; và của một điểm có $a = 5$, $b = 4$.
4. Vì sao không thể chọn $k$ bằng cách lấy $k$ có inertia nhỏ nhất?
5. Nhãn −1 trong kết quả của DBSCAN nghĩa là gì?
6. Dữ liệu gồm hai nửa vầng trăng lồng nhau. Nên dùng K-Means hay DBSCAN?
7. Đúng/Sai: trên hình t-SNE, hai cụm nằm xa nhau nghĩa là chúng rất khác nhau trong không gian gốc.
8. `sklearn.manifold.TSNE` có biến đổi được điểm dữ liệu mới (không có trong lúc fit) không?
9. Áp `StandardScaler` (chia $n$) lên $(2, 4, 6)$. Kết quả là gì?
10. Áp `MinMaxScaler` lên $(10, 20, 50)$. Kết quả là gì?
11. Đã `scaler.fit(X_train)`. Với `X_test` thì gọi hàm nào?
12. Pipeline có bước tên `"clf"` là một `LogisticRegression`. Viết tên tham số để tune `C` trong `GridSearchCV`.
13. "Vùng tốt" chiếm 10% không gian tìm kiếm. Random search cần tối thiểu bao nhiêu lần thử để có ≥ 95% khả năng rơi vào vùng đó ít nhất một lần?
14. *(2đ)* So sánh (a) grid search $4 \times 4 \times 4$ với CV 5-fold và (b) random search `n_iter=30` với CV 5-fold: mỗi cách fit bao nhiêu lần? Với mỗi tham số, mỗi cách thử được tối đa bao nhiêu giá trị phân biệt?

## Phần 2 — Tự luận (4 câu, 30 phút)

**T1 · K-Means tính tay (4đ).** Bảy điểm $A(1,1)$, $B(1,2)$, $C(2,1)$, $D(6,5)$, $E(7,5)$, $F(6,6)$, $G(3,3)$; $k = 2$; tâm ban đầu là $A$ và $G$.
- (a) Chạy Lloyd tới khi hội tụ. Ở mỗi vòng ghi phép gán, các tâm mới và inertia.
- (b) Tính silhouette của điểm $G$ ở kết quả cuối cùng. *(Cho sẵn: $\|G - A\| = 2.828$, $\|G - B\| = \|G - C\| = 2.236$, $\|G - D\| = 3.606$, $\|G - E\| = 4.472$, $\|G - F\| = 4.243$.)*
- (c) Điều gì xảy ra nếu một đặc trưng được đo bằng đơn vị lớn gấp 1000 lần đặc trưng kia mà không chuẩn hóa?

**T2 · Tìm lỗi (3đ).** Tìm ít nhất **4** lỗi trong đoạn code sau và sửa:

```python
df = df.fillna(df.mean(numeric_only=True))
X = pd.get_dummies(df.drop(columns=["bad"]))
X = StandardScaler().fit_transform(X)
X = SelectKBest(f_classif, k=10).fit_transform(X, df["bad"])
X_tr, X_te, y_tr, y_te = train_test_split(X, df["bad"], test_size=0.2)
search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1, 10]}, scoring="accuracy").fit(X_tr, y_tr)
for C in [0.1, 1, 10]:
    print(C, LogisticRegression(C=C).fit(X_tr, y_tr).score(X_te, y_te))
joblib.dump(search.best_estimator_, "model.joblib")
```

**T3 · Lập kế hoạch tuning (4đ).** Mỗi lần fit HistGradientBoosting trên toàn bộ dữ liệu train mất 30 giây. Máy chạy được 8 lần fit song song. Ngân sách là 1 giờ.
- (a) Với CV 5-fold, thử được tối đa bao nhiêu cấu hình?
- (b) Thiết kế không gian tìm kiếm cho 4 tham số (`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `max_iter`), nói rõ thang đo của từng tham số.
- (c) Có cách nào thử được nhiều cấu hình hơn trong cùng ngân sách? Nêu 2 cách.
- (d) Quy trình đánh giá cuối cùng là gì, để con số báo cáo không bị lạc quan?

**T4 · Code (4đ).** Viết `silhouette(X, labels) -> float` (silhouette trung bình, khoảng cách Euclid), **vector hóa** bằng ma trận khoảng cách. Quy ước: điểm nằm trong cụm chỉ có 1 điểm thì có $s = 0$. Ước lượng bộ nhớ cần cho $n = 100\,000$ điểm (float64), và nêu cách xử lý.

---

<details><summary><b>Đáp án</b> — chỉ mở sau khi làm xong</summary>

**Quick test**

1. Inertia **không tăng** sau mỗi bước, nên thuật toán hội tụ; nhưng chỉ tới một **cực tiểu địa phương**.
2. Vòng 1: {0} · {2, 10, 12}, tâm 0 và 8. Vòng 2: {0, 2} · {10, 12}, tâm **1 và 11**. Inertia $= 1 + 1 + 1 + 1 =$ **4**.
3. $(6 - 2)/6 = 0.667$; $(4 - 5)/5 = -0.2$ (điểm thứ hai gần cụm khác hơn cụm của chính nó).
4. Inertia **luôn giảm** khi $k$ tăng, và bằng 0 khi $k = n$.
5. Điểm **nhiễu**: không thuộc cụm nào.
6. **DBSCAN** (ARI 1.00 so với 0.26 của K-Means trong bài 08).
7. **Sai**. Khoảng cách giữa các cụm trên hình t-SNE có thể vô nghĩa.
8. **Không**: `TSNE` chỉ có `fit_transform`.
9. $\mu = 4$, $\sigma = \sqrt{8/3} = 1.633$, nên kết quả là $(-1.225,\ 0,\ 1.225)$.
10. $(0,\ 0.25,\ 1)$.
11. `scaler.transform(X_test)`, **không** gọi `fit_transform`.
12. `"clf__C"`.
13. $1 - 0.9^n \ge 0.95 \Rightarrow n \ge \ln 0.05/\ln 0.9 = 28.4$, nên **29** lần.
14. (a) $64 \times 5 =$ **320** lần fit; mỗi tham số chỉ có **4** giá trị. (b) $30 \times 5 =$ **150** lần fit; mỗi tham số có tới **30** giá trị phân biệt. Random search rẻ hơn mà khám phá mỗi chiều kỹ hơn. *(1đ mỗi ý)*

**T1** *(đã chạy kiểm chứng)*

(a) *(2.5đ)*
- **Vòng 1** (tâm $A$, $G$). $B$, $C$ gần $A$ hơn; $D$, $E$, $F$ gần $G$ hơn; $G$ thuộc tâm của chính nó. Các cụm là {A, B, C} · {D, E, F, G}. Tâm mới: $(1.333,\ 1.333)$ và $(5.5,\ 4.75)$. Inertia $= 1.333 + 13.75 = 15.083$.
- **Vòng 2.** $G(3,3)$: khoảng cách bình phương tới $(1.333, 1.333)$ là $5.56$, tới $(5.5, 4.75)$ là $9.31$, nên $G$ **chuyển sang cụm 1**. Các cụm là {A, B, C, G} · {D, E, F}. Tâm mới: $(1.75,\ 1.75)$ và $(6.333,\ 5.333)$. Inertia $=$ **6.833**.
- **Vòng 3:** không đổi, nên dừng.

(b) $a(G) = (2.828 + 2.236 + 2.236)/3 = 2.433$; $b(G) = (3.606 + 4.472 + 4.243)/3 = 4.107$; $s = (4.107 - 2.433)/4.107 \approx$ **0.41**. *(1đ)*

(c) Khoảng cách gần như chỉ do đặc trưng có thang lớn quyết định; đặc trưng kia bị bỏ qua, và K-Means chỉ cắt theo một trục. Phải chuẩn hóa trước. *(0.5đ)*

**T2** *(3đ: 0.75đ cho mỗi lỗi tìm đúng và có cách sửa, tối đa 4 lỗi)*

1. `fillna(df.mean())` trên **toàn bộ** dữ liệu: rò rỉ. Đưa vào Pipeline bằng `SimpleImputer`.
2. `StandardScaler().fit_transform` **trước khi** chia: rò rỉ. Đưa vào Pipeline.
3. `SelectKBest` trên toàn bộ dữ liệu **có nhìn nhãn**: rò rỉ nặng (thí nghiệm 0.795 so với 0.545). Đưa vào Pipeline.
4. `train_test_split` thiếu `stratify=y` và `random_state`.
5. `scoring="accuracy"` với dữ liệu mất cân bằng: dùng `"roc_auc"` hoặc `"average_precision"`.
6. Vòng `for` chọn `C` bằng **test set**: tune trên test.
7. `pd.get_dummies` không xử lý được giá trị phân loại lạ lúc chạy thật, và các cột có thể lệch giữa lúc train và lúc phục vụ: dùng `OneHotEncoder(handle_unknown="ignore")`.
8. Model lưu ra không chứa bước tiền xử lý: phải lưu **cả Pipeline**.

Bản sửa:

```python
X, y = df[FEATURES], df["bad"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
pipe = Pipeline([("pre", build_preprocessor("linear")), ("sel", SelectKBest(f_classif, k=10)),
                 ("clf", LogisticRegression(max_iter=2000))])
search = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]}, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0))
search.fit(X_tr, y_tr)
print(roc_auc_score(y_te, search.predict_proba(X_te)[:, 1]))     # đo test ĐÚNG MỘT LẦN
joblib.dump(search.best_estimator_, "model.joblib")              # cả pipeline
```

**T3**

(a) Số lần fit $= 3\,600 \times 8/30 = 960$; mỗi cấu hình cần 5 lần, nên có **192** cấu hình. (Mỗi fold chỉ train trên 80% dữ liệu nên nhanh hơn một chút; ước lượng trên là bảo thủ.) *(1đ)*

(b) *(1đ)*
- `learning_rate`: `loguniform(0.01, 0.3)`, thang **log**.
- `max_leaf_nodes`: `randint(4, 64)`; có thể dùng log, vì từ 4 lên 8 là thay đổi lớn hơn từ 60 lên 64.
- `min_samples_leaf`: `randint(5, 200)` hoặc log.
- `max_iter`: **không tune trực tiếp**; đặt lớn (ví dụ 1 000) và dùng `early_stopping=True` để tự dừng.

(c) Hai cách: *(1đ)*
- **Successive halving / Hyperband** (`HalvingRandomSearchCV`): thử nhiều cấu hình trên tập con nhỏ hoặc với ít vòng boosting, chỉ giữ lại phần tốt nhất.
- **Early stopping** để không phí thời gian vào các vòng thừa.

Ngoài ra có thể dùng 3-fold thay cho 5 ở vòng lọc thô, hoặc Bayesian optimization (Optuna) để chọn điểm thử thông minh hơn.

(d) Tune bằng CV **chỉ trên train**; fit lại cấu hình tốt nhất trên toàn bộ train; đo **một lần** trên test giữ kín; báo cáo kèm khoảng tin cậy (bootstrap). Nếu dữ liệu ít thì dùng nested CV. *(1đ)*

**T4** *(đã chạy kiểm chứng: khớp `sklearn.metrics.silhouette_score` tới $10^{-9}$ trên 5 bộ dữ liệu ngẫu nhiên, kể cả có cụm 1 điểm)*

```python
def silhouette(X, labels):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))     # (n, n)
    ks = np.unique(labels)
    n = len(X)
    S = np.stack([D[:, labels == k].sum(1) for k in ks], axis=1)            # (n, K): tổng khoảng cách tới mỗi cụm
    counts = np.array([(labels == k).sum() for k in ks])
    own = np.searchsorted(ks, labels)
    a = S[np.arange(n), own] / np.maximum(counts[own] - 1, 1)              # không tính chính nó (D[i,i] = 0)
    mean_other = S / counts
    mean_other[np.arange(n), own] = np.inf
    b = mean_other.min(1)
    s = np.where(counts[own] > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(s.mean())
```

Bộ nhớ: ma trận $n \times n$ float64 với $n = 10^5$ là $10^{10} \times 8$ B = **80 GB**. Có hai cách xử lý:
- Tính trên một **mẫu ngẫu nhiên** (ví dụ 10 000 điểm, tốn 0.8 GB; scikit-learn có tham số `sample_size`).
- Tính **theo khối hàng** (chunk), mỗi khối $b \times n$, để bộ nhớ chỉ còn $O(bn)$.

*(Chấm: code đúng được 2.5đ · xử lý cụm 1 điểm được 0.5đ · ước lượng bộ nhớ và cách xử lý được 1đ.)*

</details>
