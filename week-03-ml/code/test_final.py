"""Test phần C — final Tuần 3. Chạy từ thư mục tuần: python -m pytest -q code/test_final.py"""
import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from final_lib import apply_encoding, brier, calibration_table, ece, nested_cv, target_encode_cv

CATS = ["a", "a", "b", "b", "b", "c"]
Y = [1, 0, 1, 1, 0, 1]


# ---------- C1 ----------

def test_target_encoding_mapping_with_smoothing():
    _, mapping, prior = target_encode_cv(CATS, Y, k=3, m=2.0)
    assert prior == pytest.approx(4 / 6)
    assert mapping["a"] == pytest.approx((1 + 2 * 4 / 6) / 4)
    assert mapping["b"] == pytest.approx((2 + 2 * 4 / 6) / 5)
    assert mapping["c"] == pytest.approx((1 + 2 * 4 / 6) / 3)
    assert apply_encoding(["b", "z"], mapping, prior).tolist() == pytest.approx([mapping["b"], prior])


def test_target_encoding_is_out_of_fold():
    enc, _, _ = target_encode_cv(CATS, Y, k=6, m=2.0)          # k = n: mỗi dòng được mã hóa từ 5 dòng còn lại
    assert enc[0] == pytest.approx((0 + 2 * 0.6) / (1 + 2))     # dòng "a" (y=1): chỉ thấy "a" còn lại có y=0
    assert enc[5] == pytest.approx(0.6)                          # "c" chỉ xuất hiện một lần → dùng prior của fold


def test_target_encoding_prevents_leakage_on_id_like_column():
    rng = np.random.default_rng(0)
    n = 2000
    cats = [f"id{i}" for i in rng.integers(0, 1500, n)]          # gần như mỗi dòng một giá trị, không liên quan nhãn
    y = rng.integers(0, 2, n)
    oof, mapping, prior = target_encode_cv(cats, y, k=5, m=1.0)
    naive = apply_encoding(cats, mapping, prior)                  # mã hóa trên chính dữ liệu đã dùng để tính
    assert roc_auc_score(y, naive) > 0.85                         # ảo
    assert abs(roc_auc_score(y, oof) - 0.5) < 0.06                # đúng: cột ID không có thông tin


# ---------- C2 ----------

GRID = {"C": [0.01, 0.1, 1.0, 10.0]}


def test_nested_cv_structure():
    X, y = make_classification(600, 10, n_informative=8, random_state=0)
    est = LogisticRegression(max_iter=1000)
    res = nested_cv(est, GRID, X, y, outer_k=3, inner_k=3)
    assert set(res) == {"outer_scores", "inner_best", "best_params"}
    assert len(res["outer_scores"]) == len(res["inner_best"]) == len(res["best_params"]) == 3
    assert all(p["C"] in GRID["C"] for p in res["best_params"])
    assert np.mean(res["outer_scores"]) > 0.9
    assert not hasattr(est, "coef_")                              # estimator gốc không bị fit


def test_nested_cv_is_reproducible_and_uses_outer_k():
    X, y = make_classification(400, 10, n_informative=4, random_state=1)
    a = nested_cv(LogisticRegression(max_iter=1000), GRID, X, y, outer_k=4, inner_k=3, seed=7)
    b = nested_cv(LogisticRegression(max_iter=1000), GRID, X, y, outer_k=4, inner_k=3, seed=7)
    assert a["outer_scores"] == b["outer_scores"] and len(a["outer_scores"]) == 4
    assert all(0.5 < s <= 1 for s in a["outer_scores"])


# ---------- C3 ----------

def test_brier_hand_and_sklearn():
    assert brier([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3]) == pytest.approx(0.0375)
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 500), rng.random(500)
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_calibration_table_matches_sklearn():
    rng = np.random.default_rng(3)
    p = rng.random(5000)
    y = (rng.random(5000) < p ** 2).astype(int)                   # model quá tự tin: xác suất thật = p²
    t = calibration_table(y, p, n_bins=10)
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="uniform")
    assert list(t.columns) == ["bin", "count", "mean_pred", "frac_pos"]
    assert np.allclose(t["frac_pos"], frac) and np.allclose(t["mean_pred"], mean)
    assert t["count"].sum() == 5000


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(4)
    p = rng.random(20000)
    y_good = (rng.random(20000) < p).astype(int)
    y_bad = (rng.random(20000) < p ** 2).astype(int)
    assert ece(y_good, p) < 0.02
    assert ece(y_bad, p) > 0.1
    assert ece([1, 0], [1.0, 0.0]) == 0.0                         # p = 1.0 vẫn rơi vào bin cuối
