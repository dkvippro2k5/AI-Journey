import numpy as np
import pytest
from sklearn import metrics as skm

from quality.metrics import (average_precision, best_threshold, confusion, precision_recall_f1, roc_auc,
                             stratified_kfold)

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.6, 0.35, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])      # ví dụ tính tay của bài 04


def test_confusion_and_prf_hand_example():
    pred = (S >= 0.5).astype(int)
    assert confusion(Y, pred).tolist() == [[4, 2], [1, 3]]
    p, r, f1 = precision_recall_f1(Y, pred)
    assert (p, r) == pytest.approx((0.6, 0.75)) and f1 == pytest.approx(2 / 3)


def test_prf_zero_division():
    assert precision_recall_f1([0, 1, 1], [0, 0, 0]) == (0.0, 0.0, 0.0)


def test_auc_and_ap_hand_example():
    assert roc_auc(Y, S) == pytest.approx(20 / 24)
    assert average_precision(Y, S) == pytest.approx((1 + 1 + 3 / 4 + 4 / 7) / 4)


def test_auc_and_ap_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 300)
        s = np.round(rng.random(300) + 0.3 * y, 1)                    # làm tròn để có nhiều giá trị hòa
        assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
        assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.3, 0.4])


def test_best_threshold_by_cost():
    assert best_threshold(Y, S, cost_fp=1, cost_fn=5) == (0.35, 3.0)
    assert best_threshold(Y, S, cost_fp=1, cost_fn=1) == (0.8, 2.0)    # hòa cost 2 ở 0.8 và 0.6 → chọn ngưỡng lớn hơn
    t, cost = best_threshold([0, 0, 1], [0.9, 0.8, 0.1], cost_fp=10, cost_fn=1)
    assert t == np.inf and cost == 1.0                                 # không dự đoán dương nào là rẻ nhất


def test_stratified_kfold_properties():
    y = np.array([1] * 30 + [0] * 70)
    folds = stratified_kfold(y, k=5, seed=0)
    assert len(folds) == 5
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests.tolist()) == list(range(100))                  # mỗi mẫu nằm trong đúng 1 fold test
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == 100
        assert y[te].sum() == 6 and len(te) == 20                      # đúng tỉ lệ 30%
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, stratified_kfold(y, 5, seed=0)))
    with pytest.raises(ValueError):
        stratified_kfold(y, k=1)
