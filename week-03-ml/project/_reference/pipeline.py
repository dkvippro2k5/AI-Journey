"""Ngày 3 · Tiền xử lý và model dưới dạng scikit-learn Pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quality.data import FEATURES_CAT, FEATURES_NUM, TARGET


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Chỉ giữ các cột đặc trưng hợp lệ — loại cột ID và cột rò rỉ."""
    return df[FEATURES_NUM + FEATURES_CAT].copy(), df[TARGET].copy()


def build_preprocessor(kind: str) -> ColumnTransformer:
    if kind == "linear":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    elif kind == "tree":
        num = SimpleImputer(strategy="median", add_indicator=True)     # cây không cần chuẩn hóa
    else:
        raise ValueError(f"kind không hợp lệ: {kind!r}")
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, FEATURES_NUM), ("cat", cat, FEATURES_CAT)])


DEFAULTS = {
    "logreg": (LogisticRegression, "linear", {"max_iter": 2000}),
    "rf": (RandomForestClassifier, "tree", {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 0}),
    "hgb": (HistGradientBoostingClassifier, "tree",
            {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7, "min_samples_leaf": 50, "random_state": 0}),
}


def build_model(name: str, **params) -> Pipeline:
    """Pipeline gồm hai bước "pre" và "clf". `params` ghi đè tham số mặc định của bộ phân loại."""
    if name not in DEFAULTS:
        raise ValueError(f"model không hỗ trợ: {name!r}")
    cls, kind, defaults = DEFAULTS[name]
    return Pipeline([("pre", build_preprocessor(kind)), ("clf", cls(**{**defaults, **params}))])
