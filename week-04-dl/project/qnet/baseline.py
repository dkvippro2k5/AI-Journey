"""Baseline của Tuần 3 (cho sẵn, không cần sửa): cùng tiền xử lý và tham số mặc định với `quality.pipeline`."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from qnet.data import FEATURES_CAT, FEATURES_NUM, TARGET


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Chỉ giữ các cột đặc trưng hợp lệ — loại cột ID và cột rò rỉ."""
    return df[FEATURES_NUM + FEATURES_CAT].copy(), df[TARGET].copy()


def build_baseline(name: str) -> Pipeline:
    """"logreg" hoặc "hgb", giống hệt `build_model` của Tuần 3."""
    if name == "logreg":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
        clf = LogisticRegression(max_iter=2000)
    elif name == "hgb":
        num = SimpleImputer(strategy="median", add_indicator=True)
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=7,
                                             min_samples_leaf=50, random_state=0)
    else:
        raise ValueError(f"baseline không hỗ trợ: {name!r}")
    pre = ColumnTransformer([("num", num, FEATURES_NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])
