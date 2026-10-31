"""Ngày 4 · Tiền xử lý bảng → tensor, và hai mạng cho dữ liệu bảng: TabMLP và TabAttn."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from qnet.attention import AttnPool
from qnet.data import FEATURES_CAT, FEATURES_NUM
from qnet.layers import BatchNorm1d, Dropout, LayerNorm, init_


class TabularPrep:
    """Học thống kê CHỈ trên tập train, rồi biến DataFrame thành (x_num float32, x_cat int64).

    Số: điền median → chuẩn hóa (mean/std của train, std lệch) → nối thêm cột chỉ báo thiếu (0/1)
    cho những cột có NaN trong train. Phân loại: giá trị train sắp xếp → 1..K; giá trị lạ hoặc NaN → 0.
    """

    def __init__(self, num_cols: list[str] = FEATURES_NUM, cat_cols: list[str] = FEATURES_CAT):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)

    def fit(self, X: pd.DataFrame) -> "TabularPrep":
        num = X[self.num_cols].to_numpy(dtype=float)
        self.median_ = np.nanmedian(num, axis=0)
        filled = np.where(np.isnan(num), self.median_, num)
        self.mean_ = filled.mean(axis=0)
        self.std_ = filled.std(axis=0)
        self.std_[self.std_ == 0] = 1.0                           # cột hằng số: tránh chia 0
        self.missing_cols_ = [c for c, has in zip(self.num_cols, np.isnan(num).any(axis=0)) if has]
        self.vocab_ = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].dropna().unique()))} for c in self.cat_cols}
        return self

    @property
    def n_num(self) -> int:
        return len(self.num_cols) + len(self.missing_cols_)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab_[c]) + 1 for c in self.cat_cols]   # +1 cho chỉ số 0 = "không biết"

    @property
    def token_names(self) -> list[str]:
        return self.num_cols + [f"{c}_missing" for c in self.missing_cols_] + self.cat_cols

    def transform(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = X[self.num_cols].to_numpy(dtype=float)
        miss = np.isnan(X[self.missing_cols_].to_numpy(dtype=float)).astype(float)
        z = (np.where(np.isnan(num), self.median_, num) - self.mean_) / self.std_
        x_num = np.concatenate([z, miss], axis=1).astype(np.float32)
        x_cat = np.stack([X[c].map(self.vocab_[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.cat_cols], axis=1)
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)


class TabMLP(nn.Module):
    """[số ‖ embedding của từng cột phân loại] → (Linear → norm → ReLU → Dropout) × len(hidden) → Linear(1)."""

    def __init__(self, n_num: int, cardinalities: list[int], emb_dim: int = 4, hidden: tuple[int, ...] = (64, 64),
                 dropout: float = 0.1, norm: str = "none"):
        super().__init__()
        if norm not in ("none", "batch", "layer"):
            raise ValueError(f"norm không hợp lệ: {norm!r}")
        self.embs = nn.ModuleList(nn.Embedding(c, emb_dim) for c in cardinalities)
        d, layers = n_num + emb_dim * len(cardinalities), []
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if norm == "batch":
                layers.append(BatchNorm1d(h))
            elif norm == "layer":
                layers.append(LayerNorm(h))
            layers += [nn.ReLU(), Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        init_(self.body, "he")                                  # He cho ReLU (bài 04)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.head(self.body(torch.cat([x_num, *embs], dim=1))).squeeze(-1)    # logit, shape (B,)


class TabAttn(nn.Module):
    """Mỗi đặc trưng thành MỘT token d chiều, gộp bằng AttnPool, rồi qua head MLP.

    Token số i: x_i · w_i + b_i (mỗi cột có w_i, b_i riêng). Token phân loại: embedding d chiều.
    Trọng số attention cho biết model "nhìn" vào đặc trưng nào với từng mẫu.
    """

    def __init__(self, n_num: int, cardinalities: list[int], d: int = 16, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d) / d ** 0.5)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.embs = nn.ModuleList(nn.Embedding(c, d) for c in cardinalities)
        self.token_mlp = nn.Sequential(LayerNorm(d), nn.Linear(d, d), nn.ReLU())
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), Dropout(dropout), nn.Linear(hidden, 1))

    def tokens(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        t_num = x_num.unsqueeze(-1) * self.num_w + self.num_b                       # (B, n_num, d)
        t_cat = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.token_mlp(torch.cat([t_num, t_cat], dim=1))                    # (B, T, d)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pool(self.tokens(x_num, x_cat))
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def attention(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Trọng số attention (B, T), thứ tự token khớp `TabularPrep.token_names`."""
        self.eval()
        return self.pool(self.tokens(x_num, x_cat))[1]


def build_net(arch: str, prep: TabularPrep, **kw) -> nn.Module:
    if arch == "mlp":
        return TabMLP(prep.n_num, prep.cardinalities, **kw)
    if arch == "attn":
        return TabAttn(prep.n_num, prep.cardinalities, **kw)
    raise ValueError(f"arch không hỗ trợ: {arch!r}")
