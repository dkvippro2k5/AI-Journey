"""Ngày 4 · Vòng lặp huấn luyện chung cho mọi loss và optimizer."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from optlab.optim import Optimizer, clip_by_norm

LossFn = Callable[..., tuple[float, np.ndarray]]


@dataclass
class History:
    losses: list[float] = field(default_factory=list)   # loss trên TOÀN BỘ dữ liệu, trước epoch 1 và sau mỗi epoch
    dists: list[float] = field(default_factory=list)    # ‖θ − θ*‖ tại cùng các thời điểm (nếu có theta_star)
    thetas: list[np.ndarray] = field(default_factory=list)  # bản sao θ tại cùng các thời điểm (để vẽ quỹ đạo)
    steps: int = 0                                     # tổng số lần gọi optimizer.step
    diverged: bool = False
    reached_tol: bool = False


def train(loss_fn: LossFn, theta0: np.ndarray, X: np.ndarray | None, y: np.ndarray | None,
          opt: Optimizer, *, epochs: int, batch_size: int | None = None, seed: int = 0,
          clip: float | None = None, theta_star: np.ndarray | None = None, tol: float | None = None,
          max_loss: float = 1e6) -> tuple[np.ndarray, History]:
    """Huấn luyện `epochs` epoch. batch_size=None → full batch (1 bước/epoch).

    - Mỗi epoch xáo trộn dữ liệu bằng np.random.default_rng(seed) (tạo MỘT lần).
    - Dừng sớm, đặt diverged=True, nếu loss không hữu hạn hoặc > max_loss.
    - Dừng sớm, đặt reached_tol=True, nếu có theta_star, tol và ‖θ − θ*‖ < tol.
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=float)
    hist = History()

    def record() -> bool:
        loss, _ = loss_fn(theta, X, y)
        hist.losses.append(loss)
        hist.thetas.append(theta.copy())
        if theta_star is not None:
            hist.dists.append(float(np.linalg.norm(theta - theta_star)))
        if not math.isfinite(loss) or loss > max_loss:
            hist.diverged = True
            return True
        if tol is not None and hist.dists and hist.dists[-1] < tol:
            hist.reached_tol = True
            return True
        return False

    if record():
        return theta, hist
    n = 0 if X is None else len(X)
    for _ in range(epochs):
        if X is None or batch_size is None or batch_size >= n:
            batches = [None]
        else:
            perm = rng.permutation(n)
            batches = [perm[i:i + batch_size] for i in range(0, n, batch_size)]
        for idx in batches:
            if idx is None:
                _, g = loss_fn(theta, X, y)
            else:
                _, g = loss_fn(theta, X[idx], y[idx])
            if clip is not None:
                g = clip_by_norm(g, clip)
            theta = opt.step(theta, g)
            hist.steps += 1
        if record():
            break
    return theta, hist
