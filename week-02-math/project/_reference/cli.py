"""Ngày 4 · Bốn thí nghiệm tối ưu hóa, ghi bảng CSV và hình PNG.

Chạy: python -m optlab.cli [all|lr-sweep|scaling|optimizers|rosenbrock] [--out results] [--no-plot]
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from optlab.data import make_classification, make_regression, rosenbrock
from optlab.grad import logistic_loss, mse_loss
from optlab.linalg import add_bias, hessian_eigs_mse, lstsq, standardize
from optlab.optim import SGD, make_optimizer
from optlab.stats import mean_ci
from optlab.train import train

log = logging.getLogger("optlab")

LR_MULTIPLIERS = (0.1, 0.5, 1.0, 1.9, 2.1)
OPT_LRS = {"sgd": 0.1, "momentum": 0.1, "rmsprop": 0.01, "adam": 0.01}
ROSEN_LRS = {"sgd": 1e-3, "momentum": 1e-3, "rmsprop": 1e-2, "adam": 2e-2}


def exp_lr_sweep(seed: int = 0):
    """GD full batch trên MSE với lr = m/L. Lý thuyết: hội tụ khi và chỉ khi lr < 2/L."""
    X, y, _ = make_regression(500, 2, scales=[1, 3], seed=seed)
    Xb = add_bias(X)
    L = hessian_eigs_mse(Xb)[-1]
    theta_star = lstsq(Xb, y)
    rows, curves = [], {}
    for m in LR_MULTIPLIERS:
        _, h = train(mse_loss, np.zeros(Xb.shape[1]), Xb, y, SGD(m / L), epochs=2000,
                     theta_star=theta_star, tol=1e-6)
        rows.append({"lr_over_L": m, "lr": m / L, "steps": h.steps, "reached_tol": h.reached_tol,
                     "final_loss": h.losses[-1], "diverged": h.diverged})
        curves[f"lr={m}/L"] = h.losses
    return pd.DataFrame(rows), curves


def exp_scaling(seed: int = 0):
    """Cùng dữ liệu, trước và sau chuẩn hóa: κ của Hessian quyết định số bước của GD (lr = 1/L)."""
    X, y, _ = make_regression(500, 2, scales=[1, 100], offsets=[0, 50], seed=seed)
    rows, curves = [], {}
    for name, M in (("raw", add_bias(X)), ("standardized", add_bias(standardize(X)[0]))):
        ev = hessian_eigs_mse(M)
        kappa = float(ev[-1] / ev[0])
        theta_star = lstsq(M, y)
        _, h = train(mse_loss, np.zeros(M.shape[1]), M, y, SGD(1 / ev[-1]), epochs=5000,
                     theta_star=theta_star, tol=1e-3)
        rate = 1 - 1 / kappa                       # hệ số co của mode chậm nhất
        predicted = math.log(h.dists[0] / 1e-3) / -math.log(rate) if rate > 0 else 1.0
        rows.append({"setting": name, "kappa": kappa, "steps": h.steps, "reached_tol": h.reached_tol,
                     "final_dist": h.dists[-1], "predicted_steps": math.ceil(predicted)})
        curves[name] = h.dists
    return pd.DataFrame(rows), curves


def exp_optimizers(seeds=range(5), epochs: int = 5):
    """Logistic regression, mini-batch 32. Đo excess loss = loss − loss tối ưu, ghép cặp theo seed."""
    best = {}
    for s in seeds:
        X, y = make_classification(400, seed=s)
        _, h = train(logistic_loss, np.zeros(3), add_bias(X), y, SGD(1.0), epochs=2000)
        best[s] = h.losses[-1]
    rows, curves = [], {}
    for name, lr in OPT_LRS.items():
        excess = []
        for s in seeds:
            X, y = make_classification(400, seed=s)
            _, h = train(logistic_loss, np.zeros(3), add_bias(X), y, make_optimizer(name, lr),
                         epochs=epochs, batch_size=32, seed=s)
            excess.append(h.losses[-1] - best[s])
            if s == 0:
                curves[name] = h.losses
        m, lo, hi = mean_ci(excess)
        rows.append({"optimizer": name, "lr": lr, "mean_excess": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), curves


def exp_rosenbrock(epochs: int = 3000):
    """Thung lũng hẹp và cong: so sánh quỹ đạo của 4 optimizer từ (−1.5, 2)."""
    rows, paths = [], {}
    for name, lr in ROSEN_LRS.items():
        _, h = train(rosenbrock, np.array([-1.5, 2.0]), None, None, make_optimizer(name, lr),
                     epochs=epochs, theta_star=np.array([1.0, 1.0]))
        rows.append({"optimizer": name, "lr": lr, "final_loss": h.losses[-1],
                     "final_dist": h.dists[-1], "diverged": h.diverged})
        paths[name] = np.array(h.thetas)
    return pd.DataFrame(rows), paths


EXPERIMENTS = {"lr-sweep": exp_lr_sweep, "scaling": exp_scaling,
               "optimizers": exp_optimizers, "rosenbrock": exp_rosenbrock}


def plot(name: str, curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")                          # không cần màn hình
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if name == "rosenbrock":
        xs, ys = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
        ax.contour(xs, ys, (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2, levels=np.logspace(-1, 3, 15), cmap="gray")
        for label, p in curves.items():
            ax.plot(p[:, 0], p[:, 1], label=label, lw=1)
        ax.plot(1, 1, "k*", ms=10)
    else:
        for label, c in curves.items():
            ax.semilogy(np.maximum(np.asarray(c, dtype=float), 1e-16), label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("‖θ − θ*‖" if name == "scaling" else "loss")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="optlab")
    parser.add_argument("experiment", nargs="?", default="all", choices=["all", *EXPERIMENTS])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        for name in names:
            table, curves = EXPERIMENTS[name]()
            table.to_csv(args.out / f"{name}.csv", index=False)
            if not args.no_plot:
                plot(name, curves, args.out / f"{name}.png")
            log.info("đã ghi %s", args.out / f"{name}.csv")
            print(f"\n== {name} ==\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
