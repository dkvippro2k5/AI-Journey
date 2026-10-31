"""Ngày 1 · Vòng lặp huấn luyện chuẩn của PyTorch: DataLoader, train/eval, early stopping, checkpoint."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên: random, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(*tensors: torch.Tensor, batch_size: int = 128, shuffle: bool = False, seed: int = 0) -> DataLoader:
    """TensorDataset + DataLoader. Khi shuffle, thứ tự chỉ phụ thuộc `seed` (Generator riêng)."""
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer, *, clip_norm: float | None = None) -> float:
    """Một epoch: phần tử cuối của batch là nhãn, các phần tử còn lại là đầu vào của model.

    Trả về loss trung bình THEO MẪU (batch cuối có thể nhỏ hơn).
    """
    model.train()
    total, n = 0.0, 0
    for *xs, y in loader:
        optimizer.zero_grad()                                   # gradient cộng dồn → phải xóa mỗi bước
        loss = loss_fn(model(*xs), y)
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> float:
    """Loss trung bình theo mẫu, ở chế độ eval (dropout tắt, BatchNorm dùng running stats), không tính gradient."""
    model.eval()
    total, n = 0.0, 0
    for *xs, y in loader:
        total += loss_fn(model(*xs), y).item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model: nn.Module, *inputs: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Đầu ra thô (logit) của model trên toàn bộ `inputs`, theo từng batch, ở chế độ eval."""
    model.eval()
    n = len(inputs[0])
    return torch.cat([model(*(t[i:i + batch_size] for t in inputs)) for i in range(0, n, batch_size)])


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn, optimizer, *,
        max_epochs: int = 100, patience: int = 10, min_delta: float = 1e-4, clip_norm: float | None = None,
        scheduler=None, ckpt_path: str | Path | None = None) -> dict:
    """Huấn luyện có early stopping theo val loss; cuối cùng NẠP LẠI trọng số tốt nhất.

    "Cải thiện" nghĩa là val < best - min_delta. Dừng khi `patience` epoch liên tiếp không cải thiện.
    Mỗi lần cải thiện, nếu có `ckpt_path` thì lưu checkpoint (model + optimizer + epoch + best_val).
    """
    hist = {"train_loss": [], "val_loss": [], "best_epoch": -1, "best_val": float("inf"), "stopped_epoch": -1}
    best_state, wait = None, 0
    for epoch in range(max_epochs):
        hist["train_loss"].append(train_one_epoch(model, train_loader, loss_fn, optimizer, clip_norm=clip_norm))
        val = evaluate(model, val_loader, loss_fn)
        hist["val_loss"].append(val)
        if scheduler is not None:
            scheduler.step()
        hist["stopped_epoch"] = epoch
        if val < hist["best_val"] - min_delta:
            hist["best_val"], hist["best_epoch"], wait = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())      # bản sao THẬT, không phải tham chiếu
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_val=val)
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, **extra) -> None:
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()             # Adam có m, v — thiếu thì resume sẽ "nhảy"
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None) -> dict:
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
