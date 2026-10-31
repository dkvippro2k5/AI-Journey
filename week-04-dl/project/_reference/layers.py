"""Ngày 2 · Tự viết các lớp huấn luyện mạng sâu (đối chiếu với torch.nn trong test)."""
from __future__ import annotations

import math

import torch
from torch import nn


class Dropout(nn.Module):
    """Inverted dropout: lúc train giữ mỗi phần tử với xác suất 1 - p rồi chia cho (1 - p); lúc eval là hàm đồng nhất."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p phải thuộc [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)


class LayerNorm(nn.Module):
    """Chuẩn hóa theo CHIỀU CUỐI của từng mẫu (variance lệch, chia n), rồi nhân weight cộng bias."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps) * weight — không trừ trung bình, không có bias (Zhang & Sennrich 2019)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class BatchNorm1d(nn.Module):
    """BatchNorm cho đầu vào (N, d): chuẩn hóa theo CHIỀU BATCH của từng đặc trưng.

    Train: dùng mean/var (lệch) của batch; cập nhật running_mean/running_var (var KHÔNG lệch) với `momentum`.
    Eval: dùng running stats.
    """

    def __init__(self, d: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))     # buffer: nằm trong state_dict nhưng không phải tham số
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mu = x.mean(0)
            var = x.var(0, unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1 - m).add_(m * mu)
                self.running_var.mul_(1 - m).add_(m * x.var(0, unbiased=True))
        else:
            mu, var = self.running_mean, self.running_var
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


def init_(module: nn.Module, scheme: str = "he") -> nn.Module:
    """Khởi tạo lại MỌI nn.Linear bên trong `module` theo phân phối chuẩn; bias = 0.

    "he": std = sqrt(2 / fan_in) · "xavier": std = sqrt(2 / (fan_in + fan_out)) · "small": std = 0.01.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            fan_out, fan_in = m.weight.shape
            std = {"he": math.sqrt(2.0 / fan_in), "xavier": math.sqrt(2.0 / (fan_in + fan_out)), "small": 0.01}.get(scheme)
            if std is None:
                raise ValueError(f"scheme không hợp lệ: {scheme!r}")
            with torch.no_grad():
                m.weight.normal_(0.0, std)
                if m.bias is not None:
                    m.bias.zero_()
    return module


@torch.no_grad()
def activation_stds(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn đầu ra của từng nn.Linear (theo thứ tự forward), đo bằng forward hook."""
    stds: list[float] = []
    hooks = [m.register_forward_hook(lambda _m, _i, out: stds.append(out.std().item()))
             for m in model.modules() if isinstance(m, nn.Linear)]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()                                          # không gỡ hook thì lần forward sau vẫn bị ghi
    return stds


def layer_grad_norms(model: nn.Module, loss: torch.Tensor) -> list[float]:
    """Gọi backward trên `loss`, trả về chuẩn L2 gradient của weight từng nn.Linear (thứ tự đăng ký)."""
    model.zero_grad()
    loss.backward()
    return [m.weight.grad.norm().item() for m in model.modules() if isinstance(m, nn.Linear)]
