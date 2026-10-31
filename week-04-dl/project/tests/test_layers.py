import math

import pytest
import torch
from torch import nn

from qnet.layers import BatchNorm1d, Dropout, LayerNorm, RMSNorm, activation_stds, init_, layer_grad_norms


def test_dropout_train_eval_and_scaling():
    torch.manual_seed(0)
    d = Dropout(0.3)
    x = torch.ones(200_000)
    y = d(x)
    assert (y == 0).float().mean().item() == pytest.approx(0.3, abs=0.01)
    assert torch.allclose(y[y != 0], torch.tensor(1 / 0.7))                  # inverted: phần giữ lại chia (1 - p)
    assert y.mean().item() == pytest.approx(1.0, abs=0.01)                   # kỳ vọng được bảo toàn
    d.eval()
    assert torch.equal(d(x), x)
    assert torch.equal(Dropout(0.0)(x), x)
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_layernorm_matches_torch_forward_and_backward():
    torch.manual_seed(1)
    x = torch.randn(4, 5, 8, dtype=torch.float64, requires_grad=True)
    mine, ref = LayerNorm(8).double(), nn.LayerNorm(8).double()
    with torch.no_grad():
        for m in (mine, ref):
            m.weight.copy_(torch.linspace(0.5, 2, 8))
            m.bias.copy_(torch.linspace(-1, 1, 8))
    a, b = mine(x), ref(x)
    assert torch.allclose(a, b, atol=1e-10)
    ga = torch.autograd.grad(a.pow(3).sum(), x)[0]
    gb = torch.autograd.grad(b.pow(3).sum(), x)[0]
    assert torch.allclose(ga, gb, atol=1e-8)


def test_layernorm_hand_example():
    out = LayerNorm(8, eps=0.0)(torch.tensor([[2.0, 4, 4, 4, 5, 5, 7, 9]]))  # ví dụ tính tay bài 05: μ = 5, σ = 2
    assert out.detach().flatten().tolist() == pytest.approx([-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2], abs=1e-6)


def test_rmsnorm_matches_torch():
    torch.manual_seed(2)
    x = torch.randn(6, 16)
    mine, ref = RMSNorm(16, eps=1e-6), nn.RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        mine.weight.copy_(torch.arange(16.0) / 8)
        ref.weight.copy_(torch.arange(16.0) / 8)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
    assert not any("bias" in n for n, _ in mine.named_parameters())


def test_batchnorm_matches_torch_train_and_eval():
    torch.manual_seed(3)
    mine, ref = BatchNorm1d(5, momentum=0.2), nn.BatchNorm1d(5, momentum=0.2)
    for _ in range(3):
        x = torch.randn(32, 5) * 3 + 1
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(mine.running_var, ref.running_var, atol=1e-5)      # running_var dùng var KHÔNG lệch
    mine.eval(), ref.eval()
    x = torch.randn(7, 5)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)
    assert {"running_mean", "running_var"} <= set(mine.state_dict())         # buffer nằm trong state_dict
    assert {n for n, _ in mine.named_parameters()} == {"weight", "bias"}


def _deep(depth=20, width=256, act=nn.ReLU):
    return nn.Sequential(*[m for _ in range(depth) for m in (nn.Linear(width, width), act())])


def test_init_std():
    torch.manual_seed(4)
    m = nn.Sequential(nn.Linear(500, 300), nn.ReLU(), nn.Linear(300, 100))
    init_(m, "he")
    assert m[0].weight.std().item() == pytest.approx(math.sqrt(2 / 500), rel=0.02)
    assert m[0].bias.abs().sum().item() == 0
    init_(m, "xavier")
    assert m[2].weight.std().item() == pytest.approx(math.sqrt(2 / 400), rel=0.03)
    with pytest.raises(ValueError):
        init_(m, "lecun-uniform")


def test_activation_stds_he_vs_small():
    torch.manual_seed(5)
    x = torch.randn(512, 256)
    he = activation_stds(init_(_deep(), "he"), x)
    small = activation_stds(init_(_deep(), "small"), x)
    assert len(he) == 20
    assert 0.3 < he[-1] / he[0] < 3                                          # He giữ được độ lớn qua 20 lớp
    assert small[-1] < 1e-12 < small[0]                                       # std 0.01 → tín hiệu tắt dần
    m = init_(_deep(3), "he")
    activation_stds(m, x)
    assert all(len(mod._forward_hooks) == 0 for mod in m.modules())          # đã gỡ hook


def test_vanishing_gradient_sigmoid_vs_relu():
    torch.manual_seed(6)
    x, y = torch.randn(256, 64), torch.randn(256, 1)

    def net(act):
        layers = [m for _ in range(10) for m in (nn.Linear(64, 64), act())] + [nn.Linear(64, 1)]
        return init_(nn.Sequential(*layers), "xavier")

    sig = net(nn.Sigmoid)
    g_sig = layer_grad_norms(sig, nn.MSELoss()(sig(x), y))
    relu = init_(nn.Sequential(*[m for _ in range(10) for m in (nn.Linear(64, 64), nn.ReLU())], nn.Linear(64, 1)), "he")
    g_relu = layer_grad_norms(relu, nn.MSELoss()(relu(x), y))
    assert len(g_sig) == 11
    assert g_sig[0] / g_sig[-2] < 1e-3                                        # lớp đầu nhận gradient nhỏ hơn hàng nghìn lần
    assert g_relu[0] / g_relu[-2] > 0.01
