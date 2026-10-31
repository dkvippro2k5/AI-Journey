import pytest
import torch
from torch import nn

from qnet.loop import (count_params, evaluate, fit, load_checkpoint, make_loader, predict, save_checkpoint, set_seed,
                       train_one_epoch)


def _data(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=g)
    y = X @ torch.tensor([1.0, -2.0, 0.5]) + 0.1 * torch.randn(n, generator=g)
    return X, y


class Reg(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(p), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def test_make_loader_batches_and_shuffle():
    X, y = _data(10)
    batches = list(make_loader(X, y, batch_size=4))
    assert [len(b[1]) for b in batches] == [4, 4, 2]
    assert torch.equal(torch.cat([b[0] for b in batches]), X)                 # không shuffle → giữ thứ tự
    a = torch.cat([b[1] for b in make_loader(X, y, batch_size=4, shuffle=True, seed=3)])
    b = torch.cat([b[1] for b in make_loader(X, y, batch_size=4, shuffle=True, seed=3)])
    assert torch.equal(a, b) and not torch.equal(a, y)                        # cùng seed → cùng thứ tự


def test_train_one_epoch_returns_per_sample_mean():
    X, y = _data(10)
    model = Reg()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)                         # lr = 0: tham số không đổi
    got = train_one_epoch(model, make_loader(X, y, batch_size=4), nn.MSELoss(), opt)
    with torch.no_grad():
        full = nn.MSELoss()(model(X), y).item()
    assert got == pytest.approx(full, rel=1e-6)                               # batch cuối 2 mẫu vẫn tính đúng trọng số


def test_training_reduces_loss_and_clip_norm_limits_step():
    X, y = _data()
    set_seed(0)
    model = Reg()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = make_loader(X, y, batch_size=32, shuffle=True)
    first = train_one_epoch(model, loader, nn.MSELoss(), opt)
    for _ in range(30):
        last = train_one_epoch(model, loader, nn.MSELoss(), opt)
    assert last < 0.2 * first
    model2 = Reg()
    before = torch.cat([p.detach().flatten().clone() for p in model2.parameters()])
    sgd = torch.optim.SGD(model2.parameters(), lr=1.0)
    train_one_epoch(model2, make_loader(X, y * 100, batch_size=200), nn.MSELoss(), sgd, clip_norm=0.1)
    after = torch.cat([p.detach().flatten() for p in model2.parameters()])
    assert (after - before).norm().item() <= 0.1 + 1e-5                       # bước = lr · gradient đã cắt


def test_evaluate_and_predict_use_eval_mode_without_grad():
    X, y = _data(50)
    model = Reg(p=0.5)
    loader = make_loader(X, y, batch_size=16)
    a, b = evaluate(model, loader, nn.MSELoss()), evaluate(model, loader, nn.MSELoss())
    assert a == b                                                             # dropout tắt → tất định
    assert all(p.grad is None for p in model.parameters())
    out = predict(model, X, batch_size=7)
    assert out.shape == (50,) and not out.requires_grad
    model.eval()
    with torch.no_grad():
        assert torch.allclose(out, model(X))


def test_fit_early_stopping_and_restores_best():
    X, y = _data()
    set_seed(0)
    model = Reg()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)                         # val loss không đổi → không cải thiện
    h = fit(model, make_loader(X, y), make_loader(X, y), nn.MSELoss(), opt, max_epochs=50, patience=4)
    assert h["best_epoch"] == 0 and h["stopped_epoch"] == 4 and len(h["val_loss"]) == 5
    Xv, yv = _data(100, seed=1)
    set_seed(0)
    model = Reg()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    h = fit(model, make_loader(X, y, shuffle=True), make_loader(Xv, yv), nn.MSELoss(), opt, max_epochs=40, patience=5)
    assert len(h["train_loss"]) == len(h["val_loss"]) == h["stopped_epoch"] + 1
    assert h["best_val"] == pytest.approx(min(h["val_loss"]))
    assert evaluate(model, make_loader(Xv, yv), nn.MSELoss()) == pytest.approx(h["best_val"], rel=1e-5)


def test_fit_is_reproducible_with_seed():
    X, y = _data()
    runs = []
    for _ in range(2):
        set_seed(7)
        model = Reg(p=0.2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        runs.append(fit(model, make_loader(X, y, shuffle=True, seed=7), make_loader(X, y), nn.MSELoss(), opt,
                        max_epochs=5)["train_loss"])
    assert runs[0] == runs[1]


def test_checkpoint_roundtrip(tmp_path):
    X, y = _data()
    set_seed(0)
    model = Reg()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    path = tmp_path / "best.pt"
    h = fit(model, make_loader(X, y, shuffle=True), make_loader(X, y), nn.MSELoss(), opt, max_epochs=8, ckpt_path=path)
    fresh = Reg()
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    state = load_checkpoint(path, fresh, fresh_opt)
    assert state["epoch"] == h["best_epoch"] and state["best_val"] == pytest.approx(h["best_val"])
    assert torch.allclose(predict(fresh, X), predict(model, X))
    assert len(fresh_opt.state_dict()["state"]) > 0                          # trạng thái Adam (m, v) được khôi phục
    save_checkpoint(tmp_path / "m.pt", model)                                 # không có optimizer vẫn lưu được
    assert "optimizer" not in torch.load(tmp_path / "m.pt", weights_only=True)


def test_count_params():
    m = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    assert count_params(m) == 55 + 6
    m[0].requires_grad_(False)
    assert count_params(m) == 6 and count_params(m, trainable_only=False) == 61
