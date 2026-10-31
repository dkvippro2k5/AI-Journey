import pandas as pd

from optlab.cli import exp_lr_sweep, exp_optimizers, exp_rosenbrock, exp_scaling, main


def test_lr_sweep_diverges_only_above_2_over_L():
    df, curves = exp_lr_sweep()
    assert list(df.columns) == ["lr_over_L", "lr", "steps", "reached_tol", "final_loss", "diverged"]
    assert df["diverged"].tolist() == [False, False, False, False, True]
    s = df.set_index("lr_over_L")["steps"]
    assert s[0.1] > s[0.5] > s[1.0]
    assert len(curves) == 5


def test_scaling_shows_conditioning_effect():
    df, _ = exp_scaling()
    raw, std = df.set_index("setting").loc["raw"], df.set_index("setting").loc["standardized"]
    assert raw["kappa"] > 1000 and not raw["reached_tol"]
    assert std["kappa"] < 1.1 and std["reached_tol"] and std["steps"] < 10
    assert raw["predicted_steps"] > 5000                           # lý thuyết dự đoán cần > 5000 bước → đúng là không kịp


def test_optimizers_table():
    df, curves = exp_optimizers(seeds=range(3), epochs=2)
    assert list(df.columns) == ["optimizer", "lr", "mean_excess", "ci_lo", "ci_hi"]
    assert df["optimizer"].tolist() == ["sgd", "momentum", "rmsprop", "adam"]
    assert (df["ci_lo"] <= df["mean_excess"]).all() and (df["mean_excess"] <= df["ci_hi"]).all()
    assert (df["mean_excess"] > -1e-9).all()                       # không optimizer nào thấp hơn tối ưu
    assert set(curves) == {"sgd", "momentum", "rmsprop", "adam"}


def test_rosenbrock_momentum_and_adam_reach_minimum():
    df, paths = exp_rosenbrock(epochs=3000)
    d = df.set_index("optimizer")["final_dist"]
    assert d["momentum"] < 1e-3 and d["adam"] < 1e-3 and d["sgd"] > 0.1
    assert paths["adam"].shape == (3001, 2)


def test_main_writes_csv(tmp_path, capsys):
    main(["lr-sweep", "--out", str(tmp_path), "--no-plot"])
    df = pd.read_csv(tmp_path / "lr-sweep.csv")
    assert len(df) == 5
    assert "lr-sweep" in capsys.readouterr().out
