import numpy as np
import pytest
import torch

from intent.data import INTENTS, make_messages
from intent.encoder import EncoderClassifier
from intent.metrics import prf
from intent.train import (FALLBACK, build_tfidf, choose_threshold, encoder_proba, random_split, route, scores,
                          template_split, train_encoder)


@pytest.fixture(scope="module")
def df():
    return make_messages(1500, seed=7)


def test_template_split_holds_out_whole_templates(df):
    tr, te = template_split(df, 0.2, seed=0)
    assert len(tr) + len(te) == len(df)
    assert not set(tr.template_id) & set(te.template_id)                         # không template nào ở cả hai
    origin = te.template_id.str.split("/").str[0]
    assert set(origin) == set(INTENTS)                                            # intent nào cũng có template test
    tr2, te2 = template_split(df, 0.2, seed=0)
    assert te2.index.equals(te.index)                                             # cùng seed → cùng split


def test_random_split_is_stratified(df):
    tr, te = random_split(df, 0.2, seed=0)
    assert len(te) == 300
    assert np.allclose(te.intent.value_counts(normalize=True).sort_index(),
                       df.intent.value_counts(normalize=True).sort_index(), atol=0.01)


def test_encoder_param_count_and_padding_invariance():
    torch.manual_seed(0)
    net = EncoderClassifier(vocab_size=100, n_classes=9).eval()
    assert sum(p.numel() for p in net.parameters()) == 64 * 100 + 67_657
    ids = torch.tensor([[2, 5, 6, 7], [2, 9, 0, 0]])
    padded = torch.cat([ids, torch.zeros(2, 3, dtype=torch.long)], 1)              # thêm 3 cột đệm
    with torch.no_grad():
        a, b = net(ids), net(padded)
    assert a.shape == (2, 9)
    assert torch.allclose(a, b, atol=1e-5)                                        # đệm không đổi kết quả


def test_choose_threshold_and_route():
    proba = np.full((4, len(INTENTS)), 0.0)
    for i, (k, c) in enumerate([(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]):
        proba[i, k] = c
        proba[i, -1] = 1 - c
    y = [INTENTS[0], INTENTS[5], INTENTS[2], INTENTS[3]]                         # mẫu thứ 2 bị đoán sai
    assert choose_threshold(proba, y, 0.95) == pytest.approx(0.9)
    assert choose_threshold(proba, y, 0.6) == pytest.approx(0.6)
    assert route(proba, 0.75).tolist() == [INTENTS[0], INTENTS[1], FALLBACK, FALLBACK]
    s = scores(y, proba, 0.75)
    assert s["coverage"] == 0.5 and s["routed_precision"] == 0.5 and s["acc"] == 0.75


def test_tfidf_baselines_learn(df):
    tr, te = random_split(df, 0.2, seed=0)
    for kind in ("word", "char"):
        model = build_tfidf(kind).fit(tr.text, tr.intent)
        assert prf(list(te.intent), list(model.predict(te.text)), labels=INTENTS)[2] > 0.9


def test_train_encoder_runs_and_learns(df):
    tr, te = random_split(df, 0.2, seed=0)
    model, bpe, hist = train_encoder(tr.text, tr.intent, te.text, te.intent, epochs=4, patience=10)
    assert len(hist["train_loss"]) == 4 and hist["train_loss"][-1] < hist["train_loss"][0]
    assert not model.training
    p = encoder_proba(model, bpe, te.text[:5])
    assert p.shape == (5, len(INTENTS)) and np.allclose(p.sum(1), 1, atol=1e-5)
