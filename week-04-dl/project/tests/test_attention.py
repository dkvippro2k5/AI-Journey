import math

import pytest
import torch
import torch.nn.functional as F

from qnet.attention import AdditiveAttention, AttnPool, dot_attention


def test_dot_attention_hand_example():
    # ví dụ tính tay bài 09: 1 query, 3 key (d = 2), không scale
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0], [2.0], [3.0]])
    out, w = dot_attention(q, k, v, scale=False)
    e = torch.tensor([math.exp(2), math.exp(0), math.exp(1)])
    assert torch.allclose(w[0], e / e.sum())
    assert out.item() == pytest.approx((e @ torch.tensor([1.0, 2.0, 3.0]) / e.sum()).item())


def test_dot_attention_matches_sdpa_with_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 5)
    mask = torch.rand(2, 1, 4, 6) > 0.3
    mask[..., 0] = True                                                       # mỗi hàng còn ít nhất 1 vị trí
    out, w = dot_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert out.shape == (2, 3, 4, 5) and w.shape == (2, 3, 4, 6)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 4))
    assert (w.masked_select(~mask.expand_as(w)) == 0).all()                   # vị trí bị che có trọng số đúng bằng 0


def test_additive_attention_shapes_mask_and_hand_example():
    att = AdditiveAttention(d_q=2, d_k=2, d_h=2)
    with torch.no_grad():
        att.W_q.weight.copy_(torch.eye(2))
        att.W_k.weight.copy_(torch.eye(2))
        att.v.weight.copy_(torch.tensor([[1.0, 1.0]]))
    q = torch.tensor([[0.0, 0.0]])
    keys = torch.tensor([[[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]]])
    ctx, w = att(q, keys)
    s = torch.tensor([2 * math.tanh(1), -2 * math.tanh(1), 0.0])             # score_j = vᵀ tanh(q + k_j)
    assert torch.allclose(w[0], torch.softmax(s, 0))
    assert torch.allclose(ctx[0], (torch.softmax(s, 0).unsqueeze(1) * keys[0]).sum(0))
    _, w = att(q, keys, mask=torch.tensor([[False, True, True]]))
    assert w[0, 0] == 0 and w[0].sum().item() == pytest.approx(1.0)
    big = AdditiveAttention(5, 7, 4)
    ctx, w = big(torch.randn(3, 5), torch.randn(3, 9, 7))
    assert ctx.shape == (3, 7) and w.shape == (3, 9)


def test_attnpool_permutation_and_gradient():
    torch.manual_seed(1)
    pool = AttnPool(8)
    t = torch.randn(4, 6, 8, requires_grad=True)
    out, w = pool(t)
    assert out.shape == (4, 8) and w.shape == (4, 6)
    assert torch.allclose(w.sum(1), torch.ones(4))
    perm = torch.tensor([5, 2, 0, 1, 4, 3])
    out_p, w_p = pool(t[:, perm])
    assert torch.allclose(out_p, out, atol=1e-6)                             # thứ tự token không đổi kết quả gộp
    assert torch.allclose(w_p, w[:, perm])                                    # trọng số đi theo token
    out.sum().backward()
    assert pool.query.grad is not None and pool.query.grad.abs().sum() > 0
    assert {n for n, _ in pool.named_parameters()} == {"query", "W_k.weight", "W_v.weight"}
