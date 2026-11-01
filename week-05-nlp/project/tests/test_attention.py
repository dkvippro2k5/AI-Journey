import math

import pytest
import torch
from torch import nn

from intent.attention import MultiHeadAttention, apply_rope, causal_mask, padding_mask, sinusoidal_pe


def test_masks():
    assert causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    assert padding_mask(torch.tensor([2, 3]), 3).tolist() == [[True, True, False], [True, True, True]]


def test_sinusoidal_values():
    pe = sinusoidal_pe(3, 4)
    assert pe[0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert pe[1].tolist() == pytest.approx([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)], abs=1e-6)
    assert pe[2, 2].item() == pytest.approx(math.sin(0.02), abs=1e-6)


def test_rope_keeps_norm_and_depends_on_relative_position():
    torch.manual_seed(0)
    q, k = torch.randn(1, 8), torch.randn(1, 8)
    T = 10
    Q, K = apply_rope(q.expand(T, 8).clone()), apply_rope(k.expand(T, 8).clone())
    assert torch.allclose(Q.norm(dim=-1), q.norm(), atol=1e-5)               # phép xoay giữ độ dài
    assert torch.allclose(Q[0], q[0])                                         # vị trí 0: góc 0
    # q ở vị trí m, k ở vị trí n: tích vô hướng chỉ phụ thuộc m - n
    assert (Q[5] @ K[3]).item() == pytest.approx((Q[7] @ K[5]).item(), abs=1e-5)
    assert (Q[5] @ K[3]).item() != pytest.approx((Q[5] @ K[4]).item(), abs=1e-3)


def _copy_from_torch(mine: MultiHeadAttention, ref: nn.MultiheadAttention):
    d = ref.embed_dim
    with torch.no_grad():
        for i, lin in enumerate((mine.q_proj, mine.k_proj, mine.v_proj)):
            lin.weight.copy_(ref.in_proj_weight[i * d:(i + 1) * d])
            lin.bias.copy_(ref.in_proj_bias[i * d:(i + 1) * d])
        mine.out_proj.weight.copy_(ref.out_proj.weight)
        mine.out_proj.bias.copy_(ref.out_proj.bias)


def test_matches_torch_multihead_attention_with_padding():
    torch.manual_seed(1)
    d, h, B, T = 16, 4, 3, 5
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    mine = MultiHeadAttention(d, h)
    _copy_from_torch(mine, ref)
    x = torch.randn(B, T, d)
    keep = padding_mask(torch.tensor([5, 3, 1]), T)
    out_ref, w_ref = ref(x, x, x, key_padding_mask=~keep, average_attn_weights=False)   # torch: True = BỊ CHE
    out, w = mine(x, mask=keep[:, None, None, :])                                       # mình: True = được nhìn
    assert torch.allclose(out, out_ref, atol=1e-5)
    assert torch.allclose(w, w_ref, atol=1e-6)
    assert w[1, :, :, 3:].abs().max() == 0                                               # không nhìn token đệm


def test_causal_mask_blocks_the_future():
    torch.manual_seed(2)
    mha = MultiHeadAttention(8, 2).eval()
    x = torch.randn(1, 6, 8)
    y = x.clone()
    y[0, 4:] = torch.randn(2, 8)                                              # đổi tương lai (vị trí 4, 5)
    a, w = mha(x, mask=causal_mask(6))
    b, _ = mha(y, mask=causal_mask(6))
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-6) and not torch.allclose(a[0, 4:], b[0, 4:])
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 6))
    assert torch.triu(w[0, 0], diagonal=1).abs().max() == 0


def test_cross_attention_shapes_and_params():
    mha = MultiHeadAttention(12, 3)
    out, w = mha(torch.randn(2, 4, 12), kv=torch.randn(2, 7, 12))
    assert out.shape == (2, 4, 12) and w.shape == (2, 3, 4, 7)
    assert sum(p.numel() for p in mha.parameters()) == 4 * (12 * 12 + 12)
    with pytest.raises(AssertionError):
        MultiHeadAttention(10, 3)
