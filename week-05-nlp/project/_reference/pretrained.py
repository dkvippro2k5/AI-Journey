"""Ngày 4 (tùy chọn, cần tải model ~470 MB) · Feature extraction từ một encoder pretrained + logistic regression.

MODEL_ID mặc định là một encoder đa ngôn ngữ nhỏ (~118M tham số) có tiếng Việt. Không có test nào cần file này.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "intfloat/multilingual-e5-small"
PREFIX = "query: "                         # model họ E5 được train với tiền tố này cho câu truy vấn


@torch.no_grad()
def embed(texts, model_id: str = MODEL_ID, batch_size: int = 64, prefix: str = PREFIX) -> np.ndarray:
    """Mean-pooling các vector token cuối (bỏ token đệm) rồi chuẩn hóa L2. Trả về (N, d)."""
    from transformers import AutoModel, AutoTokenizer      # import muộn: không cài/không tải thì phần khác vẫn chạy

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch = tok([prefix + t for t in texts[i:i + batch_size]], padding=True, truncation=True, max_length=64,
                    return_tensors="pt")
        h = model(**batch).last_hidden_state                     # (B, T, d)
        m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out)


class _Classifier(torch.nn.Module):
    def __init__(self, encoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1))


def finetune(train_texts, train_y, val_texts, val_y, labels, *, model_id: str = MODEL_ID, epochs: int = 4,
             lr: float = 3e-5, head_lr: float = 1e-3, batch_size: int = 32, seed: int = 0, prefix: str = PREFIX):
    """Fine-tune toàn bộ encoder + một lớp Linear trên mean-pooling. Discriminative LR (Tuần 4, bài 10):
    encoder lr nhỏ, head lr lớn. Chọn epoch có val macro-F1 tốt nhất. Trả về hàm predict_proba(texts)."""
    import copy

    from transformers import AutoModel, AutoTokenizer

    from intent.metrics import prf

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = _Classifier(AutoModel.from_pretrained(model_id), len(labels))
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": lr},
                             {"params": model.head.parameters(), "lr": head_lr}], weight_decay=0.01)
    lid = {l: i for i, l in enumerate(labels)}
    y = torch.tensor([lid[l] for l in train_y])
    train_texts = [prefix + t for t in train_texts]

    @torch.no_grad()
    def predict_proba(texts) -> np.ndarray:
        model.eval()
        out = []
        texts = [prefix + t for t in texts]
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=64, return_tensors="pt")
            out.append(torch.softmax(model(b["input_ids"], b["attention_mask"]), -1).numpy())
        return np.concatenate(out)

    g = torch.Generator().manual_seed(seed)
    best, best_state, hist = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_texts), generator=g).tolist()
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            b = tok([train_texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
            loss = torch.nn.functional.cross_entropy(model(b["input_ids"], b["attention_mask"]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = np.array(labels)[predict_proba(list(val_texts)).argmax(1)]
        f1 = prf(list(val_y), list(pred), labels=labels)[2]
        hist.append(f1)
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    predict_proba.history = hist
    return predict_proba
