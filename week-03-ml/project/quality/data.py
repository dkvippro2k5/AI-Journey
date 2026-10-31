"""Dữ liệu tổng hợp: log hội thoại của chatbot và đánh giá 👎 của người dùng (cho sẵn, không cần sửa).

Mỗi dòng là một lượt trả lời. Cột mục tiêu `bad` = 1 nếu người dùng bấm 👎.
Dữ liệu được sinh có chủ đích: có giá trị thiếu, cột ID, một cột rò rỉ (leakage),
và quan hệ phi tuyến/tương tác mà model tuyến tính khó bắt được.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTENTS = ["faq", "billing", "tech_support", "chitchat", "complaint"]
MODELS = ["small", "medium", "large"]
FEATURES_NUM = ["n_turns", "query_tokens", "retrieval_score", "n_docs", "latency_s", "answer_tokens", "hour"]
FEATURES_CAT = ["intent", "model"]
TARGET = "bad"
DROP = ["session_id", "escalated"]      # cột ID và cột rò rỉ — KHÔNG được dùng làm đặc trưng


def make_feedback(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intent = rng.choice(INTENTS, n, p=[0.35, 0.15, 0.25, 0.15, 0.10])
    model = rng.choice(MODELS, n, p=[0.5, 0.3, 0.2])
    n_turns = rng.geometric(0.3, n)
    query_tokens = np.round(rng.lognormal(3.0, 0.7, n)).astype(int) + 1
    uses_rag = intent != "chitchat"
    n_docs = np.where(uses_rag, rng.integers(1, 11, n), 0)
    retrieval_score = np.where(uses_rag, rng.beta(5, 2, n), np.nan)
    size = pd.Series(model).map({"small": 0.0, "medium": 1.0, "large": 2.0}).to_numpy()
    answer_tokens = np.round(rng.lognormal(5.0 + 0.3 * (intent == "tech_support"), 0.5, n)).astype(int)
    latency_s = rng.lognormal(0.2 + 0.45 * size + 0.002 * answer_tokens / 10, 0.45, n)
    hour = rng.integers(0, 24, n)

    # log-odds của việc bị 👎 (ẩn với người học)
    z = -4.0 + 1.0 * (intent == "complaint")
    rs = np.nan_to_num(retrieval_score, nan=0.7)
    z += np.where(np.isin(intent, ["faq", "tech_support", "billing"]), 12.0 * np.maximum(0.0, 0.6 - rs), 0.0)
    z += 2.5 * (latency_s > 5.0)                                         # hiệu ứng ngưỡng
    z += 2.2 * ((model == "small") & (intent == "tech_support"))         # tương tác
    z += 2.0 * ((intent == "chitchat") & (answer_tokens > 250))          # tương tác
    z += 0.3 * np.minimum(n_turns, 8)
    z += 0.8 * ((hour >= 0) & (hour < 6))
    p_true = 1 / (1 + np.exp(-z))
    bad = (rng.random(n) < p_true).astype(int)

    # rò rỉ: phiên bị chuyển cho nhân viên SAU khi người dùng bấm 👎
    escalated = np.where(bad == 1, rng.random(n) < 0.6, rng.random(n) < 0.03).astype(int)

    # thiếu dữ liệu ngẫu nhiên do lỗi ghi log
    retrieval_score = np.where(rng.random(n) < 0.05, np.nan, retrieval_score)
    latency_s = np.where(rng.random(n) < 0.03, np.nan, latency_s)

    df = pd.DataFrame({
        "session_id": [f"s{seed}_{i:05d}" for i in range(n)],
        "intent": intent, "model": model, "n_turns": n_turns, "query_tokens": query_tokens,
        "retrieval_score": np.round(retrieval_score, 3), "n_docs": n_docs, "latency_s": np.round(latency_s, 2),
        "answer_tokens": answer_tokens, "hour": hour, "escalated": escalated, "bad": bad,
    })
    df.attrs["p_true"] = p_true          # xác suất thật — chỉ để tính "trần" AUC trong REPORT, không dùng để train
    return df
