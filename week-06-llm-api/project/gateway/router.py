"""Router intent của Tuần 5 (cho sẵn, KHÔNG cần sửa): `normalize` + TF-IDF âm tiết 1–2 gram + logistic regression,
cộng ngưỡng chuyển tiếp chọn trên tập val.

Gateway dùng router này làm tầng đầu, rẻ (dưới 1 ms mỗi tin nhắn): tin nhắn có độ tin cậy thấp được đẩy lên model lớn.
Router được train trên phần câu mẫu "đã biết" (`template_split`), nên khi gặp cách diễn đạt mới nó kém tự tin hơn,
giống ngoài đời.
"""
from __future__ import annotations

import functools
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gateway.data import INTENTS, make_messages

# viết tắt → dạng chuẩn (khóa đã ở dạng chữ thường)
TEENCODE = {"ko": "không", "k": "không", "hông": "không", "hok": "không", "dc": "được", "đc": "được", "đk": "được",
            "j": "gì", "r": "rồi", "rùi": "rồi", "v": "vậy", "mk": "mình", "mik": "mình", "t": "tôi", "tui": "tôi",
            "bn": "bao nhiêu", "bnh": "bao nhiêu", "vs": "với", "nha": "nhé", "nhe": "nhé", "ntn": "thế nào",
            "nv": "nhân viên", "tk": "tài khoản", "đt": "điện thoại", "bít": "biết", "júp": "giúp", "xiền": "tiền"}
# kiểu đặt dấu mới (dấu trên âm sau) → kiểu cũ (dấu trên âm đầu); chọn MỘT kiểu là đủ
TONE_FIX = {"uỷ": "ủy", "uý": "úy", "uỳ": "ùy", "uỹ": "ũy", "uỵ": "ụy", "oà": "òa", "oá": "óa", "oả": "ỏa",
            "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe"}

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(\+84|0)(\d[\s.]?){8,10}\d?")
_MONEY = re.compile(r"\b\d+([.,]\d{3})*\s*(k|nghìn|ngàn|đ|đồng|vnd)\b")
_TOKEN = re.compile(r"<\w+>|\w+")


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Đổi sim' → 'Doi sim'. Giữ nguyên ký tự không phải chữ có dấu."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in text if unicodedata.category(c) != "Mn"))


def normalize(text: str, *, strip: bool = False, teencode: bool = True) -> str:
    """NFC → chữ thường → thay URL/email/SĐT/tiền bằng placeholder → thống nhất kiểu dấu
    → bỏ dấu câu, emoji → (teencode) → (bỏ dấu) → gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _URL.sub(" <url> ", text)
    text = _EMAIL.sub(" <email> ", text)
    text = _PHONE.sub(" <phone> ", text)
    text = _MONEY.sub(" <money> ", text)
    for old, new in TONE_FIX.items():
        text = text.replace(old, new)
    tokens = _TOKEN.findall(text)                  # bỏ dấu câu và emoji
    if teencode:
        tokens = [TEENCODE.get(t, t) for t in tokens]
    text = " ".join(tokens)
    if strip:
        text = strip_accents(text)
    return text


def template_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Giữ lại ~test_frac số câu mẫu của mỗi intent (ít nhất 1) làm phần "chưa thấy" (Tuần 5, project)."""
    rng = np.random.default_rng(seed)
    origin = df["template_id"].str.split("/").str[0]
    held = []
    for name in INTENTS:
        temps = sorted(df.loc[origin == name, "template_id"].unique())
        held += list(rng.choice(temps, size=max(1, round(test_frac * len(temps))), replace=False))
    is_held = df["template_id"].isin(held)
    return df[~is_held], df[is_held]


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_precision: float = 0.95) -> float:
    """Ngưỡng NHỎ NHẤT trên độ tin cậy sao cho phần được nhận (conf ≥ ngưỡng) có accuracy ≥ mục tiêu; không có thì 1.0."""
    for t in np.unique(conf):
        if correct[conf >= t].mean() >= target_precision:
            return float(t)
    return 1.0


class IntentRouter:
    """`fit` trên tin nhắn có nhãn; `predict(texts)` trả về (intent, độ tin cậy); `confident(conf)` so với ngưỡng."""

    def __init__(self, C: float = 10.0):
        prep = functools.partial(normalize, strip=False)
        self.pipe = Pipeline([("tfidf", TfidfVectorizer(preprocessor=prep, token_pattern=r"<\w+>|\w+", ngram_range=(1, 2))),
                              ("clf", LogisticRegression(C=C, max_iter=3000))])
        self.threshold = 0.0

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, target_precision: float = 0.95):
        self.pipe.fit(train_df["text"], train_df["intent"])
        if val_df is not None:
            intents, conf = self.predict(val_df["text"].tolist())
            self.threshold = choose_threshold(conf, intents == val_df["intent"].to_numpy(), target_precision)
        return self

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        proba = self.pipe.predict_proba(texts)
        classes = self.pipe.classes_
        return classes[proba.argmax(1)], proba.max(1)

    def confident(self, conf) -> np.ndarray:
        return np.asarray(conf) >= self.threshold


@functools.lru_cache(maxsize=4)
def default_router(n: int = 6000, seed: int = 0, target_precision: float = 0.99) -> IntentRouter:
    """Router dùng chung: train trên câu mẫu "đã biết" (80%); chọn ngưỡng trên một mẫu MỚI gồm mọi câu mẫu, tức cùng
    phân phối với luồng request thật (precision mục tiêu 0.99, vì tin nhắn bị đẩy lên model lớn vẫn được xử lý)."""
    known, _ = template_split(make_messages(n, seed=seed), 0.2, seed=seed)
    return IntentRouter().fit(known, make_messages(2000, seed=seed + 500), target_precision)
