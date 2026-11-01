"""Ngày 1 · Tiền xử lý văn bản tiếng Việt và TF-IDF tự viết."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy import sparse

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


def syllables(text: str) -> list[str]:
    """Tách theo âm tiết (khoảng trắng); placeholder như <phone> là một token."""
    return _TOKEN.findall(text)


def char_ngrams(text: str, n_min: int = 2, n_max: int = 5) -> list[str]:
    """n-gram ký tự trong từng từ, có biên ' ' hai đầu, giống analyzer='char_wb' của scikit-learn."""
    out = []
    for w in text.split():
        w = f" {w} "
        for n in range(n_min, n_max + 1):
            if len(w) <= n:                         # từ ngắn (kể cả biên): lấy cả từ một lần rồi thôi
                out.append(w)
                break
            out.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return out


def word_ngrams(tokens: list[str], n_min: int = 1, n_max: int = 2) -> list[str]:
    """n-gram từ, nối bằng một dấu cách: ['a', 'b', 'c'] với (1, 2) → a, b, c, 'a b', 'b c'."""
    return [" ".join(tokens[i:i + n]) for n in range(n_min, n_max + 1) for i in range(len(tokens) - n + 1)]


class Tfidf:
    """TF-IDF giống TfidfVectorizer mặc định của scikit-learn: tf = số lần đếm,
    idf = ln((1 + N) / (1 + df)) + 1, rồi chuẩn hóa L2 từng hàng."""

    def __init__(self, analyzer, min_df: int = 1):
        self.analyzer = analyzer
        self.min_df = min_df

    def fit(self, docs) -> "Tfidf":
        df = Counter()
        for d in docs:
            df.update(set(self.analyzer(d)))
        terms = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        n = len(docs)
        self.idf_ = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in terms])
        return self

    def transform(self, docs) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            counts = Counter(t for t in self.analyzer(d) if t in self.vocabulary_)
            if not counts:
                continue
            idx = np.array([self.vocabulary_[t] for t in counts])
            v = np.array(list(counts.values()), dtype=float) * self.idf_[idx]
            v /= np.linalg.norm(v)
            rows += [r] * len(idx)
            cols += idx.tolist()
            vals += v.tolist()
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocabulary_)))

    def fit_transform(self, docs) -> sparse.csr_matrix:
        return self.fit(docs).transform(docs)
