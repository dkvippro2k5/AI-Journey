"""Ngày 2 · Byte Pair Encoding (BPE) mức ký tự, kiểu Sennrich et al. (2016)."""
from __future__ import annotations

from collections import Counter

EOW = "</w>"                                   # ký hiệu cuối từ: phân biệt "an" cuối từ với "an" giữa từ
SPECIALS = ("<pad>", "<unk>", "<cls>")         # id 0, 1, 2


def _pair_counts(words: dict[tuple[str, ...], int]) -> Counter:
    pairs = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[a, b] += freq
    return pairs


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPE:
    def __init__(self, merges: list[tuple[str, str]], alphabet: list[str]):
        self.merges = merges
        self.ranks = {p: r for r, p in enumerate(merges)}
        tokens = list(SPECIALS) + sorted(set(alphabet) | {EOW}) + [a + b for a, b in merges]
        self.vocab = {t: i for i, t in enumerate(dict.fromkeys(tokens))}
        self.inv = {i: t for t, i in self.vocab.items()}
        self._cache: dict[str, list[str]] = {}

    def encode_word(self, word: str) -> list[str]:
        """Ghép theo thứ hạng: mỗi vòng ghép cặp kề nhau có hạng NHỎ NHẤT (học sớm nhất), tới khi hết cặp."""
        if word not in self._cache:
            symbols = tuple(word) + (EOW,)
            while len(symbols) > 1:
                pairs = [(self.ranks.get(p, len(self.ranks)), p) for p in zip(symbols, symbols[1:])]
                rank, best = min(pairs)
                if rank == len(self.ranks):
                    break
                symbols = _merge(symbols, best)
            self._cache[word] = list(symbols)
        return self._cache[word]

    def tokenize(self, text: str) -> list[str]:
        return [t for w in text.split() for t in self.encode_word(w)]

    def encode(self, text: str) -> list[int]:
        unk = self.vocab["<unk>"]
        return [self.vocab.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.inv[i] for i in ids if self.inv[i] not in SPECIALS)
        return text.replace(EOW, " ").strip()


def train_bpe(texts, num_merges: int, min_freq: int = 2) -> BPE:
    """Học tối đa `num_merges` phép ghép. Mỗi bước chọn cặp có tần suất lớn nhất;
    hòa thì chọn cặp nhỏ nhất theo thứ tự từ điển (để kết quả xác định). Dừng sớm nếu tần suất < min_freq."""
    word_freq = Counter(w for t in texts for w in t.split())
    words = {tuple(w) + (EOW,): f for w, f in word_freq.items()}
    alphabet = sorted({c for w in word_freq for c in w})
    merges = []
    for _ in range(num_merges):
        pairs = _pair_counts(words)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < min_freq:
            break
        merges.append(best)
        words = {_merge(s, best): f for s, f in words.items()}
    return BPE(merges, alphabet)
