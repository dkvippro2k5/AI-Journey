import unicodedata

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from intent.data import make_messages
from intent.textprep import Tfidf, char_ngrams, normalize, strip_accents, syllables, word_ngrams


def test_strip_accents():
    assert strip_accents("Đổi sim ở đâu?") == "Doi sim o dau?"
    assert strip_accents("hủy gói") == strip_accents(unicodedata.normalize("NFD", "hủy gói")) == "huy goi"


def test_normalize_unicode_case_and_tone_style():
    nfd = unicodedata.normalize("NFD", "Huỷ gói GX30 😡")
    assert normalize(nfd) == normalize("hủy gói gx30") == "hủy gói gx30"          # NFC, chữ thường, cùng kiểu dấu
    assert normalize("Hoá đơn tháng 3!!") == "hóa đơn tháng 3"


def test_normalize_placeholders_and_teencode():
    assert normalize("nạp 200.000đ cho số 0868 123 456 nhé") == "nạp <money> cho số <phone> nhé"
    assert normalize("bị trừ 50k vô lý") == "bị trừ <money> vô lý"
    assert normalize("mở trang https://example.com/a và gửi an@example.com") == "mở trang <url> và gửi <email>"
    assert normalize("mk cần nv gọi lại, ko dc à") == "mình cần nhân viên gọi lại không được à"
    assert normalize("mk ko dc", teencode=False) == "mk ko dc"
    assert normalize("Mình KHÔNG được", strip=True) == "minh khong duoc"


def test_syllables_and_word_ngrams():
    assert syllables("nạp <money> cho số <phone>") == ["nạp", "<money>", "cho", "số", "<phone>"]
    assert word_ngrams(["a", "b", "c"], 1, 2) == ["a", "b", "c", "a b", "b c"]
    assert word_ngrams(["a"], 2, 3) == []


def test_char_ngrams_hand_example():
    # " sim " có 5 ký tự (kể cả 2 biên)
    assert char_ngrams("sim", 2, 3) == [" s", "si", "im", "m ", " si", "sim", "im "]
    assert char_ngrams("a", 2, 4) == [" a", "a ", " a "]                            # n = 3 đã bằng cả từ → dừng
    assert char_ngrams("a", 4, 5) == [" a "]                                        # từ ngắn hơn n: lấy cả từ một lần


def test_char_ngrams_match_sklearn_char_wb():
    docs = [normalize(t) for t in make_messages(500, seed=3).text]
    sk = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False).build_analyzer()
    for d in docs:
        assert char_ngrams(d, 2, 5) == sk(d)


def test_tfidf_hand_example():
    docs = ["hủy gói", "đăng ký gói", "hủy hủy"]
    tf = Tfidf(syllables).fit(docs)
    assert list(tf.vocabulary_) == ["gói", "hủy", "ký", "đăng"]                     # sắp theo thứ tự chuỗi
    idf = dict(zip(tf.vocabulary_, tf.idf_))
    assert idf["gói"] == pytest.approx(np.log(4 / 3) + 1)                           # df = 2, N = 3
    assert idf["đăng"] == pytest.approx(np.log(4 / 2) + 1)                          # df = 1
    X = tf.transform(["hủy hủy", "xin chào"]).toarray()
    assert X[0].tolist() == pytest.approx([0, 1, 0, 0])                             # chỉ một từ → vector đơn vị
    assert X[1].tolist() == [0, 0, 0, 0]                                            # toàn từ lạ → hàng 0


@pytest.mark.parametrize("kind", ["char", "word"])
def test_tfidf_matches_sklearn(kind):
    docs = [normalize(t) for t in make_messages(1500, seed=4).text]
    analyzer = (lambda d: char_ngrams(d, 2, 4)) if kind == "char" else (lambda d: word_ngrams(syllables(d), 1, 2))
    mine = Tfidf(analyzer, min_df=2).fit(docs[:1000])
    sk = TfidfVectorizer(analyzer=analyzer, min_df=2).fit(docs[:1000])
    assert mine.vocabulary_ == sk.vocabulary_
    assert np.allclose(mine.idf_, sk.idf_)
    assert abs(mine.transform(docs[1000:]) - sk.transform(docs[1000:])).max() < 1e-12
