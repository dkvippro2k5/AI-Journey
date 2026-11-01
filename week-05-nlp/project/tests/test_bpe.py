from intent.bpe import EOW, train_bpe
from intent.data import make_messages
from intent.textprep import normalize

CORPUS = ["goi goi goi goi goi", "gia gia gia", "gian gian", "han han han han"]


def test_merges_hand_example_with_ties():
    bpe = train_bpe(CORPUS, 6)
    # bước 1: ('a','n') và ('n','</w>') cùng 6 → chọn cặp nhỏ hơn theo thứ tự từ điển
    # bước 3: bốn cặp cùng 5 → ('g','i')
    assert bpe.merges == [("a", "n"), ("an", EOW), ("g", "i"), ("g", "o"), ("go", "i"), ("goi", EOW)]


def test_stops_when_pairs_too_rare():
    bpe = train_bpe(CORPUS, 100, min_freq=2)
    assert len(bpe.merges) < 100
    assert train_bpe(["ab"], 10, min_freq=2).merges == []                    # mỗi cặp chỉ xuất hiện 1 lần


def test_encode_word_by_rank_and_end_of_word():
    bpe = train_bpe(CORPUS, 6)
    assert bpe.encode_word("goi") == ["goi</w>"]
    assert bpe.encode_word("gian") == ["gi", "an</w>"]
    assert bpe.encode_word("giang") == ["gi", "an", "g", EOW]                 # "an" giữa từ ≠ "an</w>"
    assert bpe.encode_word("hoi") == ["h", "o", "i", EOW]                     # chưa từng học "oi"


def test_vocab_layout_unknown_and_roundtrip():
    bpe = train_bpe(CORPUS, 6)
    assert [bpe.vocab[s] for s in ("<pad>", "<unk>", "<cls>")] == [0, 1, 2]
    assert len(bpe.vocab) == 3 + 7 + 6                                        # specials + (6 ký tự + </w>) + 6 merge
    ids = bpe.encode("xin goi")
    assert ids[0] == bpe.vocab["<unk>"] and ids[-1] == bpe.vocab["goi</w>"]  # 'x' chưa thấy → <unk>
    assert bpe.decode(bpe.encode("gian han goi")) == "gian han goi"


def test_more_merges_fewer_tokens_on_real_data():
    texts = [normalize(t) for t in make_messages(1500, seed=5).text]
    small, big = train_bpe(texts[:1000], 50), train_bpe(texts[:1000], 500)
    n_small = sum(len(small.tokenize(t)) for t in texts[1000:])
    n_big = sum(len(big.tokenize(t)) for t in texts[1000:])
    assert n_big < n_small
    seen = set("".join(texts[:1000]))
    for t in texts[1000:1200]:
        if set(t.replace(" ", "")) <= seen:                                    # mọi ký tự đã thấy lúc train
            assert big.decode(big.encode(t)) == " ".join(t.split())           # không mất thông tin
