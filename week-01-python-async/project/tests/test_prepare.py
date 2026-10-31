from pathlib import Path

import pandas as pd
import pytest

from multi_caller.prepare import clean, load_prompts, summarize, write_jsonl
from multi_caller.utils import read_jsonl

RAW = Path(__file__).resolve().parents[1] / "data" / "prompts_raw.csv"


@pytest.fixture
def raw():
    return load_prompts(RAW)


def test_load_has_expected_columns(raw):
    assert list(raw.columns) == ["id", "prompt", "category"]
    assert len(raw) == 12


def test_clean_rules(raw):
    out = clean(raw)
    assert out["id"].tolist() == [1, 2, 5, 6, 8, 9, 10, 11]
    assert out.loc[out["id"] == 1, "prompt"].item() == "Giải thích attention trong Transformer bằng 3 câu"
    assert out.loc[out["id"] == 5, "category"].item() == "general"
    assert out.loc[out["id"] == 9, "category"].item() == "code"
    assert (out["n_chars"] == out["prompt"].str.len()).all()
    assert out.index.tolist() == list(range(len(out)))


def test_clean_does_not_mutate_input(raw):
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("max_chars, expected", [(10_000, 8), (45, 4), (30, 0)])
def test_clean_max_chars(raw, max_chars, expected):
    assert len(clean(raw, max_chars=max_chars)) == expected


def test_summarize(raw):
    s = summarize(clean(raw))
    assert s["total"] == 8
    assert s["by_category"] == {"code": 2, "explain": 3, "general": 2, "translate": 1}
    assert isinstance(s["mean_chars"], float)


def test_write_jsonl_roundtrip(raw, tmp_path):
    out = clean(raw)
    path = tmp_path / "prompts.jsonl"
    write_jsonl(out, path)
    rows = list(read_jsonl(path))
    assert rows[0] == {"id": 1, "prompt": out["prompt"].iloc[0], "category": "explain"}
    assert "Giải thích" in path.read_text(encoding="utf-8")
