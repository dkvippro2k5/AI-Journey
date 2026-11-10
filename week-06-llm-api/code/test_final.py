"""Test phần C của final Tuần 6. Viết lời giải vào code/final_lib.py rồi chạy:
python -m pytest -q code/test_final.py   (ở thư mục week-06-llm-api)
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from final_lib import SlidingWindowLimiter, ToolCallAccumulator, allocate_budget, validate  # noqa: E402

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["reply", "intent", "slots"],
    "properties": {
        "reply": {"type": "string", "minLength": 1, "maxLength": 20},
        "intent": {"enum": ["huy_goi", "dang_ky_goi", "khac"]},
        "slots": {"type": "object", "additionalProperties": False, "properties": {
            "sdt": {"type": ["string", "null"], "pattern": "^0[0-9]{9}$"},
            "tien": {"type": ["integer", "null"], "minimum": 1000, "maximum": 10000000}}},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
}


# ---------- C1 · validator ----------
def test_validate_valid_and_types():
    ok = {"reply": "Dạ", "intent": "huy_goi", "slots": {"sdt": "0912345678", "tien": None}, "tags": ["a"]}
    assert validate(ok, SCHEMA) == []
    assert validate({**ok, "slots": {"sdt": None, "tien": 50000.0}}, SCHEMA) == []          # 50000.0 là integer
    assert validate("x", SCHEMA) == [("$", "type")]
    assert validate(True, {"type": "integer"}) == [("$", "type")]                            # bool không phải số
    assert validate(1.5, {"type": "integer"}) == [("$", "type")]
    assert validate(3, {"type": "number"}) == []


def test_validate_reports_paths_and_keywords():
    bad = {"reply": "", "intent": "hủy", "slots": {"sdt": "912345678", "tien": 50, "x": 1}, "tags": [1, "b", "c"],
           "extra": True}
    got = set(validate(bad, SCHEMA))
    assert got == {("$.reply", "minLength"), ("$.intent", "enum"), ("$.slots.sdt", "pattern"),
                   ("$.slots.tien", "minimum"), ("$.slots.x", "additionalProperties"), ("$.tags", "maxItems"),
                   ("$.tags[0]", "type"), ("$.extra", "additionalProperties")}


def test_validate_required_and_nested_type_stops_descent():
    got = validate({"intent": "khac", "slots": "không phải object"}, SCHEMA)
    assert set(got) == {("$.reply", "required"), ("$.slots", "type")}                       # không đi sâu vào slots
    assert validate({"reply": "x" * 21, "intent": "khac", "slots": {}}, SCHEMA) == [("$.reply", "maxLength")]
    assert validate({"a": 1}, {"type": "object"}) == []                                      # mặc định cho phép khóa thêm


# ---------- C2 · sliding window ----------
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_sliding_window_has_no_boundary_burst():
    c = Clock()
    lim = SlidingWindowLimiter(limit=3, window=60, clock=c)
    c.t = 59.0
    assert [lim.allow("u") for _ in range(4)] == [True, True, True, False]
    c.t = 61.0                                                                    # fixed window sẽ cho thêm 3 ở đây
    assert not lim.allow("u") and lim.retry_after("u") == pytest.approx(58.0)
    c.t = 119.0                                                                   # 59 + 60: ba request cũ vừa hết hạn
    assert lim.allow("u") and lim.allow("other")


def test_sliding_window_long_run_never_exceeds_limit():
    c, rng = Clock(), random.Random(0)
    lim = SlidingWindowLimiter(limit=5, window=10, clock=c)
    accepted = []
    for _ in range(3000):
        c.t += rng.expovariate(2.0)
        if lim.allow("u"):
            accepted.append(c.t)
    j = 0
    for i, t in enumerate(accepted):                                              # mọi cửa sổ (t-10, t] có ≤ 5
        while accepted[j] <= t - 10:
            j += 1
        assert i - j + 1 <= 5
    assert len(accepted) > 0.9 * 5 * c.t / 10                                     # vẫn dùng gần hết hạn mức


# ---------- C3 · ngân sách token ----------
PARTS = [{"name": "system", "tokens": 1500, "min": 1500, "priority": 3},
         {"name": "docs", "tokens": 3600, "min": 600, "priority": 1},
         {"name": "history", "tokens": 3600, "min": 300, "priority": 2},
         {"name": "question", "tokens": 50, "min": 50, "priority": 3}]


def test_allocate_budget_fits_and_priorities():
    assert allocate_budget(32_000, 1000, PARTS) == {"system": 1500, "docs": 3600, "history": 3600, "question": 50}
    got = allocate_budget(8192, 800, PARTS)
    assert got == {"system": 1500, "docs": 2242, "history": 3600, "question": 50}
    assert sum(got.values()) == 8192 - 800


def test_allocate_budget_ties_and_minimums():
    parts = [{"name": "a", "tokens": 100, "min": 10, "priority": 1}, {"name": "b", "tokens": 100, "min": 10, "priority": 1},
             {"name": "c", "tokens": 30}]                                         # không có min/priority → 0
    assert allocate_budget(150, 0, parts) == {"a": 100, "b": 50, "c": 0}          # hòa ưu tiên → theo thứ tự đầu vào
    assert allocate_budget(500, 0, [{"name": "x", "tokens": 5, "min": 50}]) == {"x": 5}   # min không vượt nhu cầu


def test_allocate_budget_impossible():
    with pytest.raises(ValueError):
        allocate_budget(2000, 800, PARTS)                                         # 1500 + 600 + 300 + 50 > 1200


# ---------- C4 · tool call trong luồng ----------
def _chunks():
    yield {"choices": [{"delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                                   "function": {"name": "tra_cuoc", "arguments": ""}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                                   "function": {"name": "huy_goi", "arguments": "{\"go"}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"sdt\": \"09"}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "i\": \"GX30\"}"}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "12345678\"}"}}]}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def test_tool_call_accumulator_parallel_interleaved():
    acc = ToolCallAccumulator()
    for ch in _chunks():
        acc.add(ch)
    assert acc.finish_reason == "tool_calls"
    assert acc.result() == [{"id": "call_a", "name": "tra_cuoc", "arguments": {"sdt": "0912345678"}},
                            {"id": "call_b", "name": "huy_goi", "arguments": {"goi": "GX30"}}]


def test_tool_call_accumulator_text_and_errors():
    acc = ToolCallAccumulator()
    for piece in ["Dạ, ", "em kiểm tra ", "ngay ạ."]:
        acc.add({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
    acc.add({"choices": [], "usage": {"prompt_tokens": 10}})                     # chunk usage cuối: không lỗi
    assert acc.text == "Dạ, em kiểm tra ngay ạ." and acc.result() == []
    bad = ToolCallAccumulator()
    bad.add({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{\"a\":"}}]}}]})
    with pytest.raises(ValueError):
        bad.result()
    empty = ToolCallAccumulator()
    empty.add({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]}}]})
    assert empty.result() == [{"id": "c", "name": "f", "arguments": {}}]
    assert json.dumps(empty.result())                                             # kết quả tuần tự hóa được
