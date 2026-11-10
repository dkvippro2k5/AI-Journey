import json

import pytest

from gateway.schema import Reply, extract_json, light_repair, parse_reply, repair_prompt, response_format

GOOD = {"reply": "Dạ, em hủy gói GX30 ạ.", "intent": "huy_goi", "slots": {"goi": "GX30", "sdt": None, "tien": None}}
GOOD_S = json.dumps(GOOD, ensure_ascii=False)


def test_reply_schema_rules():
    assert Reply.model_validate(GOOD).slots.goi == "GX30"
    bad = [
        {**GOOD, "intent": "hủy gói"},                                            # ngoài danh sách intent
        {**GOOD, "extra": 1},                                                      # khóa thừa
        {k: v for k, v in GOOD.items() if k != "reply"},                           # thiếu reply
        {**GOOD, "slots": {"goi": None, "sdt": "912345678", "tien": None}},        # sdt sai định dạng
        {**GOOD, "slots": {"goi": None, "sdt": None, "tien": 50}},                 # tiền quá nhỏ
    ]
    for b in bad:
        assert parse_reply(json.dumps(b, ensure_ascii=False)).reply is None
    assert list(Reply.model_json_schema()["properties"])[0] == "reply"           # reply đứng đầu
    rf = response_format()
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True


def test_extract_json_handles_fences_preface_and_braces_in_strings():
    assert extract_json(f"```json\n{GOOD_S}\n```") == GOOD_S
    assert extract_json(f"Đây là kết quả:\n{GOOD_S}\nCảm ơn!") == GOOD_S
    tricky = '{"reply": "dùng ký tự } và { trong chuỗi, cả \\" nữa", "intent": "khac", "slots": {}}'
    assert extract_json("abc " + tricky + " xyz {") == tricky
    with pytest.raises(ValueError):
        extract_json("không có json")
    with pytest.raises(ValueError):
        extract_json(GOOD_S[:-10])                                                 # bị cắt giữa chừng


def test_light_repair_only_outside_strings():
    assert light_repair('{"a": [1, 2,], "b": {"c": 3,},}') == '{"a": [1, 2], "b": {"c": 3}}'
    assert light_repair('{"reply": "1,}", "x": 1 , }') == '{"reply": "1,}", "x": 1  }'


def test_parse_reply_paths():
    ok = parse_reply(GOOD_S)
    assert ok.reply is not None and ok.error is None and not ok.repaired
    fixed = parse_reply(f"Kết quả:\n```json\n{GOOD_S[:-1]},}}\n```")               # hàng rào + dấu phẩy thừa
    assert fixed.reply is not None and fixed.repaired
    for broken in [str(GOOD), GOOD_S[: len(GOOD_S) // 2], "xin lỗi"]:             # nháy đơn, bị cắt, không có JSON
        r = parse_reply(broken)
        assert r.reply is None and r.error.startswith("JSON")
    r = parse_reply(json.dumps({**GOOD, "intent": "x"}))
    assert r.reply is None and r.error.startswith("schema") and "intent" in r.error
    assert "intent" in repair_prompt(r.error) and repair_prompt(r.error).startswith("Câu trả lời trước không phải JSON")
