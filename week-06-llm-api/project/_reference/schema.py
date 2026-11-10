"""Ngày 2 · Structured output: schema bằng Pydantic, tách JSON khỏi văn bản thừa, sửa lỗi nhẹ, vòng hỏi lại."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Intent = Literal["tra_cuoc", "nap_tien", "dang_ky_goi", "huy_goi", "loi_mang", "doi_sim", "khieu_nai", "gap_tvv", "khac"]


class Slots(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goi: str | None = None                                                   # tên gói, ví dụ "GX30"
    sdt: str | None = Field(None, pattern=r"^0\d{9}$")                       # 10 chữ số, bắt đầu bằng 0
    tien: int | None = Field(None, ge=1_000, le=10_000_000)                  # VND


class Reply(BaseModel):
    """Đầu ra của LLM. `reply` đứng ĐẦU để có thể stream câu trả lời cho khách trước khi model viết xong phần còn lại."""
    model_config = ConfigDict(extra="forbid")
    reply: str = Field(min_length=1, max_length=600)
    intent: Intent
    slots: Slots


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Lấy đối tượng JSON đầu tiên trong `text`: bỏ khối ```json … ``` nếu có, tìm '{' đầu tiên rồi đi tới dấu '}' khớp
    (bỏ qua ngoặc nằm trong chuỗi, xử lý ký tự escape). Không có '{' → ValueError; thiếu '}' → ValueError."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("không có đối tượng JSON")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("đối tượng JSON chưa đóng (có thể bị cắt vì max_tokens)")


def light_repair(s: str) -> str:
    """Xóa dấu phẩy thừa ngay trước '}' hoặc ']' (lỗi hay gặp nhất), không đụng vào nội dung trong chuỗi."""
    out, in_str, esc, i = [], False, False, 0
    while i < len(s):
        c = s[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c == ",":
            j = i + 1
            while j < len(s) and s[j] in " \t\r\n":
                j += 1
            if j >= len(s) or s[j] not in "}]":
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


@dataclass
class ParseResult:
    reply: Reply | None
    error: str | None
    repaired: bool


def _short(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '(gốc)'}: {e['msg']}" for e in exc.errors()[:3])


def parse_reply(text: str) -> ParseResult:
    """Thử parse thẳng; không được thì `extract_json` + `light_repair` rồi parse lại (`repaired=True`).
    Lỗi cú pháp JSON hoặc sai schema → `reply=None` và `error` là mô tả ngắn để gửi lại cho model."""
    candidate, repaired = text.strip(), False
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            fixed = light_repair(extract_json(text))
            data = json.loads(fixed)
            repaired = True
        except (ValueError, json.JSONDecodeError) as exc:
            return ParseResult(None, f"JSON: {exc}", False)
    try:
        return ParseResult(Reply.model_validate(data), None, repaired)
    except ValidationError as exc:
        return ParseResult(None, f"schema: {_short(exc)}", repaired)


def repair_prompt(error: str) -> str:
    """Tin nhắn gửi lại cho model khi đầu ra hỏng (vòng hỏi lại tối đa một lần)."""
    return (f"Câu trả lời trước không phải JSON hợp lệ theo schema ({error}). "
            "Chỉ trả về đúng một đối tượng JSON với các khóa reply, intent, slots; không giải thích thêm.")


def response_format(strict: bool = True) -> dict:
    """Tham số `response_format` kiểu OpenAI: structured outputs theo JSON Schema sinh từ Pydantic."""
    return {"type": "json_schema", "json_schema": {"name": "reply", "strict": strict, "schema": Reply.model_json_schema()}}
