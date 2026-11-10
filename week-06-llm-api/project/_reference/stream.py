"""Ngày 3 · Streaming: parser Server-Sent Events chạy tăng dần, gom delta kiểu Chat Completions, mã hóa SSE
phía server, và trích dần giá trị một trường JSON khi nó đang được stream."""
from __future__ import annotations

import codecs
import json
from dataclasses import dataclass


@dataclass
class SSEEvent:
    data: str
    event: str = "message"
    id: str | None = None


class SSEParser:
    """Parser SSE theo chuẩn WHATWG HTML (mục "Server-sent events"), nhận byte theo từng mảnh bất kỳ:
    - mảnh có thể cắt giữa một dòng, giữa "\\r\\n", hoặc giữa một ký tự UTF-8 nhiều byte;
    - dòng kết thúc bằng "\\r\\n", "\\n" hoặc "\\r"; dòng trống = phát sự kiện;
    - dòng bắt đầu bằng ":" là chú thích; "field: value" bỏ MỘT dấu cách đầu của value;
    - nhiều dòng `data:` được nối bằng "\\n"; sự kiện không có `data` thì không phát;
    - `id` được nhớ cho các sự kiện sau; `retry` (số nguyên) lưu ở `self.retry`."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buf = ""
        self._data: list[str] = []
        self._event = ""
        self.last_id: str | None = None
        self.retry: int | None = None

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += self._decoder.decode(chunk)
        events = []
        while True:
            i = min((k for k in (self._buf.find("\r"), self._buf.find("\n")) if k >= 0), default=-1)
            if i < 0:
                break
            if self._buf[i] == "\r" and i == len(self._buf) - 1:
                break                                                  # chờ xem ký tự sau có phải "\n" không
            line = self._buf[:i]
            step = 2 if self._buf.startswith("\r\n", i) else 1
            self._buf = self._buf[i + step:]
            ev = self._line(line)
            if ev is not None:
                events.append(ev)
        return events

    def close(self) -> list[SSEEvent]:
        """Kết thúc luồng. Theo chuẩn, sự kiện chưa có dòng trống kết thúc bị bỏ."""
        self._decoder.decode(b"", final=True)
        self._buf, self._data, self._event = "", [], ""
        return []

    def _line(self, line: str) -> SSEEvent | None:
        if line == "":
            if not self._data:
                self._event = ""
                return None
            ev = SSEEvent("\n".join(self._data), self._event or "message", self.last_id)
            self._data, self._event = [], ""
            return ev
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\0" not in value:
            self.last_id = value
        elif field == "retry" and value.isdigit():
            self.retry = int(value)
        return None


class StreamError(Exception):
    pass


class StreamAccumulator:
    """Gom các sự kiện `chat.completion.chunk` (kiểu OpenAI) thành văn bản hoàn chỉnh.
    `add(event)` trả về phần văn bản mới (có thể rỗng). "[DONE]" → `done=True`. Chunk có khóa "error" → StreamError.
    Lưu `finish_reason` và `usage` (chunk cuối khi gửi `stream_options={"include_usage": True}`)."""

    def __init__(self):
        self.parts: list[str] = []
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.done = False

    def add(self, event: SSEEvent) -> str:
        if event.data == "[DONE]":
            self.done = True
            return ""
        obj = json.loads(event.data)
        if "error" in obj:
            raise StreamError(str(obj["error"]))
        if obj.get("usage"):
            self.usage = obj["usage"]
        delta = ""
        for choice in obj.get("choices", []):
            delta += choice.get("delta", {}).get("content") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if delta:
            self.parts.append(delta)
        return delta

    @property
    def text(self) -> str:
        return "".join(self.parts)


def sse_encode(data, event: str | None = None) -> bytes:
    """Một sự kiện SSE: `data` là str hoặc dict (→ JSON, giữ nguyên tiếng Việt); chuỗi nhiều dòng thành nhiều dòng
    `data:`; có `event` thì thêm dòng `event:`; kết thúc bằng một dòng trống."""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    lines = ([f"event: {event}"] if event else []) + [f"data: {line}" for line in data.split("\n")]
    return ("\n".join(lines) + "\n\n").encode("utf-8")


_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class FieldStreamer:
    """Trích dần giá trị chuỗi của khóa `field` trong một đối tượng JSON đang được stream, để hiện câu trả lời cho khách
    trước khi JSON đầy đủ. `feed(delta)` trả về các ký tự MỚI của giá trị (đã giải escape, kể cả \\uXXXX).
    Giả định khóa `field` xuất hiện ở cấp ngoài cùng và chỉ một lần (schema `Reply` đặt `reply` đứng đầu)."""

    def __init__(self, field: str = "reply"):
        self.key = f'"{field}"'
        self.buf = ""
        self.pos = -1                     # vị trí ngay sau dấu " mở của giá trị; -1 = chưa tìm thấy
        self.closed = False

    def feed(self, delta: str) -> str:
        self.buf += delta
        if self.closed:
            return ""
        if self.pos < 0:
            k = self.buf.find(self.key)
            if k < 0:
                return ""
            j = k + len(self.key)
            while j < len(self.buf) and self.buf[j] in " \t\r\n":
                j += 1
            if j < len(self.buf) and self.buf[j] == ":":
                j += 1
                while j < len(self.buf) and self.buf[j] in " \t\r\n":
                    j += 1
            if j >= len(self.buf) or self.buf[j] != '"':
                return ""                                             # chưa đủ dữ liệu để thấy dấu " mở
            self.pos = j + 1
        out = []
        while self.pos < len(self.buf):
            c = self.buf[self.pos]
            if c == '"':
                self.closed = True
                break
            if c != "\\":
                out.append(c)
                self.pos += 1
                continue
            if self.pos + 1 >= len(self.buf):
                break                                                 # escape bị cắt ở cuối mảnh: chờ mảnh sau
            e = self.buf[self.pos + 1]
            if e == "u":
                if self.pos + 6 > len(self.buf):
                    break
                out.append(chr(int(self.buf[self.pos + 2:self.pos + 6], 16)))
                self.pos += 6
            else:
                out.append(_ESC.get(e, e))
                self.pos += 2
        return "".join(out)
