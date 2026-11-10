import json
import random

import pytest

from gateway.stream import FieldStreamer, SSEEvent, SSEParser, StreamAccumulator, StreamError, sse_encode

RAW = ("retry: 3000\n: keep-alive\n\n"
       "event: delta\nid: 7\ndata: {\"t\": \"hủy gói\"}\n\n"
       "data: dòng 1\r\ndata: dòng 2\r\n\r\n"
       "data:không có dấu cách\rdata\r\r"
       "event: ping\n\n"                                                           # không có data → không phát
       "data: [DONE]\n\n").encode("utf-8")


def _events_in_chunks(raw: bytes, sizes) -> list:
    p, out, i = SSEParser(), [], 0
    for n in sizes:
        out += p.feed(raw[i:i + n])
        i += n
    out += p.feed(raw[i:])
    return out, p


def test_sse_parser_whole_stream():
    events, p = _events_in_chunks(RAW, [])
    assert [(e.event, e.data) for e in events] == [
        ("delta", '{"t": "hủy gói"}'), ("message", "dòng 1\ndòng 2"), ("message", "không có dấu cách\n"),
        ("message", "[DONE]")]
    assert events[0].id == "7" and events[1].id == "7"                            # id được nhớ cho sự kiện sau
    assert p.retry == 3000


def test_sse_parser_any_chunking_gives_same_events():
    whole, _ = _events_in_chunks(RAW, [])
    rng = random.Random(0)
    for _ in range(200):                                                          # cắt ở mọi chỗ, kể cả giữa \r\n và giữa UTF-8
        sizes = [rng.randint(1, 7) for _ in range(len(RAW))]
        got, _ = _events_in_chunks(RAW, sizes)
        assert got == whole
    one_by_one, _ = _events_in_chunks(RAW, [1] * len(RAW))
    assert one_by_one == whole


def test_sse_incomplete_event_is_dropped_on_close():
    p = SSEParser()
    assert p.feed(b"data: a\n\ndata: b\n") == [SSEEvent("a")]
    assert p.close() == []


def test_stream_accumulator():
    acc = StreamAccumulator()
    chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
              {"choices": [{"delta": {"content": "Dạ, "}, "finish_reason": None}]},
              {"choices": [{"delta": {"content": "em chào"}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 4}}]
    deltas = [acc.add(SSEEvent(json.dumps(c, ensure_ascii=False))) for c in chunks]
    assert deltas == ["", "Dạ, ", "em chào", "", ""]
    assert acc.add(SSEEvent("[DONE]")) == "" and acc.done
    assert acc.text == "Dạ, em chào" and acc.finish_reason == "stop" and acc.usage["completion_tokens"] == 4
    with pytest.raises(StreamError):
        StreamAccumulator().add(SSEEvent('{"error": {"message": "overloaded"}}'))


def test_sse_encode_roundtrip():
    raw = sse_encode({"delta": "hủy\ngói"}, event="delta") + sse_encode("dòng 1\ndòng 2")
    assert raw.startswith(b"event: delta\ndata: ")
    events = SSEParser().feed(raw)
    assert json.loads(events[0].data) == {"delta": "hủy\ngói"} and events[0].event == "delta"
    assert events[1].data == "dòng 1\ndòng 2"


def test_field_streamer_any_chunking():
    reply = 'Dạ, gói "GX30" giá 70k\nhủy: soạn HUY \\ gửi 1234 ✓'
    obj = json.dumps({"reply": reply, "intent": "huy_goi", "slots": {"goi": "GX30"}})   # ensure_ascii → có \uXXXX
    rng = random.Random(1)
    for _ in range(100):
        fs, out, i = FieldStreamer("reply"), [], 0
        while i < len(obj):
            n = rng.randint(1, 5)
            out.append(fs.feed(obj[i:i + n]))
            i += n
        assert "".join(out) == reply
        assert fs.closed
    fs = FieldStreamer("reply")
    assert fs.feed('{"intent": "khac", "re') == "" and fs.feed('ply" :  "Dạ') == "Dạ"
