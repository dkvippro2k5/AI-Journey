from gateway.cache import TTLCache, cache_key, canonical_json

MSGS = [{"role": "system", "content": "abc"}, {"role": "user", "content": "hủy gói"}]


def test_canonical_json_and_cache_key():
    assert canonical_json({"b": 1, "a": "hủy"}) == canonical_json({"a": "hủy", "b": 1}) == '{"a":"hủy","b":1}'
    k = cache_key("m", MSGS, {"temperature": 0, "max_tokens": 200})
    assert len(k) == 64
    assert k == cache_key("m", MSGS, {"max_tokens": 200, "temperature": 0, "stream": True, "user": "u1"})
    assert k == cache_key("m", [{**m, "name": "x"} for m in MSGS], {"temperature": 0, "max_tokens": 200})
    assert k != cache_key("m2", MSGS, {"temperature": 0, "max_tokens": 200})
    assert k != cache_key("m", MSGS, {"temperature": 0.7, "max_tokens": 200})
    assert k != cache_key("m", MSGS[::-1], {"temperature": 0, "max_tokens": 200})


def test_ttl_cache_expiry_and_stats():
    now = [0.0]
    c = TTLCache(max_items=10, ttl=60, clock=lambda: now[0])
    assert c.get("a") is None
    c.set("a", {"reply": "x"})
    now[0] = 59.9
    assert c.get("a") == {"reply": "x"}
    now[0] = 60.0
    assert c.get("a") is None and len(c) == 0                                     # hết hạn bị xóa
    assert (c.hits, c.misses) == (1, 2) and c.hit_rate == 1 / 3


def test_ttl_cache_lru_eviction():
    c = TTLCache(max_items=2, ttl=1e9)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")                                                                    # a vừa được dùng
    c.set("c", 3)                                                                 # đầy → bỏ b (dùng lâu nhất)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
