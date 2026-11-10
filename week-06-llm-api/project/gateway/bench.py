"""Các thí nghiệm của project (cho sẵn). Chạy được khi bạn đã viết xong mọi module của Ngày 1–4.

    python -m gateway.bench                     # E1–E7 trên FakeLLM, thời gian mô phỏng, khoảng 2 phút
    python -m gateway.bench --only E1 E3        # chỉ chạy vài thí nghiệm
    python -m gateway.bench --provider local --n 100        # E8: cùng gateway, LLM nhỏ chạy local (tùy chọn)
    python -m gateway.bench --provider openai --base-url https://api.groq.com/openai/v1 \\
        --api-key-env GROQ_API_KEY --small MODEL_ID --large MODEL_ID --n 50          # E8 với API thật (tùy chọn)

Mô phỏng TUẦN TỰ: mỗi request được xử lý trên dòng thời gian riêng bắt đầu từ lúc nó đến (`SimClock.set(ts)`);
rate limiter và cache dùng đồng hồ "lúc đến". Không mô phỏng hàng đợi khi nhiều request chồng lên nhau.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random

import numpy as np
import pandas as pd

from gateway.app import TIERS, Gateway, GatewayConfig
from gateway.cache import TTLCache
from gateway.client import LLMClient
from gateway.clock import RealClock, SimClock
from gateway.data import make_workload
from gateway.fake import FakeLLM, build_oracle
from gateway.limits import KeyedLimiter
from gateway.router import default_router

ANSWERED = {"llm", "cache", "human", "refuse"}


def _slots_ok(rec, truth) -> bool:
    s = rec["slots"] or {}
    want = {"goi": truth.goi if isinstance(truth.goi, str) else None,
            "sdt": truth.sdt if isinstance(truth.sdt, str) else None,
            "tien": None if pd.isna(truth.tien) else int(truth.tien)}
    return all(s.get(k) == v for k, v in want.items())


async def replay(wl: pd.DataFrame, config: GatewayConfig, *, fake_kw: dict | None = None, limiter_kw=None,
                 cache: bool = True, max_attempts: int = 3, seed: int = 0) -> pd.DataFrame:
    """Chạy cả workload qua một gateway mới; trả về bảng log (một dòng mỗi request) kèm cột đáp án đúng."""
    clock, arrival = SimClock(), {"t": 0.0}
    fake = FakeLLM(oracle=build_oracle(wl), clock=clock, seed=seed, **(fake_kw or {}))
    client = LLMClient(fake.client(), "test-key", clock=clock, max_attempts=max_attempts, rng=random.Random(seed))
    limiter = KeyedLimiter(clock=lambda: arrival["t"], **limiter_kw) if limiter_kw else None
    gw = Gateway(client, default_router(), clock=clock, config=config, limiter=limiter,
                 cache=TTLCache(clock=lambda: arrival["t"]) if cache else None)
    first_reply = []
    for r in wl.itertuples():
        arrival["t"] = r.ts
        clock.set(r.ts)
        seen = []
        cb = (lambda d, seen=seen: seen or seen.append(clock.now() - r.ts)) if config.stream else None
        await gw.handle(r.user_id, r.text, on_reply_delta=cb)
        first_reply.append(seen[0] if seen else np.nan)
    df = pd.DataFrame(gw.records)
    df["visible_ttft_s"] = first_reply
    df["truth"] = wl["intent"].to_numpy()
    df["injected"] = wl["injected"].to_numpy()
    df["intent_ok"] = df["intent"] == df["truth"]
    df["slots_ok"] = [_slots_ok(rec, t) if rec["route"] in ("llm", "cache") else np.nan
                      for rec, t in zip(gw.records, wl.itertuples())]
    return df


def _p(x, q):
    x = pd.Series(x).dropna()
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summary(df: pd.DataFrame) -> dict:
    llm = df[df.route == "llm"]
    return {"cost_per_1k_usd": 1000 * df.cost_usd.mean(), "llm_calls_per_req": df.calls.mean(),
            "share_llm": (df.route == "llm").mean(), "share_cache": (df.route == "cache").mean(),
            "share_human": (df.route == "human").mean(), "share_refuse": (df.route == "refuse").mean(),
            "share_error": (df.route == "error").mean(), "answered": df.route.isin(ANSWERED).mean(),
            "intent_acc": df.loc[df.route.isin(ANSWERED), "intent_ok"].mean(),
            "p50_s": _p(llm.latency_s, 50), "p95_s": _p(llm.latency_s, 95)}


def e1_routing(wl):
    rows = {}
    for name, cfg, cache in [("all-large", GatewayConfig(routing="all-large"), False),
                             ("all-small", GatewayConfig(routing="all-small"), False),
                             ("intent", GatewayConfig(), False), ("intent + cache", GatewayConfig(), True)]:
        rows[name] = summary(asyncio.run(replay(wl, cfg, cache=cache)))
    return pd.DataFrame(rows).T


def e2_cache_key(wl):
    rows = {}
    for kind in ("raw", "light", "aggressive"):
        df = asyncio.run(replay(wl, GatewayConfig(cache_key_fn=kind)))
        hits = df[df.route == "cache"]
        rows[kind] = {"hit_rate": len(hits) / len(df), "wrong_slots_on_hit": int((hits.slots_ok == 0).sum()),
                      "cost_per_1k_usd": 1000 * df.cost_usd.mean()}
    return pd.DataFrame(rows).T


def e3_reliability(wl):
    outage = {"outages": {"model-small": [(30.0, 60.0)]}, "error_rate": 0.05}
    rows = {}
    for name, cfg, attempts in [("1 lần, không fallback", GatewayConfig(fallback=False), 1),
                                ("3 lần, không fallback", GatewayConfig(fallback=False), 3),
                                ("3 lần + fallback", GatewayConfig(), 3)]:
        df = asyncio.run(replay(wl, cfg, fake_kw=outage, max_attempts=attempts))
        s = summary(df)
        win = df[(df.ts >= 30) & (df.ts < 60)]
        rows[name] = {"answered": s["answered"], "answered_in_outage": win.route.isin(ANSWERED).mean(),
                      "fallback_used": df.fallback_used.mean(), "p95_s": s["p95_s"],
                      "cost_per_1k_usd": s["cost_per_1k_usd"]}
    return pd.DataFrame(rows).T


def e4_streaming(wl):
    rows = {}
    for name, stream in [("không stream", False), ("stream", True)]:
        df = asyncio.run(replay(wl, GatewayConfig(stream=stream), cache=False))
        llm = df[df.route == "llm"]
        seen = llm.visible_ttft_s if stream else llm.latency_s
        rows[name] = {"p50_first_text_s": _p(seen, 50), "p95_first_text_s": _p(seen, 95),
                      "p50_total_s": _p(llm.latency_s, 50), "p95_total_s": _p(llm.latency_s, 95)}
    return pd.DataFrame(rows).T


def e5_structured(wl):
    rows = {}
    for structured in ("none", "json_object", "json_schema"):
        for reask in (False, True):
            df = asyncio.run(replay(wl, GatewayConfig(structured=structured, reask=reask, fallback=False), cache=False))
            called = df[df.calls > 0]
            rows[f"{structured}, hỏi lại={'có' if reask else 'không'}"] = {
                "valid": called.valid.mean(), "repaired_locally": called.repaired.mean(),
                "reasked": called.reasked.mean(), "calls_per_llm_req": called.calls.mean(),
                "cost_per_1k_usd": 1000 * df.cost_usd.mean()}
    return pd.DataFrame(rows).T


def e6_rate_limit(wl):
    df = asyncio.run(replay(wl, GatewayConfig(), limiter_kw={"rate": 0.5, "capacity": 10}))
    top = df.user_id.value_counts().index[0]
    lim = df.route == "rate_limited"
    return pd.DataFrame({
        "tất cả": {"requests": len(df), "rate_limited": lim.mean()},
        f"khách nhiều nhất ({top})": {"requests": int((df.user_id == top).sum()),
                                      "rate_limited": lim[df.user_id == top].mean()},
        "các khách còn lại": {"requests": int((df.user_id != top).sum()), "rate_limited": lim[df.user_id != top].mean()},
    }).T


def e7_injection(n, seed):
    wl = make_workload(n, seed, inject_frac=0.2)
    rows = {}
    for wrap in (False, True):
        for guard in (False, True):
            df = asyncio.run(replay(wl, GatewayConfig(wrap_input=wrap, output_guard=guard), cache=False))
            inj = df[df.injected]
            rows[f"bọc thẻ={'có' if wrap else 'không'}, chặn đầu ra={'có' if guard else 'không'}"] = {
                "injected_requests": len(inj), "flagged_by_heuristic": inj.injection_flag.mean(),
                "model_leaked": inj.leaked.mean(), "leak_reached_user": (inj.leaked & (inj.route != "blocked")).mean(),
                "false_flag_on_clean": df[~df.injected].injection_flag.mean()}
    return pd.DataFrame(rows).T


async def e8_real(wl, args):
    """Cùng gateway, provider thật: LLM local hoặc API kiểu OpenAI. Không cache, không giới hạn tốc độ."""
    import httpx
    if args.provider == "local":
        from gateway.local_llm import LocalLLM
        backend = LocalLLM()
        http, key = backend.client(), "local"
    else:
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise SystemExit(f"thiếu biến môi trường {args.api_key_env} (đặt trong .env, không commit)")
        http = httpx.AsyncClient(base_url=args.base_url, timeout=60)
        TIERS["small"] = TIERS["small"].__class__(args.small, TIERS["small"].price)
        TIERS["large"] = TIERS["large"].__class__(args.large, TIERS["large"].price)
    clock = RealClock()
    gw = Gateway(LLMClient(http, key, clock=clock, max_attempts=4), default_router(), clock=clock,
                 config=GatewayConfig(stream=True, structured="json_object" if args.provider != "local" else "none"))
    for r in wl.itertuples():
        await gw.handle(r.user_id, r.text, on_reply_delta=lambda d: None)
    df = pd.DataFrame(gw.records)
    df["intent_ok"] = df["intent"] == wl["intent"].to_numpy()
    df["slots_ok"] = [_slots_ok(rec, t) if rec["route"] == "llm" else np.nan for rec, t in zip(gw.records, wl.itertuples())]
    llm = df[df.calls > 0]
    return pd.DataFrame({"giá trị": {
        "requests": len(df), "llm_requests": len(llm), "valid_json": llm.valid.mean(),
        "repaired_locally": llm.repaired.mean(), "reasked": llm.reasked.mean(),
        "intent_acc_llm": llm.loc[llm.valid, "intent_ok"].mean(), "slot_acc_llm": llm.slots_ok.dropna().astype(float).mean(),
        "router_acc_same_requests": (llm.router_intent == wl.loc[llm.index, "intent"]).mean(),
        "p50_ttft_s": _p(llm.ttft_s, 50), "p50_total_s": _p(llm.latency_s, 50), "p95_total_s": _p(llm.latency_s, 95),
        "out_tokens_per_llm_req": llm.output_tokens.mean()}})


EXPERIMENTS = {"E1": ("Chi phí và chất lượng theo cách định tuyến", e1_routing),
               "E2": ("Khóa cache: thô / chuẩn hóa nhẹ / chuẩn hóa mạnh", e2_cache_key),
               "E3": ("Độ tin cậy khi model nhỏ sập 30 s (cộng 5% lỗi ngẫu nhiên)", e3_reliability),
               "E4": ("Streaming: thời điểm khách thấy chữ đầu tiên", e4_streaming),
               "E5": ("Structured output: response_format và hỏi lại", e5_structured),
               "E6": ("Rate limit theo khách (0.5 request/s, burst 10)", e6_rate_limit),
               "E7": ("Prompt injection (20% request có câu tấn công)", e7_injection)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--provider", choices=["fake", "local", "openai"], default="fake")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--small", default="")
    ap.add_argument("--large", default="")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200, "display.precision", 3)
    if args.provider != "fake":
        print("## E8 · Gateway với provider thật\n")
        print(asyncio.run(e8_real(make_workload(args.n, args.seed), args)).to_string())
        return
    wl = make_workload(args.n, args.seed)
    for key, (title, fn) in EXPERIMENTS.items():
        if args.only and key not in args.only:
            continue
        print(f"\n## {key} · {title}\n")
        out = fn(args.n, args.seed) if key == "E7" else fn(wl)
        print(out.to_string())


if __name__ == "__main__":
    main()
