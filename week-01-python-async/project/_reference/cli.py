import argparse
import asyncio
import dataclasses
import json
import logging
import os
import tomllib
from pathlib import Path

import pandas as pd

from multi_caller.caller import ask
from multi_caller.fake import FakeLLM
from multi_caller.models import CallResult, ProviderConfig
from multi_caller.utils import read_jsonl, timed

log = logging.getLogger("multi_caller")


def load_providers(path) -> list[ProviderConfig]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return [ProviderConfig(**p) for p in data["providers"]]


async def run_all(client, providers, prompts, *, concurrency: int, **ask_kwargs) -> list[CallResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(provider, item):
        async with sem:
            return await ask(client, provider, item["id"], item["prompt"], **ask_kwargs)

    jobs = [one(p, item) for item in prompts for p in providers]
    return await asyncio.gather(*jobs)


def summarize_results(results: list[CallResult]) -> pd.DataFrame:
    df = pd.DataFrame([{**dataclasses.asdict(r), "ok": r.ok} for r in results])
    df["tokens"] = df["input_tokens"] + df["output_tokens"]
    return df.groupby("provider").agg(
        n=("ok", "size"),
        success_rate=("ok", "mean"),
        p50_latency_s=("latency_s", "median"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        total_tokens=("tokens", "sum"),
        total_cost_usd=("cost_usd", "sum"),
    )


def write_results(results: list[CallResult], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")


@timed
async def _run(args) -> list[CallResult]:
    providers = load_providers(args.providers)
    prompts = list(read_jsonl(args.prompts))
    if args.fake:
        for p in providers:
            os.environ.setdefault(p.api_key_env, "fake-key")
        fake = FakeLLM(seed=args.seed)
        client = fake.client()
    else:
        import httpx
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    async with client:
        return await run_all(client, providers, prompts, concurrency=args.concurrency)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Send prompts to several LLM providers concurrently")
    parser.add_argument("prompts", type=Path)
    parser.add_argument("--providers", type=Path, default=Path("providers.toml"))
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fake", action="store_true", help="use the offline fake server")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    results = asyncio.run(_run(args))
    write_results(results, args.out)
    print(summarize_results(results).round(4).to_string())


if __name__ == "__main__":
    main()
