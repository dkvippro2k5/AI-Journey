import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("multi_caller.prepare")


def load_prompts(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"prompt": "str", "category": "str"})


def _normalize(s: pd.Series) -> pd.Series:
    return s.str.casefold().str.split().str.join(" ")


def clean(df: pd.DataFrame, max_chars: int = 2000) -> pd.DataFrame:
    out = df.copy()
    out["prompt"] = out["prompt"].str.strip()
    out = out[out["prompt"].notna() & (out["prompt"] != "")]
    out["category"] = out["category"].str.strip().str.lower().fillna("general")
    out = out.drop_duplicates(subset="id", keep="first")
    out = out[~_normalize(out["prompt"]).duplicated(keep="first")]
    out["n_chars"] = out["prompt"].str.len()
    out = out[out["n_chars"] <= max_chars]
    return out.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "total": int(len(df)),
        "by_category": {k: int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "mean_chars": round(float(df["n_chars"].mean()), 1),
    }


def write_jsonl(df: pd.DataFrame, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in df[["id", "prompt", "category"]].to_dict("records"):
            fh.write(json.dumps({**rec, "id": int(rec["id"])}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a prompts CSV into JSONL")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = load_prompts(args.src)
    out = clean(raw, max_chars=args.max_chars)
    write_jsonl(out, args.dst)
    log.info("kept %d/%d rows -> %s | %s", len(out), len(raw), args.dst, summarize(out))


if __name__ == "__main__":
    main()
