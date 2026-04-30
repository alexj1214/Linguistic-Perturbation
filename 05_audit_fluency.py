"""
05_audit_fluency.py — Audit Multi-VALUE dialect transformations on three axes:
grammaticality, semantic preservation, dialect authenticity.

Usage:
    python 05_audit_fluency.py --dataset chalearn --n 30
    python 05_audit_fluency.py --dataset persuade --n 30 --auditor claude-opus-4-7

Reads:  outputs/transformed/{dataset}_{dialect}.jsonl
Writes: outputs/audits/{dataset}_fluency_audit.jsonl     (raw per-item ratings)
        outputs/logs/{dataset}_fluency_summary.csv       (mean ± std per dialect × axis)

The auditor model is held out from the JUDGE_MODELS pool by default
(claude-sonnet-4-6 is fine for sanity checks; use claude-opus-4-7 for camera-ready).

IMPORTANT: Run with /opt/anaconda3/envs/dialect_bias/bin/python
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import anthropic
import numpy as np
import pandas as pd

import config
from utils import ensure_dirs, get_logger, safe_api_call

ensure_dirs()
log = get_logger("audit", "05_audit_fluency.log")

AUDITS_DIR = config.OUTPUTS_DIR / "audits"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

DIALECT_NAME_MAP = {
    "aave":      "African American Vernacular English",
    "indian":    "Indian English",
    "singapore": "Colloquial Singapore English (Singlish)",
}


def load_prompt() -> str:
    return (config.PROMPTS_DIR / "fluency_audit.txt").read_text()


def build_prompt(template: str, original: str, transformed: str, dialect_name: str) -> str:
    return (template
            .replace("{dialect_name}", dialect_name)
            .replace("{original}", original)
            .replace("{transformed}", transformed))


_AXIS_PATTERNS = {
    "grammaticality": re.compile(r"GRAMMATICALITY:\s*([1-5])"),
    "semantic":       re.compile(r"SEMANTIC:\s*([1-5])"),
    "authenticity":   re.compile(r"AUTHENTICITY:\s*([1-5])"),
}


def parse_audit(response: str) -> dict[str, int | None]:
    out = {}
    for axis, pat in _AXIS_PATTERNS.items():
        m = pat.search(response)
        out[axis] = int(m.group(1)) if m else None
    return out


def call_auditor(client, model: str, prompt: str) -> tuple[str, int, int]:
    def _call():
        kwargs = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not model.startswith("claude-opus-4-7"):
            kwargs["temperature"] = 0
        msg = client.messages.create(**kwargs)
        usage = getattr(msg, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        if not msg.content:
            return "", in_tok, out_tok  # refusal
        return msg.content[0].text.strip(), in_tok, out_tok
    return safe_api_call(_call)


def load_pairs(dataset: str, dialect: str, n: int, seed: int) -> list[dict]:
    """
    Returns a list of {sample_id, original, transformed} dicts,
    sampling n items from the dialect file (deterministic via seed).
    """
    path = config.TRANSFORMED_DIR / f"{dataset}_{dialect}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip items where the transformation didn't actually fire any rule —
            # rating those tells us nothing about Multi-VALUE quality.
            if not r.get("changed", False):
                continue
            rows.append({
                "sample_id":   r["sample_id"],
                "original":    r["original_text"],
                "transformed": r["transformed_text"],
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def run(dataset: str, n: int, auditor: str, workers: int):
    template = load_prompt()
    client = anthropic.Anthropic()
    out_path = AUDITS_DIR / f"{dataset}_fluency_audit.jsonl"

    # Resume cache: dedupe by (sample_id, dialect, auditor)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["dialect"], r["auditor"]))
                except Exception:
                    pass

    all_jobs = []
    for dialect in config.DIALECTS:
        pairs = load_pairs(dataset, dialect, n, seed=config.RANDOM_SEED)
        for p in pairs:
            if (p["sample_id"], dialect, auditor) in done:
                continue
            all_jobs.append((dialect, p))

    if not all_jobs:
        log.info("All audits cached; nothing to do.")
    else:
        log.info(f"Auditing {len(all_jobs)} items via {auditor} ({workers} workers)")
        with open(out_path, "a") as fout, \
             ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for dialect, p in all_jobs:
                dialect_name = DIALECT_NAME_MAP[dialect]
                prompt = build_prompt(template, p["original"], p["transformed"], dialect_name)
                fut = pool.submit(_audit_one, client, auditor,
                                  dataset, dialect, p, prompt)
                futs[fut] = (dialect, p["sample_id"])

            done_count = 0
            for fut in as_completed(futs):
                done_count += 1
                rec = fut.result()
                if rec is None:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if done_count % 25 == 0:
                    log.info(f"  {done_count}/{len(all_jobs)} done")

    # ── Summary CSV ───────────────────────────────────────────
    rows = []
    with open(out_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        log.warning("No audit records to summarize.")
        return
    df = pd.DataFrame(rows)
    df = df[df["auditor"] == auditor]
    summary_rows = []
    for dialect, grp in df.groupby("dialect"):
        for axis in ("grammaticality", "semantic", "authenticity"):
            vals = pd.to_numeric(grp[axis], errors="coerce").dropna()
            summary_rows.append({
                "dialect": dialect,
                "axis":    axis,
                "n":       len(vals),
                "mean":    round(vals.mean(), 3),
                "std":     round(vals.std(), 3),
                "median":  vals.median(),
            })
    summary = pd.DataFrame(summary_rows)
    log.info(f"\nFluency-audit summary ({auditor}, dataset={dataset}):\n"
             f"{summary.to_string(index=False)}")
    summary_path = config.LOGS_DIR / f"{dataset}_fluency_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path}")


def _audit_one(client, auditor, dataset, dialect, pair, prompt) -> dict | None:
    try:
        raw, in_tok, out_tok = call_auditor(client, auditor, prompt)
    except Exception as e:
        log.error(f"[{dialect}] sample={pair['sample_id']!r}: {e}")
        return None
    parsed = parse_audit(raw)
    return {
        "sample_id":     pair["sample_id"],
        "dataset":       dataset,
        "dialect":       dialect,
        "auditor":       auditor,
        "raw_response":  raw,
        **parsed,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["chalearn", "persuade"])
    parser.add_argument("--n", type=int, default=30,
                        help="Number of items to audit per dialect (default 30).")
    parser.add_argument("--auditor", default="claude-sonnet-4-6",
                        help="Anthropic model used as auditor (default claude-sonnet-4-6).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.dataset, args.n, args.auditor, args.workers)
