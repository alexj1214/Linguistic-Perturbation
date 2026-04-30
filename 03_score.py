"""
03_score.py — Score dialect-transformed texts using LLM judges (Anthropic + OpenAI).

Usage:
    python 03_score.py --dataset chalearn --dialect all --judge all
    python 03_score.py --dataset persuade --dialect indian --judge claude-opus-4-7
    python 03_score.py --dataset persuade --dialect all --judge gpt-4o,claude-haiku-4-5-20251001
    python 03_score.py --dataset persuade --dialect all --judge all --workers 20

    --judge   accepts: a model name, comma-separated names, or 'all' (uses config.JUDGE_MODELS)
    --workers number of concurrent API calls per (dialect, judge) batch (default 10)
    --limit   only score the first N samples per dialect (smoke tests / pilots)

Reads:  outputs/transformed/{dataset}_{dialect}.jsonl
Writes: outputs/scores/{dataset}_{dialect}_{judge}.jsonl

Each output line:
    {"sample_id", "dialect", "judge_model", "prompt_hash",
     "transformed_text", "raw_response", "parsed_score",
     "input_tokens", "output_tokens"}

IMPORTANT: Run with /opt/anaconda3/envs/dialect_bias/bin/python
           Anthropic models require ANTHROPIC_API_KEY.
           OpenAI models require OPENAI_API_KEY.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env (if present) before reading any API keys.
try:
    from dotenv import load_dotenv
    # override=True so an empty/stale value in the parent shell can't shadow .env
    load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv not installed — fall back to shell-exported env vars

import anthropic

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

import config
from utils import ensure_dirs, get_logger, safe_api_call

ensure_dirs()
log = get_logger("score", "03_score.log")

SCORE_RANGE = {
    "chalearn": (1, 5),
    "persuade": (1, 6),
}

# Cost per million tokens (input, output) in USD — used for cost reporting only.
_MODEL_COSTS = {
    "claude-opus-4-7":           (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80,  4.0),
    "claude-sonnet-4-6":         (3.0,  15.0),
    "gpt-4o":                    (2.50, 10.0),
    "gpt-4o-mini":               (0.15,  0.60),
}


# ── Provider helpers ───────────────────────────────────────────

def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def _make_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def _make_openai_client():
    if not _OPENAI_AVAILABLE:
        log.error("openai package not installed. Run: pip install openai")
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set in environment.")
        sys.exit(1)
    return _openai_module.OpenAI(api_key=api_key)


# ── Prompt loading ─────────────────────────────────────────────

def load_prompt_template(dataset: str) -> str:
    prompt_file = config.PROMPTS_DIR / f"{dataset}_judge.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text()


def build_prompt(template: str, text: str) -> str:
    return template.replace("{text}", text)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# ── Score parsing ──────────────────────────────────────────────

# Pre-compiled patterns for stripping rubric mentions ("out of 6", "/6",
# "1-6 scale", "on a 1 to 6 scale") so a literal scale max isn't mistaken
# for the judge's score.
_RUBRIC_PATTERNS = [
    re.compile(r"\bout\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bon\s+a\s+\d+\s*[-–to ]+\s*\d+\s*(?:point\s+)?scale\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*[-–]\s*\d+\s*(?:point\s+)?scale\b", re.IGNORECASE),
    re.compile(r"/\s*\d+\b"),
]


def parse_score(response_text: str, dataset: str) -> int | None:
    """
    Extract an integer score from a judge response.
    1. Strip phrases that mention the scale itself (so "out of 6" can't leak).
    2. Among remaining digits in the valid range, return the LAST match
       (judges typically conclude with the score).
    """
    lo, hi = SCORE_RANGE[dataset]
    text = response_text.strip()
    for pat in _RUBRIC_PATTERNS:
        text = pat.sub(" ", text)
    matches = re.findall(rf"\b([{lo}-{hi}])\b", text)
    if not matches:
        return None
    return int(matches[-1])


# ── API calls ──────────────────────────────────────────────────

def _supports_temperature(model: str) -> bool:
    # Newer Claude reasoning-class models reject the `temperature` parameter.
    if model.startswith("claude-opus-4-7"):
        return False
    return True


def call_judge(client, model: str, prompt: str) -> tuple[str, int, int]:
    """Returns (response_text, input_tokens, output_tokens)."""
    if _is_openai_model(model):
        def _call():
            kwargs = {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": prompt}],
            }
            if _supports_temperature(model):
                kwargs["temperature"] = 0
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            return resp.choices[0].message.content.strip(), in_tok, out_tok
    else:
        def _call():
            kwargs = {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": prompt}],
            }
            if _supports_temperature(model):
                kwargs["temperature"] = 0
            msg = client.messages.create(**kwargs)
            usage = getattr(msg, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) if usage else 0
            out_tok = getattr(usage, "output_tokens", 0) if usage else 0
            return msg.content[0].text.strip(), in_tok, out_tok
    return safe_api_call(_call)


# ── Cache helpers ──────────────────────────────────────────────

def load_cache(out_path: Path) -> set[tuple]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["dialect"], r["judge_model"], r["prompt_hash"]))
                except Exception:
                    pass
    return done


# ── Worker ─────────────────────────────────────────────────────

def _score_one(
    client, model: str, dataset: str,
    sid: str, dialect_key: str, text: str,
    prompt: str, p_hash: str,
) -> dict | None:
    """Run a single API call and return a JSONL-ready record (or None on failure)."""
    try:
        raw_response, in_tok, out_tok = call_judge(client, model, prompt)
    except Exception as e:
        log.error(f"[{dialect_key}] sample={sid!r}: API call failed: {e}")
        return None

    return {
        "sample_id":        sid,
        "dialect":          dialect_key,
        "judge_model":      model,
        "prompt_hash":      p_hash,
        "transformed_text": text,
        "raw_response":     raw_response,
        "parsed_score":     parse_score(raw_response, dataset),
        "input_tokens":     in_tok,
        "output_tokens":    out_tok,
    }


# ── Main ───────────────────────────────────────────────────────

def run(dataset: str, dialect_keys: list[str], judge_model: str, workers: int = 10,
        limit: int | None = None):
    client = _make_openai_client() if _is_openai_model(judge_model) else _make_anthropic_client()

    prompt_template = load_prompt_template(dataset)
    log.info(f"Dataset: {dataset}  |  Dialects: {dialect_keys}  |  Judge: {judge_model}  "
             f"|  Workers: {workers}")

    total_scored = 0
    total_cached = 0
    total_unparseable = 0
    total_in_tok = 0
    total_out_tok = 0
    write_lock = threading.Lock()

    for dialect_key in dialect_keys:
        in_path  = config.TRANSFORMED_DIR / f"{dataset}_{dialect_key}.jsonl"
        out_path = config.SCORES_DIR / f"{dataset}_{dialect_key}_{judge_model}.jsonl"

        if not in_path.exists():
            log.warning(f"[{dialect_key}] Input file not found: {in_path} — skipping.")
            continue

        cache = load_cache(out_path)

        # Build the work list up front so we can submit it to a thread pool.
        jobs = []
        with open(in_path) as fin:
            for i, line in enumerate(fin):
                if limit is not None and i >= limit:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid    = record["sample_id"]
                text   = record["transformed_text"]
                prompt = build_prompt(prompt_template, text)
                p_hash = prompt_hash(prompt)
                if (sid, dialect_key, judge_model, p_hash) in cache:
                    total_cached += 1
                    continue
                jobs.append((sid, text, prompt, p_hash))

        if not jobs:
            log.info(f"[{dialect_key}] Nothing to score (all cached or empty).")
            continue

        log.info(f"[{dialect_key}] Submitting {len(jobs)} jobs across {workers} workers.")

        with open(out_path, "a") as fout, \
             ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _score_one, client, judge_model, dataset,
                    sid, dialect_key, text, prompt, p_hash,
                ): sid
                for (sid, text, prompt, p_hash) in jobs
            }

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                rec = fut.result()
                if rec is None:
                    continue
                if rec["parsed_score"] is None:
                    total_unparseable += 1
                    log.warning(f"[{dialect_key}] sample={rec['sample_id']!r}: "
                                f"unparseable: {rec['raw_response']!r}")
                with write_lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                total_scored += 1
                total_in_tok += rec["input_tokens"]
                total_out_tok += rec["output_tokens"]
                if done_count % 25 == 0:
                    log.info(f"[{dialect_key}] {done_count}/{len(jobs)} done")

        log.info(f"[{dialect_key}] Done  →  {out_path}")

    log.info(f"Run complete — new scores: {total_scored}, "
             f"cached: {total_cached}, unparseable: {total_unparseable}")
    log.info(f"Tokens used — input: {total_in_tok:,}  output: {total_out_tok:,}")

    cost_in, cost_out = _MODEL_COSTS.get(judge_model, (3.0, 15.0))
    est_cost = (total_in_tok / 1_000_000 * cost_in) + (total_out_tok / 1_000_000 * cost_out)
    log.info(f"Actual API cost for this run: ~${est_cost:.4f} "
             f"(based on real token usage)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["chalearn", "persuade"])
    parser.add_argument("--dialect", default="all",
                        help="Dialect key, 'all', or comma-separated list")
    parser.add_argument("--judge", default="all",
                        help="Model name, 'all' (config.JUDGE_MODELS), or comma-separated list")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API workers (default 10). Lower if you hit rate limits.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N samples per dialect (default: all).")
    args = parser.parse_args()

    dialect_keys = (["sae"] + config.DIALECTS) if args.dialect == "all" else args.dialect.split(",")

    if args.judge == "all":
        judges = config.JUDGE_MODELS
    else:
        judges = args.judge.split(",")

    for judge in judges:
        run(args.dataset, dialect_keys, judge.strip(),
            workers=args.workers, limit=args.limit)
