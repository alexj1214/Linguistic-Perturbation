"""
03_score.py — Score dialect-transformed texts using LLM judges (Anthropic + OpenAI).

Usage:
    python 03_score.py --dataset chalearn --dialect all --judge all
    python 03_score.py --dataset persuade --dialect indian --judge claude-opus-4-7
    python 03_score.py --dataset persuade --dialect all --judge gpt-4o,claude-haiku-4-5-20251001

    --judge accepts: a model name, comma-separated names, or 'all' (uses config.JUDGE_MODELS)

Reads:  outputs/transformed/{dataset}_{dialect}.jsonl
Writes: outputs/scores/{dataset}_{dialect}_{judge}.jsonl

Each output line:
    {"sample_id", "dialect", "judge_model", "prompt_hash",
     "transformed_text", "raw_response", "parsed_score", "cached"}

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
from pathlib import Path

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

# Rough cost per million tokens (input, output) in USD
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

def parse_score(response_text: str, dataset: str) -> int | None:
    lo, hi = SCORE_RANGE[dataset]
    matches = re.findall(r"\b([1-6])\b", response_text.strip())
    for m in matches:
        val = int(m)
        if lo <= val <= hi:
            return val
    return None


# ── API calls ──────────────────────────────────────────────────

def call_judge(client, model: str, prompt: str) -> str:
    if _is_openai_model(model):
        def _call():
            resp = client.chat.completions.create(
                model=model,
                max_tokens=16,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
    else:
        def _call():
            msg = client.messages.create(
                model=model,
                max_tokens=16,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
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


# ── Main ───────────────────────────────────────────────────────

def run(dataset: str, dialect_keys: list[str], judge_model: str):
    client = _make_openai_client() if _is_openai_model(judge_model) else _make_anthropic_client()

    prompt_template = load_prompt_template(dataset)
    log.info(f"Dataset: {dataset}  |  Dialects: {dialect_keys}  |  Judge: {judge_model}")

    total_scored = 0
    total_cached = 0
    total_unparseable = 0

    for dialect_key in dialect_keys:
        in_path  = config.TRANSFORMED_DIR / f"{dataset}_{dialect_key}.jsonl"
        out_path = config.SCORES_DIR / f"{dataset}_{dialect_key}_{judge_model}.jsonl"

        if not in_path.exists():
            log.warning(f"[{dialect_key}] Input file not found: {in_path} — skipping.")
            continue

        cache = load_cache(out_path)

        with open(in_path) as fin, open(out_path, "a") as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid     = record["sample_id"]
                text    = record["transformed_text"]
                prompt  = build_prompt(prompt_template, text)
                p_hash  = prompt_hash(prompt)
                cache_key = (sid, dialect_key, judge_model, p_hash)

                if cache_key in cache:
                    total_cached += 1
                    continue

                try:
                    raw_response = call_judge(client, judge_model, prompt)
                except Exception as e:
                    log.error(f"[{dialect_key}] sample={sid!r}: API call failed: {e}")
                    continue

                parsed_score = parse_score(raw_response, dataset)
                if parsed_score is None:
                    total_unparseable += 1
                    log.warning(f"[{dialect_key}] sample={sid!r}: unparseable: {raw_response!r}")

                out_record = {
                    "sample_id":        sid,
                    "dialect":          dialect_key,
                    "judge_model":      judge_model,
                    "prompt_hash":      p_hash,
                    "transformed_text": text,
                    "raw_response":     raw_response,
                    "parsed_score":     parsed_score,
                    "cached":           False,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                cache.add(cache_key)
                total_scored += 1

                if total_scored % 25 == 0:
                    log.info(f"Scored {total_scored} so far...")

        log.info(f"[{dialect_key}] Done  →  {out_path}")

    log.info(f"Run complete — new scores: {total_scored}, "
             f"cached: {total_cached}, unparseable: {total_unparseable}")

    cost_in, cost_out = _MODEL_COSTS.get(judge_model, (3.0, 15.0))
    est_cost = (total_scored * 500 / 1_000_000 * cost_in) + (total_scored * 2 / 1_000_000 * cost_out)
    log.info(f"Estimated API cost for this run: ~${est_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["chalearn", "persuade"])
    parser.add_argument("--dialect", default="all",
                        help="Dialect key, 'all', or comma-separated list")
    parser.add_argument("--judge", default="all",
                        help="Model name, 'all' (config.JUDGE_MODELS), or comma-separated list")
    args = parser.parse_args()

    dialect_keys = (["sae"] + config.DIALECTS) if args.dialect == "all" else args.dialect.split(",")

    if args.judge == "all":
        judges = config.JUDGE_MODELS
    else:
        judges = args.judge.split(",")

    for judge in judges:
        run(args.dataset, dialect_keys, judge.strip())
