"""
03_score.py — Score dialect-transformed texts with an LLM judge.

Usage:
    python 03_score.py --dataset chalearn --dialect all --judge claude-sonnet-4-5
    python 03_score.py --dataset persuade --dialect indian --judge claude-sonnet-4-5

Reads:  outputs/transformed/{dataset}_{dialect}.jsonl
Writes: outputs/scores/{dataset}_{dialect}_{judge}.jsonl

Each output line:
    {"sample_id", "dialect", "judge_model", "prompt_hash",
     "transformed_text", "raw_response", "parsed_score", "cached"}

IMPORTANT: Run with /opt/anaconda3/envs/dialect_bias/bin/python
           Requires ANTHROPIC_API_KEY in environment.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import anthropic

import config
from utils import ensure_dirs, get_logger, safe_api_call

ensure_dirs()
log = get_logger("score", "03_score.log")

# Score ranges per dataset
SCORE_RANGE = {
    "chalearn": (1, 5),
    "persuade": (1, 6),
}


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
    """Extract integer score from model response. Returns None if unparseable."""
    lo, hi = SCORE_RANGE[dataset]
    # Try to find a standalone integer in range
    matches = re.findall(r"\b([1-6])\b", response_text.strip())
    for m in matches:
        val = int(m)
        if lo <= val <= hi:
            return val
    return None


# ── API call ───────────────────────────────────────────────────

def call_judge(client: anthropic.Anthropic, model: str, prompt: str) -> str:
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
    """Return set of (sample_id, dialect, judge_model, prompt_hash) already done."""
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

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
                    log.warning(f"[{dialect_key}] sample={sid!r}: unparseable response: {raw_response!r}")

                out_record = {
                    "sample_id":       sid,
                    "dialect":         dialect_key,
                    "judge_model":     judge_model,
                    "prompt_hash":     p_hash,
                    "transformed_text": text,
                    "raw_response":    raw_response,
                    "parsed_score":    parsed_score,
                    "cached":          False,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                cache.add(cache_key)
                total_scored += 1

                if total_scored % 25 == 0:
                    log.info(f"Scored {total_scored} so far...")

        log.info(f"[{dialect_key}] Done  →  {out_path}")

    log.info(f"Run complete — new scores: {total_scored}, "
             f"cached: {total_cached}, unparseable: {total_unparseable}")

    # Rough cost estimate (claude-sonnet-4-5: ~$3/M input, ~$15/M output)
    # Estimate: avg ~500 tokens input, 2 tokens output
    est_input_tokens  = total_scored * 500
    est_output_tokens = total_scored * 2
    est_cost = (est_input_tokens / 1_000_000 * 3.0) + (est_output_tokens / 1_000_000 * 15.0)
    log.info(f"Estimated API cost for this run: ~${est_cost:.4f} "
             f"({est_input_tokens:,} input tokens, {est_output_tokens:,} output tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, choices=["chalearn", "persuade"])
    parser.add_argument("--dialect",  default="all",
                        help="Dialect key or 'all' (default: all)")
    parser.add_argument("--judge",    default=config.JUDGE_MODELS[0])
    args = parser.parse_args()

    if args.dialect == "all":
        dialect_keys = ["sae"] + config.DIALECTS
    else:
        dialect_keys = [args.dialect]

    run(args.dataset, dialect_keys, args.judge)
