"""
02_transform.py — Apply Multi-VALUE dialect transformations.

Usage:
    python 02_transform.py --dataset chalearn --n_samples 5
    python 02_transform.py --dataset persuade --n_samples 5
    python 02_transform.py --dataset chalearn          # all samples

Output: outputs/transformed/{dataset}_{dialect}.jsonl
        One JSON line per transformation:
        {"sample_id", "dialect", "original_text", "transformed_text",
         "rules_fired", "n_rules", "changed"}

IMPORTANT: Must run with /opt/anaconda3/envs/dialect_bias/bin/python
"""

# ── Patch torch.load before stanza is imported via multivalue ──
import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

import argparse
import json
import random
import sys
from pathlib import Path

from multivalue import Dialects

import config
from utils import (
    ensure_dirs, get_logger,
    load_chalearn_transcripts, load_persuade,
)

ensure_dirs()
log = get_logger("transform", "02_transform.log")


# ── Dialect loader ─────────────────────────────────────────────

def get_dialect_instance(dialect_key: str):
    """Instantiate a Multi-VALUE dialect class by config key."""
    class_name = config.DIALECT_CLASS_MAP.get(dialect_key)
    if class_name is None:
        return None  # SAE baseline — no transformation
    cls = getattr(Dialects, class_name, None)
    if cls is None:
        raise ValueError(f"Dialect class '{class_name}' not found in multivalue.Dialects")
    return cls()


# ── Core transform ─────────────────────────────────────────────

import re as _re

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple regex.
    Multi-VALUE asserts spaCy and Stanza agree on sentence count;
    processing one sentence at a time avoids that mismatch entirely.
    """
    # Split on sentence-ending punctuation followed by whitespace or end
    parts = _re.split(r'(?<=[.!?])\s+', text.strip())
    # Remove empties, preserve non-empty parts
    return [p for p in parts if p.strip()]


def transform_one(text: str, dialect_obj) -> tuple[str, dict]:
    """
    Apply dialect_obj.transform() sentence-by-sentence.
    Concatenating per-sentence results avoids the spaCy/Stanza
    sentence-count assertion that fails on multi-sentence texts.
    Returns (transformed_text, rules_fired_dict).
    """
    if dialect_obj is None:
        return text, {}

    sentences = _split_sentences(text)
    if not sentences:
        return text, {}

    transformed_parts = []
    all_rules: dict = {}

    for i, sent in enumerate(sentences):
        dialect_obj.executed_rules = {}
        try:
            transformed_sent = dialect_obj.transform(sent)
        except AssertionError:
            # spaCy/Stanza still disagree on this sentence — keep original
            transformed_sent = sent
        transformed_parts.append(transformed_sent)
        # Sentence index keeps rule keys unique across sentences without
        # any character-offset arithmetic.
        for k, v in dialect_obj.executed_rules.items():
            all_rules[f"s{i}:{k}"] = v

    transformed_text = " ".join(transformed_parts)
    return transformed_text, all_rules


# ── Dataset loaders → list of (sample_id, text) ───────────────

def get_chalearn_samples(n: int | None) -> list[tuple[str, str]]:
    transcripts = load_chalearn_transcripts()
    samples = [(k, v) for k, v in transcripts.items() if v and v.strip()]
    random.shuffle(samples)
    if n is not None:
        samples = samples[:n]
    return samples


def get_persuade_samples(n: int | None) -> list[tuple[str, str]]:
    df = load_persuade()
    df = df[df["full_text"].notna() & (df["full_text"].str.strip() != "")]
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    if n is not None:
        df = df.iloc[:n]
    return list(zip(df["essay_id"].astype(str), df["full_text"]))


# ── Main ───────────────────────────────────────────────────────

def run(dataset: str, n_samples: int | None):
    random.seed(config.RANDOM_SEED)

    if dataset == "chalearn":
        samples = get_chalearn_samples(n_samples)
    elif dataset == "persuade":
        samples = get_persuade_samples(n_samples)
    else:
        log.error(f"Unknown dataset: {dataset}")
        sys.exit(1)

    log.info(f"Dataset: {dataset}  |  Samples: {len(samples)}  |  "
             f"Dialects: sae + {config.DIALECTS}")

    all_dialects = ["sae"] + config.DIALECTS

    for dialect_key in all_dialects:
        out_path = config.TRANSFORMED_DIR / f"{dataset}_{dialect_key}.jsonl"

        # Resume: find already-done sample IDs
        done_ids: set[str] = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["sample_id"])
                    except Exception:
                        pass

        todo = [(sid, txt) for sid, txt in samples if sid not in done_ids]
        if not todo:
            log.info(f"[{dialect_key}] All {len(samples)} samples already done — skipping.")
            continue

        log.info(f"[{dialect_key}] Loading dialect... ({len(todo)} samples to process)")
        try:
            dialect_obj = get_dialect_instance(dialect_key)
        except Exception as e:
            log.error(f"[{dialect_key}] Failed to load dialect: {e}")
            continue

        n_zero_rules = 0
        n_errors = 0

        with open(out_path, "a") as fout:
            for i, (sample_id, original_text) in enumerate(todo):
                try:
                    transformed_text, rules_fired = transform_one(original_text, dialect_obj)
                except Exception as e:
                    log.warning(f"[{dialect_key}] sample={sample_id!r}: transform failed: {e}")
                    n_errors += 1
                    continue

                n_rules = len(rules_fired)
                changed = transformed_text != original_text

                if dialect_key != "sae" and n_rules == 0:
                    n_zero_rules += 1
                    log.debug(f"[{dialect_key}] sample={sample_id!r}: 0 rules fired")

                record = {
                    "sample_id":        sample_id,
                    "dialect":          dialect_key,
                    "original_text":    original_text,
                    "transformed_text": transformed_text,
                    "rules_fired":      rules_fired,
                    "n_rules":          n_rules,
                    "changed":          changed,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                if (i + 1) % 50 == 0:
                    log.info(f"[{dialect_key}] {i + 1}/{len(todo)} done")

        log.info(f"[{dialect_key}] Complete — zero-rule samples: {n_zero_rules}, "
                 f"errors: {n_errors}  →  {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, choices=["chalearn", "persuade"])
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (default: N_PILOT_SAMPLES from config)")
    args = parser.parse_args()

    n = args.n_samples
    if n is None:
        n = config.N_PILOT_SAMPLES  # default to pilot size; override with --n_samples

    run(args.dataset, n)
