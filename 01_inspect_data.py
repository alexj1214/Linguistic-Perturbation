"""
01_inspect_data.py — Inspect ChaLearn and PERSUADE datasets.
Prints schema, stats, samples. Saves summary to outputs/logs/data_summary.txt.
Run with: /opt/anaconda3/envs/dialect_bias/bin/python 01_inspect_data.py
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_LOG = Path("outputs/logs/data_summary.txt")
OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)

CHALEARN_TRANSCRIPTS = Path("transcription_training.pkl")
CHALEARN_ANNOTATIONS  = Path("annotation_training.pkl")
PERSUADE_TRAIN        = Path("persuade_corpus_2.0_train.csv")
PERSUADE_TEST         = Path("persuade_corpus_2.0_test.csv")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

lines = []  # accumulate all output; written to log at the end

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def separator(title=""):
    bar = "=" * 60
    if title:
        log(f"\n{bar}")
        log(f"  {title}")
        log(bar)
    else:
        log(bar)

# ──────────────────────────────────────────────────────────────
# 1. ChaLearn Transcripts
# ──────────────────────────────────────────────────────────────
separator("CHALEARN TRANSCRIPTS  (transcription_training.pkl)")

with open(CHALEARN_TRANSCRIPTS, "rb") as f:
    transcripts_raw = pickle.load(f)

log(f"Top-level type : {type(transcripts_raw)}")

# Discover structure
if isinstance(transcripts_raw, dict):
    log(f"Number of keys : {len(transcripts_raw)}")
    sample_key = next(iter(transcripts_raw))
    log(f"Sample key     : {repr(sample_key)}")
    log(f"Value type     : {type(transcripts_raw[sample_key])}")
    log(f"Sample value   : {repr(transcripts_raw[sample_key])[:200]}")

    # Build flat list of (id, text) pairs — handle nested dicts/lists
    samples = []
    for k, v in transcripts_raw.items():
        if isinstance(v, str):
            samples.append((k, v))
        elif isinstance(v, dict):
            # might be {video_id: {transcript: "..."}} or similar
            log(f"\nNested dict keys: {list(v.keys())[:10]}")
            for subk, subv in v.items():
                if isinstance(subv, str):
                    samples.append((f"{k}/{subk}", subv))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    samples.append((f"{k}[{i}]", item))

elif isinstance(transcripts_raw, list):
    log(f"Number of items : {len(transcripts_raw)}")
    log(f"Item[0] type    : {type(transcripts_raw[0])}")
    log(f"Item[0] value   : {repr(transcripts_raw[0])[:200]}")
    samples = list(enumerate(transcripts_raw)) if all(isinstance(x, str) for x in transcripts_raw) else []
    if not samples:
        log("WARNING: list items are not plain strings; inspect further.")
else:
    log(f"Unexpected type: {type(transcripts_raw)}")
    samples = []

if samples:
    texts = [text for _, text in samples]
    word_counts = [len(t.split()) for t in texts]
    wc = np.array(word_counts)

    log(f"\n--- Word-count stats ({len(samples)} transcripts) ---")
    log(f"  Mean   : {wc.mean():.1f}")
    log(f"  Median : {np.median(wc):.1f}")
    log(f"  Min    : {wc.min()}")
    log(f"  Max    : {wc.max()}")

    # Warnings
    if np.median(wc) < 30:
        log("WARNING: Median word count < 30 — transcripts may be very short.")

    has_punct = sum(1 for t in texts if any(c in t for c in ".!?,;"))
    if has_punct / len(texts) < 0.5:
        log("WARNING: Fewer than 50% of transcripts contain punctuation.")

    log(f"\n--- 3 random sample transcripts ---")
    chosen = random.sample(samples, min(3, len(samples)))
    for sid, text in chosen:
        wc_s = len(text.split())
        log(f"\n  [ID: {sid}]  ({wc_s} words)")
        log(f"  {repr(text[:400])}")
else:
    log("WARNING: Could not extract (id, text) pairs — check raw structure above.")

# ──────────────────────────────────────────────────────────────
# 2. ChaLearn Annotations
# ──────────────────────────────────────────────────────────────
separator("CHALEARN ANNOTATIONS  (annotation_training.pkl)")

with open(CHALEARN_ANNOTATIONS, "rb") as f:
    try:
        annotations_raw = pickle.load(f)
    except UnicodeDecodeError:
        f.seek(0)
        annotations_raw = pickle.load(f, encoding="latin1")

log(f"Top-level type : {type(annotations_raw)}")

if isinstance(annotations_raw, dict):
    log(f"Number of keys : {len(annotations_raw)}")
    sample_key = next(iter(annotations_raw))
    log(f"Sample key     : {repr(sample_key)}")
    sample_val = annotations_raw[sample_key]
    log(f"Value type     : {type(sample_val)}")
    if isinstance(sample_val, dict):
        log(f"Annotation keys: {list(sample_val.keys())}")
        log(f"Full sample    : {sample_val}")
    elif isinstance(sample_val, (list, np.ndarray)):
        log(f"Length         : {len(sample_val)}")
        log(f"First item     : {sample_val[0] if len(sample_val) else 'empty'}")
    else:
        log(f"Value          : {repr(sample_val)[:200]}")

elif isinstance(annotations_raw, pd.DataFrame):
    log(f"Shape    : {annotations_raw.shape}")
    log(f"Columns  : {list(annotations_raw.columns)}")
    log(f"Dtypes:\n{annotations_raw.dtypes.to_string()}")
    log(f"\nOne sample row:\n{annotations_raw.iloc[0].to_string()}")

elif isinstance(annotations_raw, list):
    log(f"Number of items : {len(annotations_raw)}")
    log(f"Item[0] type    : {type(annotations_raw[0])}")
    log(f"Item[0]         : {repr(annotations_raw[0])[:200]}")

else:
    log(f"Unexpected type: {type(annotations_raw)}")

# Check for hireability / personality keys
required_labels = ["interview", "hireability", "extraversion", "neuroticism",
                   "agreeableness", "conscientiousness", "openness"]
if isinstance(annotations_raw, dict):
    flat_val = next(iter(annotations_raw.values()))
    if isinstance(flat_val, dict):
        missing = [k for k in required_labels if k not in flat_val]
        if missing:
            log(f"\nWARNING: Expected label keys not found: {missing}")
        else:
            log(f"\nAll expected label keys present: {required_labels}")

# ──────────────────────────────────────────────────────────────
# 3. PERSUADE Corpus
# ──────────────────────────────────────────────────────────────
for label, path in [("TRAIN", PERSUADE_TRAIN), ("TEST", PERSUADE_TEST)]:
    separator(f"PERSUADE {label}  ({path.name})")

    df = pd.read_csv(path, low_memory=False)
    log(f"Shape   : {df.shape}")
    log(f"Columns : {list(df.columns)}")
    log(f"\nDtypes:\n{df.dtypes.to_string()}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log("\nNo missing values.")
    else:
        log(f"\nMissing values:\n{missing.to_string()}")

    # Identify score column(s)
    score_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["score", "holistic", "grade", "rating"]
    )]
    log(f"\nScore column candidates : {score_candidates}")
    for sc in score_candidates:
        log(f"  {sc}: min={df[sc].min()}, max={df[sc].max()}, "
            f"mean={df[sc].mean():.2f}, nunique={df[sc].nunique()}")

    # Text column candidates
    text_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["text", "essay", "full_text", "response"]
    )]
    log(f"\nText column candidates  : {text_candidates}")
    if text_candidates:
        tc = text_candidates[0]
        wc = df[tc].dropna().apply(lambda x: len(str(x).split()))
        log(f"  '{tc}' word count — mean: {wc.mean():.0f}, "
            f"median: {wc.median():.0f}, min: {wc.min()}, max: {wc.max()}")

    # 2 sample rows
    log(f"\n--- 2 sample rows ---")
    for _, row in df.sample(2, random_state=RANDOM_SEED).iterrows():
        log(row.to_string())
        log("")

# ──────────────────────────────────────────────────────────────
# Write log
# ──────────────────────────────────────────────────────────────
with open(OUTPUT_LOG, "w") as f:
    f.write("\n".join(lines))

print(f"\nSummary saved to {OUTPUT_LOG}")
