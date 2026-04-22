"""
config.py — Central configuration for the dialect bias pipeline.
"""

from pathlib import Path

# ── Dialect settings ──────────────────────────────────────────
# Multi-VALUE class names (from multivalue.Dialects)
DIALECTS = ["indian", "aave", "singapore"]

DIALECT_CLASS_MAP = {
    "indian":    "IndianDialect",
    "aave":      "AfricanAmericanVernacular",
    "singapore": "ColloquialSingaporeDialect",
    "sae":       None,  # baseline: untransformed Standard American English
}

# ── Judge models ──────────────────────────────────────────────
JUDGE_MODELS = ["claude-sonnet-4-5"]

# ── Sample sizes ──────────────────────────────────────────────
N_PILOT_SAMPLES = 5
N_FULL_SAMPLES  = None   # None = all available samples

# ── Reproducibility ───────────────────────────────────────────
RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR    = Path(".")
OUTPUTS_DIR = Path("outputs")

CHALEARN_TRANSCRIPTS = DATA_DIR / "transcription_training.pkl"
CHALEARN_ANNOTATIONS = DATA_DIR / "annotation_training.pkl"
PERSUADE_TRAIN       = DATA_DIR / "persuade_corpus_2.0_train.csv"

TRANSFORMED_DIR = OUTPUTS_DIR / "transformed"
SCORES_DIR      = OUTPUTS_DIR / "scores"
LOGS_DIR        = OUTPUTS_DIR / "logs"
PROMPTS_DIR     = Path("prompts")
