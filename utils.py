"""
utils.py — Shared helpers for the dialect bias pipeline.
"""

import pickle
import time
import logging
from pathlib import Path

import pandas as pd

from config import (
    CHALEARN_TRANSCRIPTS, CHALEARN_ANNOTATIONS, PERSUADE_TRAIN,
    TRANSFORMED_DIR, SCORES_DIR, LOGS_DIR,
)


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [TRANSFORMED_DIR, SCORES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: str) -> logging.Logger:
    """Return a logger that writes to both stdout and a file."""
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOGS_DIR / log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_chalearn_transcripts(path=None) -> dict:
    """
    Load ChaLearn transcription pickle.
    Returns dict: {clip_id (str) -> transcript (str)}
    """
    path = path or CHALEARN_TRANSCRIPTS
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    return data


def load_chalearn_annotations(path=None) -> dict:
    """
    Load ChaLearn annotation pickle.
    Returns dict: {trait (str) -> {clip_id (str) -> score (float)}}
    Traits: extraversion, neuroticism, agreeableness, conscientiousness,
            openness, interview
    """
    path = path or CHALEARN_ANNOTATIONS
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    return data


def load_persuade(path=None) -> pd.DataFrame:
    """
    Load PERSUADE train CSV, deduplicated to one row per essay.
    Returns DataFrame with at minimum: essay_id, full_text, holistic_essay_score.
    """
    path = path or PERSUADE_TRAIN
    df = pd.read_csv(path, low_memory=False)
    df = df.drop_duplicates("essay_id").reset_index(drop=True)
    assert "full_text" in df.columns, "Missing 'full_text' column"
    assert "holistic_essay_score" in df.columns, "Missing 'holistic_essay_score' column"
    return df


def safe_api_call(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call fn() with exponential backoff on failure.
    Raises the last exception if all retries fail.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            wait = base_delay * (2 ** attempt)
            time.sleep(wait)
    raise last_exc
