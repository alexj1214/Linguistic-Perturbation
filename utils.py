"""
utils.py — Shared helpers for the dialect bias pipeline.
"""

import pickle
import random
import time
import logging
from pathlib import Path

import pandas as pd

from config import (
    CHALEARN_TRANSCRIPTS, CHALEARN_ANNOTATIONS, PERSUADE_TRAIN,
    TRANSFORMED_DIR, SCORES_DIR, LOGS_DIR,
)

# Status codes worth retrying. Auth (401), bad-request (400), and content-policy
# refusals (403) are NOT retryable — they will never succeed on a retry.
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


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


def _is_retryable(exc: Exception) -> bool:
    """
    True only for transient errors worth retrying:
    rate limits, server errors, and connection problems.
    Auth / bad-request / content-policy errors are NOT retried.
    """
    # Anthropic/OpenAI SDK errors expose .status_code
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUSES:
        return True
    # Network-level: anthropic.APIConnectionError, openai.APIConnectionError,
    # plain ConnectionError, TimeoutError, etc.
    name = type(exc).__name__
    if name in {
        "APIConnectionError", "APITimeoutError",
        "RateLimitError", "InternalServerError", "APIStatusError",
        "ConnectionError", "TimeoutError",
    }:
        # APIStatusError still has a status_code; only retry if it's retryable.
        if name == "APIStatusError" and status is not None:
            return status in _RETRYABLE_STATUSES
        return True
    return False


def safe_api_call(fn, max_retries: int = 4, base_delay: float = 1.0,
                  max_delay: float = 30.0):
    """
    Call fn() with exponential backoff + jitter on transient failures.
    Non-retryable errors (auth, bad request, content policy) raise immediately.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            wait = min(base_delay * (2 ** attempt), max_delay)
            wait *= 1 + random.random() * 0.3   # 0–30% jitter
            time.sleep(wait)
    raise last_exc  # unreachable, but keeps type-checkers happy
