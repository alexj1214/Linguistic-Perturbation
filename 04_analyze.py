"""
04_analyze.py — Compute score deltas, run stats, generate plots.

Usage:
    python 04_analyze.py --dataset chalearn
    python 04_analyze.py --dataset persuade

Reads:  outputs/scores/{dataset}_*_*.jsonl
Writes: outputs/logs/{dataset}_analysis.csv
        outputs/logs/{dataset}_analysis.png
"""

import argparse
import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config
from utils import ensure_dirs, get_logger

ensure_dirs()


def load_scores(dataset: str) -> pd.DataFrame:
    """Load all score JSONL files for a dataset into a single DataFrame."""
    records = []
    for path in config.SCORES_DIR.glob(f"{dataset}_*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("parsed_score") is not None:
                        records.append(r)
                except json.JSONDecodeError:
                    pass
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def run(dataset: str):
    log = get_logger("analyze", f"04_analyze_{dataset}.log")
    log.info(f"Loading scores for dataset: {dataset}")

    df = load_scores(dataset)
    if df.empty:
        log.error("No scored records found. Run 03_score.py first.")
        sys.exit(1)

    log.info(f"Loaded {len(df)} scored records  |  "
             f"dialects: {df['dialect'].unique().tolist()}  |  "
             f"judges: {df['judge_model'].unique().tolist()}")

    # ── Per-dialect mean scores ────────────────────────────────
    summary = (
        df.groupby(["judge_model", "dialect"])["parsed_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
    )
    log.info(f"\n{summary.to_string(index=False)}")

    # ── Score deltas vs SAE baseline ───────────────────────────
    # Pivot so each row is one (sample_id, judge_model)
    pivoted = df.pivot_table(
        index=["sample_id", "judge_model"],
        columns="dialect",
        values="parsed_score",
    ).reset_index()

    if "sae" not in pivoted.columns:
        log.warning("No SAE baseline scores found — skipping delta computation.")
        deltas_df = pd.DataFrame()
    else:
        non_sae = [c for c in pivoted.columns if c not in ("sample_id", "judge_model", "sae")]
        delta_rows = []
        for dialect in non_sae:
            col = pivoted[["sample_id", "judge_model", "sae", dialect]].dropna()
            delta = col[dialect] - col["sae"]
            delta_rows.append({
                "dialect":    dialect,
                "mean_delta": delta.mean(),
                "std_delta":  delta.std(),
                "n_pairs":    len(delta),
            })
        deltas_df = pd.DataFrame(delta_rows)
        log.info(f"\nScore deltas vs SAE:\n{deltas_df.to_string(index=False)}")

    # ── Wilcoxon signed-rank tests ─────────────────────────────
    test_rows = []
    if "sae" in pivoted.columns:
        for dialect in non_sae:
            col = pivoted[["sae", dialect]].dropna()
            if len(col) < 5:
                log.warning(f"Too few paired samples for {dialect} ({len(col)}) — skipping test.")
                continue
            stat, p = stats.wilcoxon(col["sae"], col[dialect], alternative="two-sided")
            test_rows.append({
                "dialect": dialect,
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_p05": p < 0.05,
                "n_pairs": len(col),
            })
        if test_rows:
            tests_df = pd.DataFrame(test_rows)
            log.info(f"\nWilcoxon signed-rank tests:\n{tests_df.to_string(index=False)}")
        else:
            tests_df = pd.DataFrame()
    else:
        tests_df = pd.DataFrame()

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = config.LOGS_DIR / f"{dataset}_analysis.csv"
    summary.to_csv(csv_path, index=False)
    log.info(f"Summary saved to {csv_path}")

    if not deltas_df.empty:
        delta_path = config.LOGS_DIR / f"{dataset}_deltas.csv"
        deltas_df.to_csv(delta_path, index=False)
        log.info(f"Deltas saved to {delta_path}")

    if not tests_df.empty:
        tests_path = config.LOGS_DIR / f"{dataset}_wilcoxon.csv"
        tests_df.to_csv(tests_path, index=False)
        log.info(f"Wilcoxon results saved to {tests_path}")

    # ── Plot: score distributions per dialect ──────────────────
    dialects_present = sorted(df["dialect"].unique())
    n_dialects = len(dialects_present)

    fig, axes = plt.subplots(1, n_dialects, figsize=(4 * n_dialects, 4), sharey=True)
    if n_dialects == 1:
        axes = [axes]

    score_min, score_max = (1, 5) if dataset == "chalearn" else (1, 6)
    bins = range(score_min, score_max + 2)

    for ax, dialect in zip(axes, dialects_present):
        scores = df[df["dialect"] == dialect]["parsed_score"].dropna()
        ax.hist(scores, bins=bins, align="left", color="steelblue", edgecolor="white")
        ax.set_title(dialect.upper())
        ax.set_xlabel("Score")
        ax.set_xlim(score_min - 0.5, score_max + 0.5)
        mean_val = scores.mean()
        ax.axvline(mean_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.2f}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{dataset.upper()} — Score distributions by dialect", fontsize=12)
    plt.tight_layout()

    plot_path = config.LOGS_DIR / f"{dataset}_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["chalearn", "persuade"])
    args = parser.parse_args()
    run(args.dataset)
