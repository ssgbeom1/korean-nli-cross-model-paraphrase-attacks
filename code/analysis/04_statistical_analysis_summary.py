from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CROSS_MODEL_EVAL_DIR,
    GENERATORS,
    SUMMARY_TABLE_DIR,
    TARGETS,
    cross_model_generator_dirname,
    model_file_key,
    target_output_filename,
)
from utils.original_cache import apply_original_cache  # noqa: E402


LABEL_NAMES = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


def load_cross_model_results(input_dir: str) -> pd.DataFrame:
    base_dir = Path(input_dir)
    frames = []
    for generator in GENERATORS:
        generator_dir = base_dir / cross_model_generator_dirname(generator)
        for target in TARGETS:
            path = generator_dir / target_output_filename(target)
            if not path.exists():
                continue
            df = pd.read_csv(path, encoding="utf-8-sig")
            target_key = model_file_key(target)
            df = apply_original_cache(df, base_dir, target_key)
            df["generator_key"] = model_file_key(generator)
            df["target_key"] = target_key
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cross-model evaluation CSVs found under {input_dir}")
    combined = pd.concat(frames, ignore_index=True)
    required = {"id", "label", "original_correct", "attack_success", "generator_key", "target_key"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return combined


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_group(df: pd.DataFrame, category: str, name: str) -> dict:
    correct = df[df["original_correct"] == 1]
    n = len(correct)
    successes = int(correct["attack_success"].sum())
    asr = successes / n if n else 0.0
    lower, upper = wilson_ci(successes, n)
    return {
        "category": category,
        "name": name,
        "n_original_correct": n,
        "n_attack_success": successes,
        "asr_pct": asr * 100,
        "wilson_ci_lower_pct": lower * 100,
        "wilson_ci_upper_pct": upper * 100,
    }


def group_summaries(combined: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_group(combined, "Overall", "Total")]
    for generator, subset in combined.groupby("generator_key", sort=True):
        rows.append(summarize_group(subset, "Generator", generator))
    for target, subset in combined.groupby("target_key", sort=True):
        rows.append(summarize_group(subset, "Target", target))
    for label, subset in combined.groupby("label", sort=True):
        rows.append(summarize_group(subset, "Label", LABEL_NAMES.get(int(label), str(label))))
    return pd.DataFrame(rows)


def asr_for_subset(df: pd.DataFrame) -> float:
    correct = df[df["original_correct"] == 1]
    if len(correct) == 0:
        return 0.0
    return float(correct["attack_success"].mean())


def cluster_bootstrap_diff(
    combined: pd.DataFrame,
    column: str,
    left: str | int,
    right: str | int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    correct = combined[combined["original_correct"] == 1].copy()
    ids = np.array(sorted(correct["id"].unique()))

    grouped = (
        correct[correct[column].isin([left, right])]
        .groupby(["id", column])["attack_success"]
        .agg(["count", "sum"])
    )
    full_index = pd.MultiIndex.from_product([ids, [left, right]], names=["id", column])
    grouped = grouped.reindex(full_index, fill_value=0).reset_index()

    left_rows = grouped[grouped[column] == left].sort_values("id")
    right_rows = grouped[grouped[column] == right].sort_values("id")
    left_n = left_rows["count"].to_numpy(dtype=float)
    left_success = left_rows["sum"].to_numpy(dtype=float)
    right_n = right_rows["count"].to_numpy(dtype=float)
    right_success = right_rows["sum"].to_numpy(dtype=float)

    left_asr = left_success.sum() / left_n.sum() if left_n.sum() else 0.0
    right_asr = right_success.sum() / right_n.sum() if right_n.sum() else 0.0
    observed = left_asr - right_asr

    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, len(ids), size=len(ids))
        boot_left_n = left_n[sampled].sum()
        boot_right_n = right_n[sampled].sum()
        boot_left = left_success[sampled].sum() / boot_left_n if boot_left_n else 0.0
        boot_right = right_success[sampled].sum() / boot_right_n if boot_right_n else 0.0
        diffs[i] = boot_left - boot_right

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    p_left = float((np.sum(diffs <= 0.0) + 1) / (n_bootstrap + 1))
    p_right = float((np.sum(diffs >= 0.0) + 1) / (n_bootstrap + 1))
    p_two_sided = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "contrast": f"{left} minus {right}",
        "diff_pct": observed * 100,
        "bootstrap_ci_lower_pct": float(lower * 100),
        "bootstrap_ci_upper_pct": float(upper * 100),
        "bootstrap_p_two_sided": p_two_sided,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def pairwise_summaries(combined: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    specs = [
        ("Generator", "generator_key", sorted(combined["generator_key"].unique())),
        ("Target", "target_key", sorted(combined["target_key"].unique())),
        ("Label", "label", [0, 1, 2]),
    ]
    rows = []
    for category, column, values in specs:
        category_rows = []
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                row = cluster_bootstrap_diff(combined, column, left, right, n_bootstrap, seed + len(rows))
                row["category"] = category
                row["left"] = LABEL_NAMES.get(left, left)
                row["right"] = LABEL_NAMES.get(right, right)
                row["contrast"] = f"{row['left']} minus {row['right']}"
                category_rows.append(row)
        adjusted = holm_adjust([row["bootstrap_p_two_sided"] for row in category_rows])
        for row, p_holm in zip(category_rows, adjusted):
            row["holm_p"] = p_holm
            rows.append(row)
    return pd.DataFrame(rows)


def write_manuscript_summary(group_df: pd.DataFrame, pairwise_df: pd.DataFrame, output_path: Path) -> None:
    def pick(category: str, name: str) -> pd.Series:
        return group_df[(group_df["category"] == category) & (group_df["name"] == name)].iloc[0]

    overall = pick("Overall", "Total")
    neutral = pick("Label", "Neutral")
    entailment = pick("Label", "Entailment")
    contradiction = pick("Label", "Contradiction")
    lines = [
        "Statistical analysis summary",
        "",
        (
            f"Overall ASR was {overall.asr_pct:.2f}% "
            f"({int(overall.n_attack_success)}/{int(overall.n_original_correct)}), "
            f"with Wilson 95% CI [{overall.wilson_ci_lower_pct:.2f}%, "
            f"{overall.wilson_ci_upper_pct:.2f}%]."
        ),
        (
            f"Label-wise ASR was {entailment.asr_pct:.2f}% for Entailment, "
            f"{neutral.asr_pct:.2f}% for Neutral, and "
            f"{contradiction.asr_pct:.2f}% for Contradiction."
        ),
        "",
        "Pairwise bootstrap comparisons use sample-id cluster resampling and Holm correction within each comparison family.",
        "",
        "Manuscript note: the four-model accuracy-vulnerability correlation should be described as a descriptive observation only, not as an inferential claim.",
    ]
    if not pairwise_df.empty:
        strongest = pairwise_df.sort_values("holm_p").head(5)
        lines.append("")
        lines.append("Smallest Holm-adjusted pairwise p-values:")
        for _, row in strongest.iterrows():
            lines.append(
                f"- {row['category']}: {row['contrast']}; diff={row['diff_pct']:.2f} pp, "
                f"95% bootstrap CI [{row['bootstrap_ci_lower_pct']:.2f}, "
                f"{row['bootstrap_ci_upper_pct']:.2f}], Holm p={row['holm_p']:.4f}"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate statistical summaries for cross-model ASR results.")
    parser.add_argument("--input_dir", default=CROSS_MODEL_EVAL_DIR)
    parser.add_argument("--output_dir", default=SUMMARY_TABLE_DIR)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    combined = load_cross_model_results(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_df = group_summaries(combined)
    pairwise_df = pairwise_summaries(combined, args.n_bootstrap, args.seed)

    group_path = output_dir / "statistical_analysis_group_summary.csv"
    pairwise_path = output_dir / "statistical_analysis_pairwise.csv"
    text_path = output_dir / "statistical_analysis_summary.txt"
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    write_manuscript_summary(group_df, pairwise_df, text_path)

    print(f"Saved: {group_path}")
    print(f"Saved: {pairwise_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
