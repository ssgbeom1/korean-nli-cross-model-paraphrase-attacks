from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    BACKTRANSLATION_EVAL_DIR,
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
DISPLAY_NAMES = {
    "hyperclova_x": "HyperCLOVA X",
    "gemini": "Gemini",
    "gpt": "GPT",
    "claude_sonnet": "Claude Sonnet",
    "llm": "LLM pooled",
    "bt": "Back Translation",
}


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_llm_rows(input_dir: Path, allowed_ids: set[int] | None = None) -> pd.DataFrame:
    frames = []
    for generator in GENERATORS:
        generator_key = model_file_key(generator)
        generator_dir = input_dir / cross_model_generator_dirname(generator)
        for target in TARGETS:
            target_key = model_file_key(target)
            path = generator_dir / target_output_filename(target)
            df = pd.read_csv(path, encoding="utf-8-sig")
            if allowed_ids is not None:
                df = df[df["id"].astype(int).isin(allowed_ids)].copy()
            df = apply_original_cache(df, input_dir, target_key)
            df["condition_type"] = "llm"
            df["generator_key"] = generator_key
            df["target_key"] = target_key
            df["lpr_condition_key"] = generator_key
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return normalize_raw_rows(out)


def load_bt_rows(input_dir: Path, allowed_ids: set[int] | None = None) -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        target_key = model_file_key(target)
        path = input_dir / target_output_filename(target)
        df = pd.read_csv(path, encoding="utf-8-sig")
        if allowed_ids is not None:
            df = df[df["id"].astype(int).isin(allowed_ids)].copy()
        df["condition_type"] = "bt"
        df["generator_key"] = "backtranslation"
        df["target_key"] = target_key
        df["lpr_condition_key"] = target_key
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return normalize_raw_rows(out)


def normalize_raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["id", "label", "original_correct", "attack_success"]:
        out[column] = out[column].astype(int)
    out["label_name"] = out["label"].map(LABEL_NAMES)
    return out[
        [
            "condition_type",
            "generator_key",
            "target_key",
            "lpr_condition_key",
            "id",
            "label",
            "label_name",
            "original_correct",
            "attack_success",
        ]
    ]


def bt_ids(bt_dir: Path) -> set[int]:
    ids: set[int] | None = None
    for target in TARGETS:
        path = bt_dir / target_output_filename(target)
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["id"])
        current = set(df["id"].astype(int))
        ids = current if ids is None else ids.intersection(current)
    return ids or set()


def load_lpr(majority_results_path: Path) -> pd.DataFrame:
    results = pd.read_csv(majority_results_path, encoding="utf-8-sig")
    rows = []
    for (condition_type, condition_key, label, label_name), group in results.groupby(
        ["condition_type", "condition_key", "label", "label_name"],
        sort=True,
    ):
        n_total = len(group)
        n_preserved = int(group["label_preserved"].astype(bool).sum())
        resolved = group[group["label_preservation_status"].isin(["preserved", "not_preserved"])]
        n_resolved = len(resolved)
        n_preserved_resolved = int(resolved["label_preserved"].astype(bool).sum())
        all_lo, all_hi = wilson_ci(n_preserved, n_total)
        resolved_lo, resolved_hi = wilson_ci(n_preserved_resolved, n_resolved)
        rows.append(
            {
                "condition_type": condition_type,
                "lpr_condition_key": condition_key,
                "label": int(label),
                "label_name": label_name,
                "lpr_n_total": n_total,
                "lpr_n_preserved": n_preserved,
                "lpr_all": n_preserved / n_total if n_total else 0.0,
                "lpr_all_ci_lower": all_lo,
                "lpr_all_ci_upper": all_hi,
                "lpr_n_resolved": n_resolved,
                "lpr_n_preserved_resolved": n_preserved_resolved,
                "lpr_resolved": n_preserved_resolved / n_resolved if n_resolved else 0.0,
                "lpr_resolved_ci_lower": resolved_lo,
                "lpr_resolved_ci_upper": resolved_hi,
            }
        )
    return pd.DataFrame(rows)


def build_strata(raw_rows: pd.DataFrame, lpr: pd.DataFrame) -> pd.DataFrame:
    correct = raw_rows[raw_rows["original_correct"] == 1].copy()
    raw = (
        correct.groupby(
            ["condition_type", "generator_key", "target_key", "lpr_condition_key", "label", "label_name"],
            as_index=False,
        )
        .agg(raw_n=("id", "size"), raw_success=("attack_success", "sum"))
        .sort_values(["condition_type", "generator_key", "target_key", "label"])
    )
    raw["raw_asr"] = raw["raw_success"] / raw["raw_n"]
    merged = raw.merge(
        lpr,
        on=["condition_type", "lpr_condition_key", "label", "label_name"],
        how="left",
        validate="many_to_one",
    )
    if merged[["lpr_all", "lpr_resolved"]].isna().any().any():
        missing = merged[merged["lpr_all"].isna()][
            ["condition_type", "lpr_condition_key", "label", "label_name"]
        ].drop_duplicates()
        raise ValueError(f"Missing LPR rows:\n{missing.to_string(index=False)}")
    merged["adjusted_success_all"] = merged["raw_success"] * merged["lpr_all"]
    merged["adjusted_success_resolved"] = merged["raw_success"] * merged["lpr_resolved"]
    return merged


def bootstrap_adjusted(group: pd.DataFrame, lpr_column: str, lpr_n_column: str, seed: int, n_bootstrap: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    denominator = int(group["raw_n"].sum())
    if denominator <= 0:
        return (0.0, 0.0)
    samples = np.zeros(n_bootstrap, dtype=float)
    for _, row in group.iterrows():
        raw_n = int(row["raw_n"])
        raw_p = float(row["raw_asr"])
        lpr_n = int(row[lpr_n_column])
        lpr_p = float(row[lpr_column])
        raw_draws = rng.binomial(raw_n, raw_p, size=n_bootstrap) if raw_n else np.zeros(n_bootstrap)
        lpr_draws = rng.binomial(lpr_n, lpr_p, size=n_bootstrap) / lpr_n if lpr_n else np.zeros(n_bootstrap)
        samples += raw_draws * lpr_draws
    adjusted = samples / denominator
    lower, upper = np.percentile(adjusted, [2.5, 97.5])
    return (float(lower), float(upper))


def summarize_group(
    strata: pd.DataFrame,
    group_name: str,
    group_value: str,
    group: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    raw_n = int(group["raw_n"].sum())
    raw_success = int(group["raw_success"].sum())
    adjusted_success_all = float(group["adjusted_success_all"].sum())
    adjusted_success_resolved = float(group["adjusted_success_resolved"].sum())
    raw_lo, raw_hi = wilson_ci(raw_success, raw_n)
    adj_all_lo, adj_all_hi = bootstrap_adjusted(group, "lpr_all", "lpr_n_total", seed, n_bootstrap)
    adj_res_lo, adj_res_hi = bootstrap_adjusted(group, "lpr_resolved", "lpr_n_resolved", seed + 17, n_bootstrap)
    weighted_lpr_all = adjusted_success_all / raw_success if raw_success else 0.0
    weighted_lpr_resolved = adjusted_success_resolved / raw_success if raw_success else 0.0
    return {
        "group_name": group_name,
        "group_value": group_value,
        "raw_n_original_correct": raw_n,
        "raw_attack_success": raw_success,
        "raw_asr_pct": raw_success / raw_n * 100 if raw_n else 0.0,
        "raw_asr_ci_lower_pct": raw_lo * 100,
        "raw_asr_ci_upper_pct": raw_hi * 100,
        "weighted_lpr_all_pct": weighted_lpr_all * 100,
        "weighted_lpr_resolved_pct": weighted_lpr_resolved * 100,
        "adjusted_success_all": adjusted_success_all,
        "adjusted_asr_all_pct": adjusted_success_all / raw_n * 100 if raw_n else 0.0,
        "adjusted_asr_all_ci_lower_pct": adj_all_lo * 100,
        "adjusted_asr_all_ci_upper_pct": adj_all_hi * 100,
        "adjusted_success_resolved": adjusted_success_resolved,
        "adjusted_asr_resolved_pct": adjusted_success_resolved / raw_n * 100 if raw_n else 0.0,
        "adjusted_asr_resolved_ci_lower_pct": adj_res_lo * 100,
        "adjusted_asr_resolved_ci_upper_pct": adj_res_hi * 100,
    }


def make_summaries(strata: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    running_seed = seed

    for condition_type, group in strata.groupby("condition_type", sort=True):
        rows.append(summarize_group(strata, "condition_type", condition_type, group, running_seed, n_bootstrap))
        running_seed += 101

    llm = strata[strata["condition_type"] == "llm"]
    for generator_key, group in llm.groupby("generator_key", sort=True):
        rows.append(summarize_group(strata, "llm_generator", generator_key, group, running_seed, n_bootstrap))
        running_seed += 101
    for target_key, group in llm.groupby("target_key", sort=True):
        rows.append(summarize_group(strata, "llm_target", target_key, group, running_seed, n_bootstrap))
        running_seed += 101

    bt = strata[strata["condition_type"] == "bt"]
    for target_key, group in bt.groupby("target_key", sort=True):
        rows.append(summarize_group(strata, "bt_target", target_key, group, running_seed, n_bootstrap))
        running_seed += 101

    for (condition_type, label_name), group in strata.groupby(["condition_type", "label_name"], sort=True):
        rows.append(
            summarize_group(strata, f"{condition_type}_label", label_name, group, running_seed, n_bootstrap)
        )
        running_seed += 101

    return pd.DataFrame(rows)


def write_text_summary(path: Path, summary: pd.DataFrame, intersection: pd.DataFrame) -> None:
    lines = [
        "Human-Adjusted ASR Summary",
        "==========================",
        "",
        "Primary adjusted ASR treats U/TIE human-validation outcomes as not label-preserving.",
        "The resolved version excludes U/TIE outcomes as a sensitivity analysis.",
        "",
        "Main estimates:",
    ]
    main = summary[summary["group_name"].eq("condition_type")]
    for _, row in main.iterrows():
        lines.append(
            f"- {row.group_value}: raw ASR={row.raw_asr_pct:.2f}%, "
            f"LPR-weighted adjusted ASR={row.adjusted_asr_all_pct:.2f}% "
            f"[{row.adjusted_asr_all_ci_lower_pct:.2f}, {row.adjusted_asr_all_ci_upper_pct:.2f}], "
            f"resolved={row.adjusted_asr_resolved_pct:.2f}%"
        )
    lines.extend(["", "BT-intersection comparison:"])
    for _, row in intersection.iterrows():
        lines.append(
            f"- {row.group_value}: raw ASR={row.raw_asr_pct:.2f}%, "
            f"adjusted={row.adjusted_asr_all_pct:.2f}% "
            f"[{row.adjusted_asr_all_ci_lower_pct:.2f}, {row.adjusted_asr_all_ci_upper_pct:.2f}]"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human-adjusted ASR from majority annotations.")
    parser.add_argument("--majority_results", default="results/05_human_evaluation/analysis/human_validation_majority_results.csv")
    parser.add_argument("--cross_model_dir", default=CROSS_MODEL_EVAL_DIR)
    parser.add_argument("--bt_eval_dir", default=BACKTRANSLATION_EVAL_DIR)
    parser.add_argument("--output_dir", default=SUMMARY_TABLE_DIR)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lpr = load_lpr(Path(args.majority_results))

    llm_rows = load_llm_rows(Path(args.cross_model_dir))
    bt_rows = load_bt_rows(Path(args.bt_eval_dir))
    raw_rows = pd.concat([llm_rows, bt_rows], ignore_index=True)
    strata = build_strata(raw_rows, lpr)
    summary = make_summaries(strata, args.n_bootstrap, args.seed)

    ids = bt_ids(Path(args.bt_eval_dir))
    llm_intersection = load_llm_rows(Path(args.cross_model_dir), allowed_ids=ids)
    bt_intersection = load_bt_rows(Path(args.bt_eval_dir), allowed_ids=ids)
    intersection_strata = build_strata(pd.concat([llm_intersection, bt_intersection], ignore_index=True), lpr)
    intersection_summary = make_summaries(intersection_strata, args.n_bootstrap, args.seed + 999)
    intersection_summary = intersection_summary[intersection_summary["group_name"].eq("condition_type")].copy()

    strata.to_csv(output_dir / "human_adjusted_asr_components.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "human_adjusted_asr_summary.csv", index=False, encoding="utf-8-sig")
    intersection_summary.to_csv(
        output_dir / "human_adjusted_asr_bt_vs_llm_intersection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_text_summary(output_dir / "human_adjusted_asr_summary.txt", summary, intersection_summary)

    print("Human-adjusted ASR complete.")
    print(summary[summary["group_name"].eq("condition_type")].to_string(index=False))
    print("BT-intersection comparison:")
    print(intersection_summary.to_string(index=False))


if __name__ == "__main__":
    main()
