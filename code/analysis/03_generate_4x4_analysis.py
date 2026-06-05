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
LABEL_ORDER = [0, 1, 2]


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(np.mean(values))
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return (point, float(lower), float(upper))


def load_results(input_dir: Path, expected_rows: int) -> pd.DataFrame:
    frames = []
    problems = []
    for generator in GENERATORS:
        generator_key = model_file_key(generator)
        generator_dir = input_dir / cross_model_generator_dirname(generator)
        for target in TARGETS:
            target_key = model_file_key(target)
            path = generator_dir / target_output_filename(target)
            if not path.exists():
                problems.append(f"Missing file: {path}")
                continue
            df = pd.read_csv(path, encoding="utf-8-sig")
            if len(df) != expected_rows:
                problems.append(f"Unexpected row count in {path}: {len(df)} != {expected_rows}")
            for column in ["id", "label", "pred_original", "pred_attacked", "original_correct", "attacked_correct", "attack_success"]:
                if column not in df.columns:
                    problems.append(f"Missing column in {path}: {column}")
            if "id" in df.columns and df["id"].duplicated().any():
                problems.append(f"Duplicate ids in {path}")
            if "pred_original" in df.columns and df["pred_original"].isna().any():
                problems.append(f"Missing pred_original in {path}")
            if "pred_attacked" in df.columns and df["pred_attacked"].isna().any():
                problems.append(f"Missing pred_attacked in {path}")
            df = apply_original_cache(df, input_dir, target_key)
            df = df.copy()
            df["generator"] = generator_key
            df["target"] = target_key
            df["source_file"] = str(path)
            frames.append(df)
    if problems:
        raise ValueError("\n".join(problems))
    if not frames:
        raise FileNotFoundError(f"No 4x4 outputs found under {input_dir}")
    combined = pd.concat(frames, ignore_index=True)
    for column in ["label", "pred_original", "pred_attacked", "original_correct", "attacked_correct", "attack_success"]:
        combined[column] = combined[column].astype(int)
    return combined


def summarize_asr(df: pd.DataFrame, **keys: str) -> dict:
    original_correct = df[df["original_correct"] == 1]
    n_rows = len(df)
    n_original_correct = len(original_correct)
    n_attack_success = int(original_correct["attack_success"].sum())
    asr = n_attack_success / n_original_correct if n_original_correct else 0.0
    wilson_lower, wilson_upper = wilson_ci(n_attack_success, n_original_correct)
    return {
        **keys,
        "rows": n_rows,
        "n_original_correct": n_original_correct,
        "n_attack_success": n_attack_success,
        "original_accuracy": float(df["original_correct"].mean()) if n_rows else 0.0,
        "attacked_accuracy": float(df["attacked_correct"].mean()) if n_rows else 0.0,
        "asr": asr,
        "asr_pct": asr * 100,
        "wilson_ci_lower_pct": wilson_lower * 100,
        "wilson_ci_upper_pct": wilson_upper * 100,
    }


def build_cell_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for generator in [model_file_key(g) for g in GENERATORS]:
        for target in [model_file_key(t) for t in TARGETS]:
            subset = combined[(combined["generator"] == generator) & (combined["target"] == target)]
            rows.append(summarize_asr(subset, generator=generator, target=target))
    return pd.DataFrame(rows)


def build_group_summary(combined: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = [summarize_asr(combined, category="Overall", name="Total")]
    for generator, subset in combined.groupby("generator", sort=True):
        rows.append(summarize_asr(subset, category="Generator", name=generator))
    for target, subset in combined.groupby("target", sort=True):
        rows.append(summarize_asr(subset, category="Target", name=target))
    for label in LABEL_ORDER:
        subset = combined[combined["label"] == label]
        rows.append(summarize_asr(subset, category="Label", name=LABEL_NAMES[label]))
    out = pd.DataFrame(rows)
    boot_rows = []
    for idx, row in out.iterrows():
        if row["category"] == "Overall":
            subset = combined
        elif row["category"] == "Generator":
            subset = combined[combined["generator"] == row["name"]]
        elif row["category"] == "Target":
            subset = combined[combined["target"] == row["name"]]
        else:
            label = {v: k for k, v in LABEL_NAMES.items()}[row["name"]]
            subset = combined[combined["label"] == label]
        values = subset[subset["original_correct"] == 1]["attack_success"].to_numpy()
        point, lower, upper = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=42 + idx)
        boot_rows.append(
            {
                "bootstrap_asr_pct": point * 100,
                "bootstrap_ci_lower_pct": lower * 100,
                "bootstrap_ci_upper_pct": upper * 100,
            }
        )
    return pd.concat([out, pd.DataFrame(boot_rows)], axis=1)


def build_labelwise_by_cell(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (generator, target, label), subset in combined.groupby(["generator", "target", "label"], sort=True):
        rows.append(summarize_asr(subset, generator=generator, target=target, label=label, label_name=LABEL_NAMES[int(label)]))
    return pd.DataFrame(rows)


def build_transition_summary(combined: pd.DataFrame) -> pd.DataFrame:
    attack_success = combined[combined["attack_success"] == 1]
    rows = []
    for gold in LABEL_ORDER:
        gold_subset = attack_success[attack_success["label"] == gold]
        total_from_gold = len(gold_subset)
        for pred in LABEL_ORDER:
            if pred == gold:
                continue
            count = int((gold_subset["pred_attacked"] == pred).sum())
            pct = count / total_from_gold * 100 if total_from_gold else 0.0
            rows.append(
                {
                    "gold_label": gold,
                    "gold_name": LABEL_NAMES[gold],
                    "pred_label": pred,
                    "pred_name": LABEL_NAMES[pred],
                    "count": count,
                    "total_from_gold": total_from_gold,
                    "percentage": pct,
                }
            )
    return pd.DataFrame(rows)


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


def cluster_bootstrap_diff(
    combined: pd.DataFrame,
    column: str,
    left: str | int,
    right: str | int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    correct = combined[combined["original_correct"] == 1].copy()
    ids = np.array(sorted(correct["id"].unique()))
    selected = correct[correct[column].isin([left, right])]
    grouped = selected.groupby(["id", column])["attack_success"].agg(["count", "sum"])
    full_index = pd.MultiIndex.from_product([ids, [left, right]], names=["id", column])
    grouped = grouped.reindex(full_index, fill_value=0).reset_index()
    left_rows = grouped[grouped[column] == left].sort_values("id")
    right_rows = grouped[grouped[column] == right].sort_values("id")
    left_n = left_rows["count"].to_numpy(dtype=float)
    left_success = left_rows["sum"].to_numpy(dtype=float)
    right_n = right_rows["count"].to_numpy(dtype=float)
    right_success = right_rows["sum"].to_numpy(dtype=float)
    observed_left = left_success.sum() / left_n.sum() if left_n.sum() else 0.0
    observed_right = right_success.sum() / right_n.sum() if right_n.sum() else 0.0
    observed = observed_left - observed_right

    rng = np.random.default_rng(seed)
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
    return {
        "left": left,
        "right": right,
        "left_asr_pct": observed_left * 100,
        "right_asr_pct": observed_right * 100,
        "diff_pct": observed * 100,
        "bootstrap_ci_lower_pct": float(lower * 100),
        "bootstrap_ci_upper_pct": float(upper * 100),
        "bootstrap_p_two_sided": min(1.0, 2.0 * min(p_left, p_right)),
    }


def build_pairwise_summary(combined: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    specs = [
        ("Generator", "generator", [model_file_key(g) for g in GENERATORS]),
        ("Target", "target", [model_file_key(t) for t in TARGETS]),
        ("Label", "label", LABEL_ORDER),
    ]
    rows = []
    seed = 20260526
    for category, column, values in specs:
        local_rows = []
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                row = cluster_bootstrap_diff(combined, column, left, right, n_bootstrap=n_bootstrap, seed=seed)
                seed += 1
                row["category"] = category
                row["left_name"] = LABEL_NAMES.get(left, left)
                row["right_name"] = LABEL_NAMES.get(right, right)
                row["contrast"] = f"{row['left_name']} minus {row['right_name']}"
                local_rows.append(row)
        adjusted = holm_adjust([row["bootstrap_p_two_sided"] for row in local_rows])
        for row, p_holm in zip(local_rows, adjusted):
            row["holm_p"] = p_holm
            rows.append(row)
    return pd.DataFrame(rows)


def write_text_summary(output_path: Path, cell_df: pd.DataFrame, group_df: pd.DataFrame, transition_df: pd.DataFrame) -> None:
    overall = group_df[(group_df["category"] == "Overall") & (group_df["name"] == "Total")].iloc[0]
    generator = group_df[group_df["category"] == "Generator"].sort_values("asr_pct", ascending=False)
    target = group_df[group_df["category"] == "Target"].sort_values("asr_pct", ascending=False)
    label = group_df[group_df["category"] == "Label"].sort_values("asr_pct", ascending=False)
    matrix = cell_df.pivot(index="generator", columns="target", values="asr_pct")
    lines = [
        "4x4 Analysis Summary",
        "====================",
        f"Total evaluated rows: {int(cell_df['rows'].sum())} across 16 cells",
        (
            f"Overall ASR: {overall.asr_pct:.2f}% "
            f"({int(overall.n_attack_success)}/{int(overall.n_original_correct)}), "
            f"Wilson 95% CI [{overall.wilson_ci_lower_pct:.2f}%, {overall.wilson_ci_upper_pct:.2f}%]"
        ),
        "",
        "ASR matrix (%):",
        matrix.round(2).to_string(),
        "",
        "Generator-side ASR (%):",
        generator[["name", "asr_pct", "n_attack_success", "n_original_correct"]].round(2).to_string(index=False),
        "",
        "Target-side ASR (%):",
        target[["name", "asr_pct", "n_attack_success", "n_original_correct"]].round(2).to_string(index=False),
        "",
        "Label-wise ASR (%):",
        label[["name", "asr_pct", "n_attack_success", "n_original_correct"]].round(2).to_string(index=False),
        "",
        "Largest label transitions among attack-success cases:",
    ]
    top_transitions = transition_df.sort_values("count", ascending=False).head(6)
    for _, row in top_transitions.iterrows():
        lines.append(
            f"- {row.gold_name} -> {row.pred_name}: {int(row['count'])}/"
            f"{int(row.total_from_gold)} ({row.percentage:.1f}%)"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean 4x4 analysis tables.")
    parser.add_argument("--input_dir", default=CROSS_MODEL_EVAL_DIR)
    parser.add_argument("--output_dir", default=SUMMARY_TABLE_DIR)
    parser.add_argument("--expected_rows", type=int, default=2209)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_results(Path(args.input_dir), expected_rows=args.expected_rows)
    cell_df = build_cell_summary(combined)
    group_df = build_group_summary(combined, n_bootstrap=args.n_bootstrap)
    label_cell_df = build_labelwise_by_cell(combined)
    transition_df = build_transition_summary(combined)
    pairwise_df = build_pairwise_summary(combined, n_bootstrap=args.n_bootstrap)

    matrix_df = cell_df.pivot(index="generator", columns="target", values="asr_pct").reset_index()
    original_accuracy_df = (
        combined.groupby("target", as_index=False)
        .agg(rows=("id", "size"), original_accuracy=("original_correct", "mean"))
        .sort_values("target")
    )
    original_accuracy_df["original_accuracy_pct"] = original_accuracy_df["original_accuracy"] * 100

    outputs = {
        "cross_model_asr_cell_summary.csv": cell_df,
        "cross_model_asr_matrix_pct.csv": matrix_df,
        "generator_target_asr_ci.csv": group_df,
        "labelwise_asr_summary.csv": group_df[group_df["category"] == "Label"].copy(),
        "labelwise_asr_by_cell.csv": label_cell_df,
        "label_transition_summary.csv": transition_df,
        "statistical_analysis_pairwise.csv": pairwise_df,
        "original_accuracy_by_target.csv": original_accuracy_df,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    write_text_summary(output_dir / "4x4_analysis_summary.txt", cell_df, group_df, transition_df)

    print(f"Loaded rows: {len(combined):,}")
    for filename in outputs:
        print(f"Saved: {output_dir / filename}")
    print(f"Saved: {output_dir / '4x4_analysis_summary.txt'}")


if __name__ == "__main__":
    main()
