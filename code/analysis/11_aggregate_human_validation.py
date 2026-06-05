from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_VALUES = ["0", "1", "2", "U"]
DRIFT_VALUES = [
    "no_semantic_drift",
    "specificity_shift",
    "entity_number_time_change",
    "negation_modality_scope_drift",
    "causality_temporal_relation_drift",
    "fluency_grammar_defect",
    "ambiguous_label",
    "other",
]
FLUENCY_VALUES = ["N", "Y"]
CONFIDENCE_VALUES = ["low", "medium", "high"]
LABEL_NAMES = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def majority(values: list[object]) -> str:
    cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
    if not cleaned:
        return "TIE"
    counts = Counter(cleaned)
    best_count = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best_count)
    if len(winners) == 1 and best_count >= 2:
        return winners[0]
    return "TIE"


def fleiss_kappa(df: pd.DataFrame, columns: list[str], categories: list[str]) -> tuple[float, float]:
    matrix = []
    for _, row in df.iterrows():
        values = [str(row[column]).strip() for column in columns]
        matrix.append([values.count(category) for category in categories])
    counts = np.asarray(matrix, dtype=float)
    if counts.size == 0:
        return (0.0, 0.0)
    n_raters = len(columns)
    p_i = ((counts * counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (len(counts) * n_raters)
    p_e = float((p_j * p_j).sum())
    if abs(1.0 - p_e) < 1e-12:
        return (1.0, p_bar)
    return (float((p_bar - p_e) / (1.0 - p_e)), p_bar)


def read_annotation_package(
    package_dir: Path,
    prefix: str,
    condition_type: str,
    text_column: str,
    answer_key_name: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    answer_key = pd.read_csv(package_dir / answer_key_name, encoding="utf-8-sig")
    if condition_type == "llm":
        group_column = "generator_key"
    else:
        group_column = "primary_target_key"
    answer_key = answer_key.copy()
    answer_key["condition_type"] = condition_type
    answer_key["condition_key"] = answer_key[group_column]
    answer_key["gold_label"] = answer_key["label"].astype(int).astype(str)

    base_text_columns = ["annotation_id", "premise"]
    if condition_type == "llm":
        base_text_columns += ["hypothesis", "attacked_hypothesis"]
    else:
        base_text_columns += ["hypothesis", "attacked_hypothesis"]
    base = answer_key[
        [
            "annotation_id",
            "condition_type",
            "condition_key",
            "label",
            "label_name",
            "gold_label",
            "source_unit_id",
            "duplicate_for_exact_balance",
            *base_text_columns[1:],
        ]
    ].copy()

    validation_rows: list[dict[str, object]] = []
    expected_ids = set(answer_key["annotation_id"].astype(str))
    rater_frames = []
    for idx in [1, 2, 3]:
        path = package_dir / f"{prefix}_annotation_sheet_researcher{idx}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rater = pd.read_csv(path, encoding="utf-8-sig")
        name = f"r{idx}"
        validation_rows.extend(validate_rater(name, rater, expected_ids, condition_type))
        fields = ["paraphrased_pair_label", "semantic_drift_type", "fluency_defect", "annotator_confidence"]
        add = rater[["annotation_id", *fields]].copy()
        add = add.rename(columns={field: f"{name}_{field}" for field in fields})
        rater_frames.append(add)

    merged = base
    for rater in rater_frames:
        merged = merged.merge(rater, on="annotation_id", how="left", validate="one_to_one")

    for column in merged.columns:
        if column.startswith(("r1_", "r2_", "r3_")):
            merged[column] = merged[column].astype("string").str.strip()

    for field in ["paraphrased_pair_label", "semantic_drift_type", "fluency_defect", "annotator_confidence"]:
        cols = [f"r{idx}_{field}" for idx in [1, 2, 3]]
        merged[f"majority_{field}"] = merged[cols].apply(lambda row: majority(list(row)), axis=1)
        merged[f"all_same_{field}"] = (
            merged[cols[0]].astype(str).eq(merged[cols[1]].astype(str))
            & merged[cols[0]].astype(str).eq(merged[cols[2]].astype(str))
        )

    merged["has_valid_label_majority"] = merged["majority_paraphrased_pair_label"].isin(["0", "1", "2"])
    merged["has_resolved_label_majority"] = merged["majority_paraphrased_pair_label"].isin(["0", "1", "2"])
    merged["label_preserved"] = (
        merged["majority_paraphrased_pair_label"].astype(str).eq(merged["gold_label"].astype(str))
    )
    merged["label_preservation_status"] = np.where(
        merged["majority_paraphrased_pair_label"].isin(["TIE", "U"]),
        "unresolved",
        np.where(merged["label_preserved"], "preserved", "not_preserved"),
    )
    merged["semantic_preserved"] = merged["majority_semantic_drift_type"].eq("no_semantic_drift")
    merged["text_column"] = text_column
    return merged, validation_rows


def validate_rater(name: str, df: pd.DataFrame, expected_ids: set[str], condition_type: str) -> list[dict[str, object]]:
    required = {
        "annotation_id",
        "premise",
        "original_hypothesis",
        "paraphrased_pair_label",
        "semantic_drift_type",
        "fluency_defect",
        "annotator_confidence",
    }
    if condition_type == "llm":
        required.add("paraphrased_hypothesis")
    else:
        required.add("backtranslated_hypothesis")

    rows: list[dict[str, object]] = []
    missing_cols = sorted(required.difference(df.columns))
    rows.append({"condition_type": condition_type, "rater": name, "check": "missing_columns", "value": ",".join(missing_cols)})
    id_set = set(df["annotation_id"].astype(str)) if "annotation_id" in df.columns else set()
    rows.extend(
        [
            {"condition_type": condition_type, "rater": name, "check": "n_rows", "value": len(df)},
            {"condition_type": condition_type, "rater": name, "check": "n_unique_ids", "value": df["annotation_id"].nunique() if "annotation_id" in df.columns else 0},
            {"condition_type": condition_type, "rater": name, "check": "duplicate_ids", "value": int(df["annotation_id"].duplicated().sum()) if "annotation_id" in df.columns else len(df)},
            {"condition_type": condition_type, "rater": name, "check": "missing_expected_ids", "value": len(expected_ids.difference(id_set))},
            {"condition_type": condition_type, "rater": name, "check": "extra_ids", "value": len(id_set.difference(expected_ids))},
        ]
    )
    allowed = {
        "paraphrased_pair_label": set(LABEL_VALUES),
        "semantic_drift_type": set(DRIFT_VALUES),
        "fluency_defect": set(FLUENCY_VALUES),
        "annotator_confidence": set(CONFIDENCE_VALUES),
    }
    for column, allowed_values in allowed.items():
        if column not in df.columns:
            rows.append({"condition_type": condition_type, "rater": name, "check": f"{column}_missing", "value": len(df)})
            rows.append({"condition_type": condition_type, "rater": name, "check": f"{column}_invalid", "value": len(df)})
            continue
        values = df[column].astype("string").str.strip()
        missing = int((values.isna() | values.eq("")).sum())
        invalid = int((~values.isna() & ~values.eq("") & ~values.isin(allowed_values)).sum())
        rows.append({"condition_type": condition_type, "rater": name, "check": f"{column}_missing", "value": missing})
        rows.append({"condition_type": condition_type, "rater": name, "check": f"{column}_invalid", "value": invalid})
    return rows


def agreement_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("paraphrased_pair_label", LABEL_VALUES),
        ("semantic_drift_type", DRIFT_VALUES),
        ("fluency_defect", FLUENCY_VALUES),
        ("annotator_confidence", CONFIDENCE_VALUES),
    ]
    for condition_type, group in results.groupby("condition_type", sort=True):
        for field, categories in specs:
            cols = [f"r{idx}_{field}" for idx in [1, 2, 3]]
            kappa, mean_agreement = fleiss_kappa(group, cols, categories)
            pair_rates = []
            for left, right in [(1, 2), (1, 3), (2, 3)]:
                pair_rates.append(float(group[f"r{left}_{field}"].astype(str).eq(group[f"r{right}_{field}"].astype(str)).mean()))
            rows.append(
                {
                    "condition_type": condition_type,
                    "field": field,
                    "n": len(group),
                    "all_three_same": int(group[f"all_same_{field}"].sum()),
                    "all_three_same_pct": float(group[f"all_same_{field}"].mean() * 100),
                    "pairwise_agreement_mean_pct": float(np.mean(pair_rates) * 100),
                    "pairwise_agreement_min_pct": float(np.min(pair_rates) * 100),
                    "pairwise_agreement_max_pct": float(np.max(pair_rates) * 100),
                    "fleiss_kappa": kappa,
                    "mean_item_agreement": mean_agreement,
                    "majority_tie_or_u": int(group[f"majority_{field}"].isin(["TIE", "U"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def lpr_summary(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_total = len(group)
        n_preserved = int(group["label_preserved"].sum())
        n_not_preserved = int((group["label_preservation_status"] == "not_preserved").sum())
        n_unresolved = int((group["label_preservation_status"] == "unresolved").sum())
        resolved = group[group["label_preservation_status"].isin(["preserved", "not_preserved"])]
        n_resolved = len(resolved)
        n_preserved_resolved = int(resolved["label_preserved"].sum())
        lo_all, hi_all = wilson_ci(n_preserved, n_total)
        lo_resolved, hi_resolved = wilson_ci(n_preserved_resolved, n_resolved)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "n_total": n_total,
                "n_resolved": n_resolved,
                "n_preserved": n_preserved,
                "n_not_preserved": n_not_preserved,
                "n_unresolved": n_unresolved,
                "lpr_all_pct": n_preserved / n_total * 100 if n_total else 0.0,
                "lpr_all_ci_lower_pct": lo_all * 100,
                "lpr_all_ci_upper_pct": hi_all * 100,
                "lpr_resolved_pct": n_preserved_resolved / n_resolved * 100 if n_resolved else 0.0,
                "lpr_resolved_ci_lower_pct": lo_resolved * 100,
                "lpr_resolved_ci_upper_pct": hi_resolved * 100,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_summary(results: pd.DataFrame, column: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total = len(group)
        counts = group[column].value_counts(dropna=False).to_dict()
        for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            row = {col: key for col, key in zip(group_cols, keys)}
            row.update({"category": value, "count": int(count), "percent_pct": count / total * 100 if total else 0.0})
            rows.append(row)
    return pd.DataFrame(rows)


def confusion_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_type, group in results.groupby("condition_type", sort=True):
        for (label, label_name, majority_label), subset in group.groupby(
            ["label", "label_name", "majority_paraphrased_pair_label"], sort=True, dropna=False
        ):
            rows.append(
                {
                    "condition_type": condition_type,
                    "gold_label": int(label),
                    "gold_label_name": label_name,
                    "majority_label": majority_label,
                    "count": len(subset),
                }
            )
    return pd.DataFrame(rows)


def write_summary_text(
    path: Path,
    lpr_overall: pd.DataFrame,
    agreement: pd.DataFrame,
    drift_summary: pd.DataFrame,
) -> None:
    lines = [
        "Human Validation Summary",
        "========================",
        "",
        "Label preservation:",
    ]
    for _, row in lpr_overall.iterrows():
        lines.append(
            f"- {row.condition_type}: {int(row.n_preserved)}/{int(row.n_total)} "
            f"({row.lpr_all_pct:.2f}%), resolved {int(row.n_preserved)}/{int(row.n_resolved)} "
            f"({row.lpr_resolved_pct:.2f}%)"
        )
    lines.extend(["", "Inter-annotator agreement:"])
    label_agreement = agreement[agreement["field"] == "paraphrased_pair_label"]
    drift_agreement = agreement[agreement["field"] == "semantic_drift_type"]
    for _, row in label_agreement.iterrows():
        lines.append(
            f"- {row.condition_type} label: Fleiss' kappa={row.fleiss_kappa:.4f}, "
            f"three-way agreement={row.all_three_same_pct:.2f}%"
        )
    for _, row in drift_agreement.iterrows():
        lines.append(
            f"- {row.condition_type} drift: Fleiss' kappa={row.fleiss_kappa:.4f}, "
            f"three-way agreement={row.all_three_same_pct:.2f}%"
        )
    lines.extend(["", "Majority semantic drift distribution:"])
    for condition_type, group in drift_summary.groupby("condition_type", sort=True):
        lines.append(f"- {condition_type}:")
        for _, row in group.sort_values("count", ascending=False).iterrows():
            lines.append(f"  {row.category}: {int(row['count'])} ({row.percent_pct:.2f}%)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LLM and BT human validation annotations.")
    parser.add_argument("--llm_dir", default="results/05_human_evaluation/label_invariance_600_exact_balanced")
    parser.add_argument("--bt_dir", default="results/05_human_evaluation/bt_label_invariance_180_exact_balanced")
    parser.add_argument("--output_dir", default="results/05_human_evaluation/analysis")
    parser.add_argument("--summary_dir", default="results/03_summary_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    llm_results, llm_validation = read_annotation_package(
        Path(args.llm_dir),
        prefix="human_validation",
        condition_type="llm",
        text_column="paraphrased_hypothesis",
        answer_key_name="human_validation_answer_key.csv",
    )
    bt_results, bt_validation = read_annotation_package(
        Path(args.bt_dir),
        prefix="bt_human_validation",
        condition_type="bt",
        text_column="backtranslated_hypothesis",
        answer_key_name="bt_human_validation_answer_key.csv",
    )
    results = pd.concat([llm_results, bt_results], ignore_index=True)
    validation = pd.DataFrame(llm_validation + bt_validation)

    agreement = agreement_summary(results)
    lpr_overall = lpr_summary(results, ["condition_type"])
    lpr_by_condition = lpr_summary(results, ["condition_type", "condition_key"])
    lpr_by_label = lpr_summary(results, ["condition_type", "label", "label_name"])
    lpr_by_condition_label = lpr_summary(results, ["condition_type", "condition_key", "label", "label_name"])
    drift = categorical_summary(results, "majority_semantic_drift_type", ["condition_type"])
    fluency = categorical_summary(results, "majority_fluency_defect", ["condition_type"])
    confidence = categorical_summary(results, "majority_annotator_confidence", ["condition_type"])
    confusion = confusion_summary(results)

    consistency_rows = []
    for condition_type, group in results.groupby("condition_type", sort=True):
        no_drift_not_preserved = int(
            (group["majority_semantic_drift_type"].eq("no_semantic_drift") & ~group["label_preserved"]).sum()
        )
        drift_preserved = int(
            (~group["majority_semantic_drift_type"].eq("no_semantic_drift") & group["label_preserved"]).sum()
        )
        consistency_rows.append(
            {
                "condition_type": condition_type,
                "no_semantic_drift_but_not_label_preserved": no_drift_not_preserved,
                "semantic_drift_but_label_preserved": drift_preserved,
            }
        )
    consistency = pd.DataFrame(consistency_rows)

    outputs = {
        "human_validation_majority_results.csv": results,
        "human_validation_rater_checks.csv": validation,
        "human_validation_agreement_summary.csv": agreement,
        "human_validation_lpr_overall.csv": lpr_overall,
        "human_validation_lpr_by_condition.csv": lpr_by_condition,
        "human_validation_lpr_by_label.csv": lpr_by_label,
        "human_validation_lpr_by_condition_label.csv": lpr_by_condition_label,
        "human_validation_drift_summary.csv": drift,
        "human_validation_fluency_summary.csv": fluency,
        "human_validation_confidence_summary.csv": confidence,
        "human_validation_label_confusion.csv": confusion,
        "human_validation_consistency_checks.csv": consistency,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        if filename != "human_validation_majority_results.csv":
            df.to_csv(summary_dir / filename, index=False, encoding="utf-8-sig")

    write_summary_text(summary_dir / "human_validation_summary.txt", lpr_overall, agreement, drift)
    write_summary_text(output_dir / "human_validation_summary.txt", lpr_overall, agreement, drift)

    print("Human validation aggregation complete.")
    print(lpr_overall.to_string(index=False))
    print(agreement[agreement["field"].isin(["paraphrased_pair_label", "semantic_drift_type"])].to_string(index=False))


if __name__ == "__main__":
    main()
