from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_sheet(answer_key: pd.DataFrame, seed: int) -> pd.DataFrame:
    sheet = answer_key.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet = sheet.rename(
        columns={
            "hypothesis": "original_hypothesis",
            "attacked_hypothesis": "backtranslated_hypothesis",
        }
    )
    sheet = sheet[["annotation_id", "premise", "original_hypothesis", "backtranslated_hypothesis"]].copy()
    sheet["paraphrased_pair_label"] = ""
    sheet["semantic_drift_type"] = ""
    sheet["fluency_defect"] = ""
    sheet["annotator_confidence"] = ""
    return sheet[
        [
            "annotation_id",
            "premise",
            "original_hypothesis",
            "backtranslated_hypothesis",
            "paraphrased_pair_label",
            "semantic_drift_type",
            "fluency_defect",
            "annotator_confidence",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create exact-balanced BT human validation package with one duplicate unit.")
    parser.add_argument("--input_dir", default="results/05_human_evaluation/bt_label_invariance_180")
    parser.add_argument("--output_dir", default="results/05_human_evaluation/bt_label_invariance_180_exact_balanced")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_key = pd.read_csv(input_dir / "bt_human_validation_answer_key.csv", encoding="utf-8-sig")
    mapping = pd.read_csv(input_dir / "bt_human_validation_instance_mapping.csv", encoding="utf-8-sig")

    remove_pool = answer_key[(answer_key["primary_target_key"] == "claude_sonnet") & (answer_key["label"] == 0)].copy()
    if remove_pool.empty:
        raise ValueError("No removable claude_sonnet entailment row found")
    remove_pool = remove_pool.sort_values(["target_count", "attack_success_instance_count", "id"])
    remove_index = remove_pool.index[0]
    removed = answer_key.loc[[remove_index]].copy()

    duplicate_pool = answer_key[(answer_key["primary_target_key"] == "gemini") & (answer_key["label"] == 2)].copy()
    if duplicate_pool.empty:
        raise ValueError("No duplicable gemini contradiction row found")
    duplicate_pool = duplicate_pool.sort_values(["target_count", "attack_success_instance_count", "id"], ascending=[False, False, True])
    duplicate = duplicate_pool.iloc[[0]].copy()
    duplicate["source_unit_id"] = duplicate["unit_id"]
    duplicate["unit_id"] = duplicate["unit_id"].astype(str) + "_DUP1"
    duplicate["duplicate_for_exact_balance"] = True

    answer_key["source_unit_id"] = answer_key.get("source_unit_id", answer_key["unit_id"])
    answer_key["duplicate_for_exact_balance"] = False
    exact = pd.concat([answer_key.drop(index=remove_index), duplicate], ignore_index=True)
    exact = exact.sort_values(["primary_target_key", "label", "id", "unit_id"]).reset_index(drop=True)
    exact["annotation_id"] = [f"BTEX{idx + 1:04d}" for idx in range(len(exact))]

    exact_mapping_rows = []
    source_to_annotation = exact[["source_unit_id", "annotation_id", "primary_target_key", "duplicate_for_exact_balance"]].copy()
    for _, row in source_to_annotation.iterrows():
        src = row["source_unit_id"]
        rows = mapping[mapping["unit_id"] == src].copy()
        rows["annotation_id"] = row["annotation_id"]
        rows["source_unit_id"] = src
        rows["primary_target_key"] = row["primary_target_key"]
        rows["duplicate_for_exact_balance"] = row["duplicate_for_exact_balance"]
        exact_mapping_rows.append(rows)
    exact_mapping = pd.concat(exact_mapping_rows, ignore_index=True)

    quota = (
        exact.groupby(["primary_target_key", "label", "label_name"], as_index=False)
        .agg(sampled=("annotation_id", "size"), unique_source_units=("source_unit_id", "nunique"))
        .sort_values(["primary_target_key", "label"])
    )
    coverage = (
        exact_mapping.groupby(["target_key", "label", "label_name"], as_index=False)
        .agg(
            sampled_annotation_rows=("annotation_id", "nunique"),
            unique_source_units=("source_unit_id", "nunique"),
            represented_attack_success_instances=("annotation_id", "size"),
        )
        .sort_values(["target_key", "label"])
    )

    exact.to_csv(output_dir / "bt_human_validation_answer_key.csv", index=False, encoding="utf-8-sig")
    exact_mapping.to_csv(output_dir / "bt_human_validation_instance_mapping.csv", index=False, encoding="utf-8-sig")
    quota.to_csv(output_dir / "bt_human_validation_sampling_quota.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "bt_human_validation_cell_coverage.csv", index=False, encoding="utf-8-sig")
    for idx in [1, 2, 3]:
        make_sheet(exact, args.seed + idx).to_csv(
            output_dir / f"bt_human_validation_annotation_sheet_researcher{idx}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    for filename in ["bt_human_validation_annotation_guideline.md", "bt_human_validation_annotation_guideline_ko.md"]:
        src = input_dir / filename
        if src.exists():
            text = src.read_text(encoding="utf-8")
            text += (
                "\n\nNote: This exact-balanced package contains 180 annotation rows "
                "with 179 unique source BT units. One duplicate source unit is included "
                "to preserve the 15-per-target-label sampling layout.\n"
            )
            (output_dir / filename).write_text(text, encoding="utf-8")

    summary_lines = [
        "BT exact-balanced human-validation sampling summary",
        "===================================================",
        "Annotation rows: 180",
        f"Unique source BT units: {exact['source_unit_id'].nunique()}",
        f"Duplicate rows for exact balance: {int(exact['duplicate_for_exact_balance'].sum())}",
        "",
        "Removed row for balance:",
        removed[["annotation_id", "unit_id", "primary_target_key", "label", "id"]].to_string(index=False),
        "",
        "Duplicated source row for balance:",
        duplicate[["unit_id", "source_unit_id", "primary_target_key", "label", "id"]].to_string(index=False),
        "",
        "Quota by primary target and label:",
        quota.to_string(index=False),
        "",
        "Output directory:",
        str(output_dir),
    ]
    (output_dir / "bt_human_validation_sampling_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
