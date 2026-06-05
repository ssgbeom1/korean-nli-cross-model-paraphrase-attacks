from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_sheet(answer_key: pd.DataFrame, seed: int) -> pd.DataFrame:
    sheet = answer_key.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet = sheet.rename(
        columns={
            "hypothesis": "original_hypothesis",
            "attacked_hypothesis": "paraphrased_hypothesis",
        }
    )
    sheet = sheet[["annotation_id", "premise", "original_hypothesis", "paraphrased_hypothesis"]].copy()
    sheet["paraphrased_pair_label"] = ""
    sheet["semantic_drift_type"] = ""
    sheet["fluency_defect"] = ""
    sheet["annotator_confidence"] = ""
    return sheet[
        [
            "annotation_id",
            "premise",
            "original_hypothesis",
            "paraphrased_hypothesis",
            "paraphrased_pair_label",
            "semantic_drift_type",
            "fluency_defect",
            "annotator_confidence",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create exact 50-per-generator-label LLM human validation package."
    )
    parser.add_argument("--input_dir", default="results/05_human_evaluation/label_invariance_600")
    parser.add_argument(
        "--output_dir",
        default="results/05_human_evaluation/label_invariance_600_exact_balanced",
    )
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_key = pd.read_csv(input_dir / "human_validation_answer_key.csv", encoding="utf-8-sig")
    mapping = pd.read_csv(input_dir / "human_validation_instance_mapping.csv", encoding="utf-8-sig")

    answer_key["source_unit_id"] = answer_key["unit_id"]
    answer_key["duplicate_for_exact_balance"] = False

    remove_pool = answer_key[(answer_key["generator_key"] == "hyperclova_x") & (answer_key["label"] == 1)].copy()
    if len(remove_pool) < 3:
        raise ValueError("Need at least three removable HyperCLOVA X neutral rows.")
    remove_pool = remove_pool.sort_values(
        ["target_count", "attack_success_instance_count", "bert_score", "id"],
        ascending=[True, True, True, True],
    )
    removed = remove_pool.head(3).copy()

    duplicate_pool = answer_key[(answer_key["generator_key"] == "gpt") & (answer_key["label"] == 2)].copy()
    if len(duplicate_pool) < 3:
        raise ValueError("Need at least three duplicable GPT contradiction rows.")
    duplicate_pool = duplicate_pool.sort_values(
        ["target_count", "attack_success_instance_count", "bert_score", "id"],
        ascending=[False, False, False, True],
    )
    duplicates = duplicate_pool.head(3).copy()
    duplicates["source_unit_id"] = duplicates["unit_id"]
    duplicates["unit_id"] = duplicates["unit_id"].astype(str) + "_DUP1"
    duplicates["duplicate_for_exact_balance"] = True

    exact = pd.concat(
        [
            answer_key.drop(index=removed.index),
            duplicates,
        ],
        ignore_index=True,
    )
    exact = exact.sort_values(["generator_key", "label", "id", "unit_id"]).reset_index(drop=True)
    exact["annotation_id"] = [f"LLMEX{idx + 1:04d}" for idx in range(len(exact))]

    exact_mapping_rows = []
    for _, row in exact[["unit_id", "source_unit_id", "annotation_id", "duplicate_for_exact_balance"]].iterrows():
        source_unit_id = row["source_unit_id"]
        rows = mapping[mapping["unit_id"] == source_unit_id].copy()
        if rows.empty:
            raise ValueError(f"No mapping rows found for source unit {source_unit_id}")
        rows["source_unit_id"] = source_unit_id
        rows["unit_id"] = row["unit_id"]
        rows["annotation_id"] = row["annotation_id"]
        rows["duplicate_for_exact_balance"] = row["duplicate_for_exact_balance"]
        exact_mapping_rows.append(rows)
    exact_mapping = pd.concat(exact_mapping_rows, ignore_index=True)

    quota = (
        exact.groupby(["generator_key", "label", "label_name"], as_index=False)
        .agg(sampled=("annotation_id", "size"), unique_source_units=("source_unit_id", "nunique"))
        .sort_values(["generator_key", "label"])
    )
    coverage = (
        exact_mapping.groupby(["generator_key", "target_key", "label", "label_name"], as_index=False)
        .agg(
            sampled_annotation_rows=("annotation_id", "nunique"),
            unique_source_units=("source_unit_id", "nunique"),
            represented_attack_success_instances=("annotation_id", "size"),
        )
        .sort_values(["generator_key", "target_key", "label"])
    )

    exact.to_csv(output_dir / "human_validation_answer_key.csv", index=False, encoding="utf-8-sig")
    exact_mapping.to_csv(
        output_dir / "human_validation_instance_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quota.to_csv(output_dir / "human_validation_sampling_quota.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "human_validation_cell_coverage.csv", index=False, encoding="utf-8-sig")

    candidates = input_dir / "human_validation_attack_success_candidates.csv"
    units = input_dir / "human_validation_unique_units.csv"
    if candidates.exists():
        pd.read_csv(candidates, encoding="utf-8-sig").to_csv(
            output_dir / candidates.name,
            index=False,
            encoding="utf-8-sig",
        )
    if units.exists():
        pd.read_csv(units, encoding="utf-8-sig").to_csv(
            output_dir / units.name,
            index=False,
            encoding="utf-8-sig",
        )

    for idx in [1, 2, 3]:
        make_sheet(exact, args.seed + idx).to_csv(
            output_dir / f"human_validation_annotation_sheet_researcher{idx}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    for filename in [
        "human_validation_annotation_guideline.md",
        "human_validation_annotation_guideline_ko.md",
    ]:
        src = input_dir / filename
        if src.exists():
            text = src.read_text(encoding="utf-8")
            text += (
                "\n\nSampling note: This exact-balanced package contains 600 annotation rows "
                "with 597 unique source paraphrase units. Three GPT-Contradiction source units "
                "are repeated to preserve the 50-per-generator-label annotation layout.\n"
            )
            (output_dir / filename).write_text(text, encoding="utf-8")

    summary_lines = [
        "LLM exact-balanced human-validation sampling summary",
        "=====================================================",
        f"Annotation rows: {len(exact)}",
        f"Unique source paraphrase units: {exact['source_unit_id'].nunique()}",
        f"Duplicate rows for exact balance: {int(exact['duplicate_for_exact_balance'].sum())}",
        "",
        "Removed rows for balance:",
        removed[["annotation_id", "unit_id", "generator_key", "label", "id"]].to_string(index=False),
        "",
        "Duplicated source rows for balance:",
        duplicates[["unit_id", "source_unit_id", "generator_key", "label", "id"]].to_string(index=False),
        "",
        "Quota by generator and label:",
        quota.to_string(index=False),
        "",
        "Output directory:",
        str(output_dir),
    ]
    (output_dir / "human_validation_sampling_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
