from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CROSS_MODEL_EVAL_DIR,
    GENERATORS,
    LABEL_NAMES,
    SHARED_VALID_DIR,
    TARGETS,
    cross_model_generator_dirname,
    model_file_key,
    shared_valid_filename,
    target_output_filename,
)


UNIT_COLS = ["generator_key", "id", "label", "premise", "hypothesis", "attacked_hypothesis"]
ANNOTATION_COLUMNS = [
    "annotation_id",
    "premise",
    "original_hypothesis",
    "paraphrased_hypothesis",
    "paraphrased_pair_label",
    "semantic_drift_type",
    "fluency_defect",
    "annotator_confidence",
]
DRIFT_TAXONOMY = [
    "no_semantic_drift",
    "specificity_shift",
    "entity_number_time_change",
    "negation_modality_scope_drift",
    "causality_temporal_relation_drift",
    "fluency_grammar_defect",
    "ambiguous_label",
    "other",
]


def load_attack_texts(generator: str) -> pd.DataFrame:
    path = Path(SHARED_VALID_DIR) / shared_valid_filename(generator)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = ["id", "premise", "hypothesis", "label", "attacked_hypothesis", "bert_score"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[needed].copy()


def load_eval(generator: str, target: str) -> pd.DataFrame:
    path = Path(CROSS_MODEL_EVAL_DIR) / cross_model_generator_dirname(generator) / target_output_filename(target)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = ["id", "label", "pred_original", "pred_attacked", "original_correct", "attack_success"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[needed].copy()


def build_attack_success_candidates() -> pd.DataFrame:
    rows = []
    for generator in GENERATORS:
        generator_key = model_file_key(generator)
        texts = load_attack_texts(generator)
        for target in TARGETS:
            target_key = model_file_key(target)
            eval_df = load_eval(generator, target)
            merged = eval_df[eval_df["attack_success"] == 1].merge(
                texts,
                on=["id", "label"],
                how="left",
                suffixes=("", "_text"),
            )
            merged["condition_type"] = "llm"
            merged["generator_key"] = generator_key
            merged["target_key"] = target_key
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    candidates = pd.concat(rows, ignore_index=True)
    candidates["label"] = candidates["label"].astype(int)
    candidates["label_name"] = candidates["label"].map(LABEL_NAMES)
    return candidates


def target_join(values: pd.Series) -> str:
    return ";".join(sorted(str(value) for value in values.dropna().unique()))


def build_unique_units(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = (
        candidates.groupby(UNIT_COLS, as_index=False)
        .agg(
            label_name=("label_name", "first"),
            bert_score=("bert_score", "first"),
            target_keys=("target_key", target_join),
            target_count=("target_key", lambda s: int(s.nunique())),
            attack_success_instance_count=("target_key", "size"),
        )
        .sort_values(["generator_key", "label", "id"])
        .reset_index(drop=True)
    )
    units.insert(0, "unit_id", [f"U{idx + 1:04d}" for idx in range(len(units))])

    mapping = candidates.merge(units[UNIT_COLS + ["unit_id"]], on=UNIT_COLS, how="inner")
    keep_cols = [
        "unit_id",
        "condition_type",
        "generator_key",
        "target_key",
        "id",
        "label",
        "label_name",
        "pred_original",
        "pred_attacked",
        "original_correct",
        "attack_success",
        "bert_score",
        "premise",
        "hypothesis",
        "attacked_hypothesis",
    ]
    mapping = mapping[keep_cols].sort_values(["unit_id", "target_key"]).reset_index(drop=True)
    return units, mapping


def allocate_quotas(units: pd.DataFrame, target_n: int, base_per_generator_label: int) -> pd.DataFrame:
    rows = []
    for (generator_key, label), group in units.groupby(["generator_key", "label"], sort=True):
        available = len(group)
        initial = min(base_per_generator_label, available)
        rows.append(
            {
                "generator_key": generator_key,
                "label": int(label),
                "label_name": LABEL_NAMES[int(label)],
                "available_unique_units": available,
                "base_quota": base_per_generator_label,
                "quota": initial,
            }
        )
    quota_df = pd.DataFrame(rows).sort_values(["generator_key", "label"]).reset_index(drop=True)
    if quota_df["available_unique_units"].sum() < target_n:
        raise ValueError(f"Only {int(quota_df['available_unique_units'].sum())} unique units available for {target_n}")

    remaining = target_n - int(quota_df["quota"].sum())
    while remaining > 0:
        quota_df["capacity"] = quota_df["available_unique_units"] - quota_df["quota"]
        eligible = quota_df[quota_df["capacity"] > 0].sort_values(
            ["capacity", "available_unique_units", "generator_key", "label"],
            ascending=[False, False, True, True],
        )
        if eligible.empty:
            raise ValueError("Quota allocation failed despite sufficient total availability")
        idx = eligible.index[0]
        quota_df.loc[idx, "quota"] += 1
        remaining -= 1
    quota_df["deficit_from_base"] = (quota_df["base_quota"] - quota_df["available_unique_units"]).clip(lower=0)
    quota_df["extra_reallocated"] = (quota_df["quota"] - quota_df["base_quota"]).clip(lower=0)
    return quota_df.drop(columns=["capacity"], errors="ignore")


def target_aware_sample(group: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    if quota <= 0:
        return group.iloc[0:0].copy()
    if len(group) <= quota:
        return group.copy()

    rng_seed = seed
    selected_ids: set[str] = set()
    target_availability = {}
    for target in sorted({target for targets in group["target_keys"] for target in str(targets).split(";")}):
        target_availability[target] = int(group["target_keys"].str.split(";").apply(lambda xs: target in xs).sum())

    for target, _ in sorted(target_availability.items(), key=lambda item: (item[1], item[0])):
        if len(selected_ids) >= quota:
            break
        candidates = group[
            group["target_keys"].str.split(";").apply(lambda xs: target in xs)
            & ~group["unit_id"].isin(selected_ids)
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["target_count", "attack_success_instance_count"], ascending=False)
        picked = candidates.sample(n=1, random_state=rng_seed).iloc[0]
        selected_ids.add(str(picked["unit_id"]))
        rng_seed += 1

    remaining = quota - len(selected_ids)
    if remaining > 0:
        rest = group[~group["unit_id"].isin(selected_ids)]
        sampled_rest = rest.sample(n=remaining, random_state=seed + 999)
        selected_ids.update(str(value) for value in sampled_rest["unit_id"].tolist())

    return group[group["unit_id"].isin(selected_ids)].copy()


def sample_units(units: pd.DataFrame, quota_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled_frames = []
    for idx, row in quota_df.iterrows():
        group = units[
            (units["generator_key"] == row["generator_key"])
            & (units["label"] == int(row["label"]))
        ].copy()
        sampled_frames.append(target_aware_sample(group, int(row["quota"]), seed + idx * 100))
    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = repair_global_triple_duplicates(sampled, units, seed + 7777)
    sampled = sampled.sort_values(["generator_key", "label", "id"]).reset_index(drop=True)
    sampled.insert(0, "annotation_id", [f"FHV{idx + 1:04d}" for idx in range(len(sampled))])
    return sampled


def repair_global_triple_duplicates(sampled: pd.DataFrame, units: pd.DataFrame, seed: int) -> pd.DataFrame:
    triple_cols = ["premise", "hypothesis", "attacked_hypothesis"]
    sampled = sampled.copy().reset_index(drop=True)
    for iteration in range(100):
        duplicate_mask = sampled.duplicated(triple_cols, keep="first")
        if not duplicate_mask.any():
            return sampled
        duplicate_indices = sampled.index[duplicate_mask].tolist()
        for local_offset, dup_idx in enumerate(duplicate_indices):
            row = sampled.loc[dup_idx]
            selected_unit_ids = set(sampled["unit_id"].astype(str).tolist())
            selected_unit_ids.discard(str(row["unit_id"]))
            used_triples = set(
                tuple(values)
                for values in sampled.drop(index=dup_idx)[triple_cols].itertuples(index=False, name=None)
            )
            pool = units[
                (units["generator_key"] == row["generator_key"])
                & (units["label"] == int(row["label"]))
                & ~units["unit_id"].astype(str).isin(selected_unit_ids)
            ].copy()
            pool["_triple"] = list(zip(pool["premise"], pool["hypothesis"], pool["attacked_hypothesis"]))
            pool = pool[~pool["_triple"].isin(used_triples)].drop(columns=["_triple"])
            if pool.empty:
                raise ValueError(
                    "Could not repair duplicate paraphrase triple within "
                    f"{row['generator_key']} label={int(row['label'])}"
                )
            replacement = pool.sample(n=1, random_state=seed + iteration * 1000 + local_offset).iloc[0]
            sampled.loc[dup_idx, replacement.index] = replacement
    raise ValueError("Could not remove all global paraphrase triple duplicates after 100 iterations")


def make_annotation_sheet(sampled: pd.DataFrame, seed: int) -> pd.DataFrame:
    sheet = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
    return sheet[ANNOTATION_COLUMNS]


def write_guideline(path: Path) -> None:
    taxonomy = "\n".join(f"- {item}" for item in DRIFT_TAXONOMY)
    text = f"""# Human Validation Annotation Guideline

Each row is one unique paraphrase unit sampled from 4x4 attack-success cases.
Duplicate target-model instances are removed before annotation. A separate mapping file
links each annotation unit to all generator-target attack-success instances represented by it.

## Task

Judge only the relation between `premise` and `paraphrased_hypothesis`.
Do not re-label the original premise-hypothesis pair. The original gold label is stored
only in the answer key and will be used during aggregation.

The `original_hypothesis` column is provided only to help identify semantic drift.

## Label Values

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = Unclear or not reliably judgeable

## Columns to Fill

- `paraphrased_pair_label`: NLI label for premise + paraphrased hypothesis.
- `semantic_drift_type`: choose one taxonomy label below.
- `fluency_defect`: Y if the paraphrase has a grammar, formatting, or readability defect; otherwise N.
- `annotator_confidence`: high, medium, or low.

## Semantic Drift Taxonomy

{taxonomy}

Use `no_semantic_drift` when the paraphrase preserves the original hypothesis meaning.
Use a drift label when the paraphrase changes meaning, narrows or broadens the claim,
changes entities/numbers/time, changes negation/modality/scope, changes causality or
temporal relation, is grammatically defective, or is ambiguous.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 600 unique human-validation units.")
    parser.add_argument("--target_n", type=int, default=600)
    parser.add_argument("--base_per_generator_label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--output_dir", default="results/05_human_evaluation/label_invariance_600")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_attack_success_candidates()
    units, mapping = build_unique_units(candidates)
    quota_df = allocate_quotas(units, args.target_n, args.base_per_generator_label)
    sampled = sample_units(units, quota_df, args.seed)
    sampled_mapping = mapping[mapping["unit_id"].isin(sampled["unit_id"])].merge(
        sampled[["unit_id", "annotation_id"]],
        on="unit_id",
        how="left",
    )

    answer_key = sampled[
        [
            "annotation_id",
            "unit_id",
            "generator_key",
            "id",
            "label",
            "label_name",
            "bert_score",
            "target_keys",
            "target_count",
            "attack_success_instance_count",
            "premise",
            "hypothesis",
            "attacked_hypothesis",
        ]
    ].copy()

    annotation_sheets = {
        "researcher1": make_annotation_sheet(sampled, args.seed + 1),
        "researcher2": make_annotation_sheet(sampled, args.seed + 2),
        "researcher3": make_annotation_sheet(sampled, args.seed + 3),
    }

    cell_coverage = (
        sampled_mapping.groupby(["generator_key", "target_key", "label", "label_name"], as_index=False)
        .agg(
            sampled_unique_units=("unit_id", "nunique"),
            represented_attack_success_instances=("unit_id", "size"),
        )
        .sort_values(["generator_key", "target_key", "label"])
    )

    candidates.to_csv(output_dir / "human_validation_attack_success_candidates.csv", index=False, encoding="utf-8-sig")
    units.to_csv(output_dir / "human_validation_unique_units.csv", index=False, encoding="utf-8-sig")
    quota_df.to_csv(output_dir / "human_validation_sampling_quota.csv", index=False, encoding="utf-8-sig")
    answer_key.to_csv(output_dir / "human_validation_answer_key.csv", index=False, encoding="utf-8-sig")
    sampled_mapping.to_csv(output_dir / "human_validation_instance_mapping.csv", index=False, encoding="utf-8-sig")
    cell_coverage.to_csv(output_dir / "human_validation_cell_coverage.csv", index=False, encoding="utf-8-sig")
    for name, sheet in annotation_sheets.items():
        sheet.to_csv(output_dir / f"human_validation_annotation_sheet_{name}.csv", index=False, encoding="utf-8-sig")
    write_guideline(output_dir / "human_validation_annotation_guideline.md")

    summary_lines = [
        "Human-validation sampling summary",
        "=================================",
        f"Attack-success instances: {len(candidates)}",
        f"Unique paraphrase units: {len(units)}",
        f"Sampled annotation units: {len(answer_key)}",
        f"Mapped attack-success instances represented by sampled units: {len(sampled_mapping)}",
        f"Target sample size: {args.target_n}",
        "",
        "Quota by generator and label:",
        quota_df.to_string(index=False),
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
