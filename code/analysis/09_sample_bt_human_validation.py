from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    BACKTRANSLATION_EVAL_DIR,
    LABEL_NAMES,
    SHARED_VALID_DIR,
    TARGETS,
    backtranslation_shared_valid_filename,
    model_file_key,
    target_output_filename,
)


TEXT_COLS = ["id", "premise", "hypothesis", "attacked_hypothesis", "bert_score"]
UNIT_COLS = ["id", "label", "premise", "hypothesis", "attacked_hypothesis"]
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


def load_bt_texts() -> pd.DataFrame:
    path = Path(SHARED_VALID_DIR) / backtranslation_shared_valid_filename()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [col for col in TEXT_COLS + ["label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[TEXT_COLS + ["label"]].copy()


def load_bt_eval(target: str) -> pd.DataFrame:
    target_key = model_file_key(target)
    path = Path(BACKTRANSLATION_EVAL_DIR) / target_output_filename(target)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["id", "label", "pred_original", "pred_attacked", "original_correct", "attack_success"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[required].copy()
    df["target_key"] = target_key
    return df


def join_unique(values: pd.Series) -> str:
    return ";".join(sorted(str(value) for value in values.dropna().unique()))


def build_candidates() -> pd.DataFrame:
    texts = load_bt_texts()
    rows = []
    for target in TARGETS:
        eval_df = load_bt_eval(target)
        merged = eval_df[eval_df["attack_success"] == 1].merge(
            texts[TEXT_COLS],
            on="id",
            how="left",
        )
        merged["condition_type"] = "backtranslation"
        merged["generator_key"] = "backtranslation"
        merged["label"] = merged["label"].astype(int)
        merged["label_name"] = merged["label"].map(LABEL_NAMES)
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_units_and_mapping(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = (
        candidates.groupby(UNIT_COLS, as_index=False)
        .agg(
            label_name=("label_name", "first"),
            bert_score=("bert_score", "first"),
            target_keys=("target_key", join_unique),
            target_count=("target_key", lambda s: int(s.nunique())),
            attack_success_instance_count=("target_key", "size"),
        )
        .sort_values(["label", "id"])
        .reset_index(drop=True)
    )
    units.insert(0, "unit_id", [f"BTU{idx + 1:04d}" for idx in range(len(units))])

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


def sample_target_label_balanced(
    units: pd.DataFrame,
    mapping: pd.DataFrame,
    per_target_label: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [(model_file_key(target), label) for target in TARGETS for label in [0, 1, 2]]
    unit_order = units.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    assignments = solve_quota_flow(unit_order, mapping, cells, per_target_label)
    assignments = add_flexible_extras(assignments, units, mapping, cells, len(cells) * per_target_label, seed + 999)

    assigned_rows = []
    for unit_id, primary_target_key in assignments:
        row = units[units["unit_id"] == unit_id].iloc[0].copy()
        row["primary_target_key"] = primary_target_key
        assigned_rows.append(row)
    sampled = pd.DataFrame(assigned_rows)
    sampled = sampled.sort_values(["primary_target_key", "label", "id"]).reset_index(drop=True)
    sampled.insert(0, "annotation_id", [f"BTHV{idx + 1:04d}" for idx in range(len(sampled))])

    quota_rows = []
    for target_key, label in cells:
        candidate_ids = mapping[(mapping["target_key"] == target_key) & (mapping["label"] == label)]["unit_id"].unique()
        sampled_count = int(((sampled["primary_target_key"] == target_key) & (sampled["label"] == label)).sum())
        quota_rows.append(
            {
                "primary_target_key": target_key,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "available_unique_units": len(candidate_ids),
                "requested": per_target_label,
                "sampled": sampled_count,
            }
        )
    quota = pd.DataFrame(quota_rows).sort_values(["primary_target_key", "label"]).reset_index(drop=True)
    return sampled, quota


class Dinic:
    def __init__(self, n: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, fr: int, to: int, cap: int) -> int:
        forward = [to, cap, len(self.graph[to])]
        backward = [fr, 0, len(self.graph[fr])]
        self.graph[fr].append(forward)
        self.graph[to].append(backward)
        return len(self.graph[fr]) - 1

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        n = len(self.graph)
        while True:
            level = [-1] * n
            queue = [source]
            level[source] = 0
            for node in queue:
                for to, cap, _ in self.graph[node]:
                    if cap > 0 and level[to] < 0:
                        level[to] = level[node] + 1
                        queue.append(to)
            if level[sink] < 0:
                return flow
            it = [0] * n

            def dfs(v: int, f: int) -> int:
                if v == sink:
                    return f
                while it[v] < len(self.graph[v]):
                    edge = self.graph[v][it[v]]
                    to, cap, rev = edge
                    if cap > 0 and level[v] < level[to]:
                        d = dfs(to, min(f, cap))
                        if d > 0:
                            edge[1] -= d
                            self.graph[to][rev][1] += d
                            return d
                    it[v] += 1
                return 0

            while True:
                pushed = dfs(source, 10**9)
                if pushed == 0:
                    break
                flow += pushed


def solve_quota_flow(
    units: pd.DataFrame,
    mapping: pd.DataFrame,
    cells: list[tuple[str, int]],
    per_target_label: int,
) -> list[tuple[str, str]]:
    source = 0
    unit_offset = 1
    cell_offset = unit_offset + len(units)
    sink = cell_offset + len(cells)
    flow = Dinic(sink + 1)

    unit_node = {str(unit_id): unit_offset + idx for idx, unit_id in enumerate(units["unit_id"].astype(str).tolist())}
    cell_node = {cell: cell_offset + idx for idx, cell in enumerate(cells)}
    unit_to_cell_edges: dict[tuple[str, tuple[str, int]], tuple[int, int]] = {}

    for unit_id in unit_node:
        flow.add_edge(source, unit_node[unit_id], 1)

    for _, row in units.iterrows():
        unit_id = str(row["unit_id"])
        unit_targets = set(str(row["target_keys"]).split(";"))
        label = int(row["label"])
        for target_key in unit_targets:
            cell = (target_key, label)
            if cell in cell_node:
                edge_idx = flow.add_edge(unit_node[unit_id], cell_node[cell], 1)
                unit_to_cell_edges[(unit_id, cell)] = (unit_node[unit_id], edge_idx)

    for cell in cells:
        flow.add_edge(cell_node[cell], sink, per_target_label)

    required = len(cells) * per_target_label
    achieved = flow.max_flow(source, sink)
    assignments = []
    for (unit_id, cell), (node, edge_idx) in unit_to_cell_edges.items():
        edge = flow.graph[node][edge_idx]
        if edge[1] == 0:
            assignments.append((unit_id, cell[0]))
    if achieved != required:
        print(f"[WARN] Exact target-label quota not feasible: achieved={achieved}, requested={required}. Reallocating extras.")
    return assignments


def add_flexible_extras(
    assignments: list[tuple[str, str]],
    units: pd.DataFrame,
    mapping: pd.DataFrame,
    cells: list[tuple[str, int]],
    target_n: int,
    seed: int,
) -> list[tuple[str, str]]:
    if len(assignments) >= target_n:
        return assignments

    selected_ids = {unit_id for unit_id, _ in assignments}
    unit_lookup = units.set_index("unit_id", drop=False)
    used_triples = set()
    for unit_id in selected_ids:
        row = unit_lookup.loc[unit_id]
        used_triples.add((row["premise"], row["hypothesis"], row["attacked_hypothesis"]))

    counts = {}
    for unit_id, primary_target_key in assignments:
        label = int(unit_lookup.loc[unit_id, "label"])
        counts[(primary_target_key, label)] = counts.get((primary_target_key, label), 0) + 1
    for cell in cells:
        counts.setdefault(cell, 0)

    rng_units = units.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    while len(assignments) < target_n:
        added = False
        for cell in sorted(cells, key=lambda item: (counts[item], item[0], item[1])):
            target_key, label = cell
            candidate_ids = set(
                mapping[(mapping["target_key"] == target_key) & (mapping["label"] == label)]["unit_id"].astype(str)
            )
            pool = rng_units[
                rng_units["unit_id"].astype(str).isin(candidate_ids)
                & ~rng_units["unit_id"].astype(str).isin(selected_ids)
            ].copy()
            if pool.empty:
                continue
            pool["_triple"] = list(zip(pool["premise"], pool["hypothesis"], pool["attacked_hypothesis"]))
            pool = pool[~pool["_triple"].isin(used_triples)]
            if pool.empty:
                continue
            picked = pool.iloc[0]
            unit_id = str(picked["unit_id"])
            assignments.append((unit_id, target_key))
            selected_ids.add(unit_id)
            used_triples.add((picked["premise"], picked["hypothesis"], picked["attacked_hypothesis"]))
            counts[cell] += 1
            added = True
            break
        if not added:
            raise ValueError(f"Could not add flexible BT extras to reach target_n={target_n}")
    return assignments


def make_annotation_sheet(sampled: pd.DataFrame, seed: int) -> pd.DataFrame:
    sheet = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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


def write_guidelines(output_dir: Path) -> None:
    taxonomy = "\n".join(f"- {item}" for item in DRIFT_TAXONOMY)
    english = f"""# Back Translation Human Validation Guideline

Each row is one unique Back Translation attack-success paraphrase unit.
Duplicate units across target models have been removed.

## Task

Judge only the NLI relation between `premise` and `backtranslated_hypothesis`.
Do not re-label the original premise-hypothesis pair. The original gold label is stored
only in the answer key and will be used during aggregation.

The `original_hypothesis` column is provided only to help identify semantic drift.

## Label Values

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = Unclear or not reliably judgeable

## Columns to Fill

- `paraphrased_pair_label`: NLI label for premise + backtranslated hypothesis.
- `semantic_drift_type`: choose one taxonomy label below.
- `fluency_defect`: Y if the backtranslation has a grammar, formatting, or readability defect; otherwise N.
- `annotator_confidence`: high, medium, or low.

## Semantic Drift Taxonomy

{taxonomy}
"""
    korean = f"""# BT Human Validation 평가 지침

각 행은 Back Translation attack-success 사례에서 뽑은 고유 문항입니다.
동일한 BT 문장이 여러 target model에서 attack-success가 된 경우에도 한 번만 평가합니다.

## 평가자가 할 일

평가자는 `premise`와 `backtranslated_hypothesis`의 NLI 관계만 판단합니다.
`original_hypothesis`는 의미 변화 여부를 참고하기 위한 문장입니다.
평가자는 original pair의 label을 다시 판단하지 않습니다.

## 채워야 하는 열

`paraphrased_pair_label`

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = 판단 불가 또는 애매함

`semantic_drift_type`

{taxonomy}

`fluency_defect`

- Y = 문법, 형식, 가독성 문제가 있음
- N = 특별한 문제가 없음

`annotator_confidence`

- high
- medium
- low

평가자에게 answer key, model prediction, target model 정보는 제공하지 않습니다.
"""
    (output_dir / "bt_human_validation_annotation_guideline.md").write_text(english, encoding="utf-8")
    (output_dir / "bt_human_validation_annotation_guideline_ko.md").write_text(korean, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 180 Back Translation human-validation units.")
    parser.add_argument("--per_target_label", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--output_dir", default="results/05_human_evaluation/bt_label_invariance_180")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    units, mapping = build_units_and_mapping(candidates)
    sampled, quota = sample_target_label_balanced(units, mapping, args.per_target_label, args.seed)
    sampled_mapping = mapping[mapping["unit_id"].isin(sampled["unit_id"])].merge(
        sampled[["unit_id", "annotation_id", "primary_target_key"]],
        on="unit_id",
        how="left",
    )

    answer_key = sampled[
        [
            "annotation_id",
            "unit_id",
            "primary_target_key",
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

    cell_coverage = (
        sampled_mapping.groupby(["target_key", "label", "label_name"], as_index=False)
        .agg(
            sampled_unique_units=("unit_id", "nunique"),
            represented_attack_success_instances=("unit_id", "size"),
        )
        .sort_values(["target_key", "label"])
    )

    candidates.to_csv(output_dir / "bt_human_validation_attack_success_candidates.csv", index=False, encoding="utf-8-sig")
    units.to_csv(output_dir / "bt_human_validation_unique_units.csv", index=False, encoding="utf-8-sig")
    quota.to_csv(output_dir / "bt_human_validation_sampling_quota.csv", index=False, encoding="utf-8-sig")
    answer_key.to_csv(output_dir / "bt_human_validation_answer_key.csv", index=False, encoding="utf-8-sig")
    sampled_mapping.to_csv(output_dir / "bt_human_validation_instance_mapping.csv", index=False, encoding="utf-8-sig")
    cell_coverage.to_csv(output_dir / "bt_human_validation_cell_coverage.csv", index=False, encoding="utf-8-sig")
    for idx in [1, 2, 3]:
        make_annotation_sheet(sampled, args.seed + idx).to_csv(
            output_dir / f"bt_human_validation_annotation_sheet_researcher{idx}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_guidelines(output_dir)

    summary_lines = [
        "BT human-validation sampling summary",
        "====================================",
        f"BT attack-success instances: {len(candidates)}",
        f"Unique BT paraphrase units: {len(units)}",
        f"Sampled annotation units: {len(answer_key)}",
        f"Represented attack-success instances: {len(sampled_mapping)}",
        f"Design: 4 targets x 3 labels x {args.per_target_label} = {4 * 3 * args.per_target_label}",
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
