from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_KEYS = ["hyperclova_x", "gemini", "gpt", "claude_sonnet"]
PRED_COLUMNS = ["pred_original", "pred_attacked"]
VALID_LABELS = {0, 1, 2}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def invalid_values(series: pd.Series) -> list[str]:
    values = set()
    for value in series.dropna().tolist():
        try:
            numeric = int(value)
        except Exception:
            values.add(str(value))
            continue
        if numeric not in VALID_LABELS:
            values.add(str(value))
    return sorted(values)


def as_int_or_none(value):
    if pd.isna(value):
        return None
    return int(value)


def check_consistency(df: pd.DataFrame) -> dict[str, int]:
    problems = {
        "original_correct_mismatch": 0,
        "attacked_correct_mismatch": 0,
        "attack_success_mismatch": 0,
    }
    for _, row in df.iterrows():
        label = as_int_or_none(row.get("label"))
        pred_original = as_int_or_none(row.get("pred_original"))
        pred_attacked = as_int_or_none(row.get("pred_attacked"))
        expected_original_correct = int(pred_original == label) if pred_original is not None else 0
        expected_attacked_correct = int(pred_attacked == label) if pred_attacked is not None else 0
        expected_attack_success = int(
            pred_original == label and pred_attacked is not None and pred_attacked != label
        )
        if int(row.get("original_correct", -1)) != expected_original_correct:
            problems["original_correct_mismatch"] += 1
        if int(row.get("attacked_correct", -1)) != expected_attacked_correct:
            problems["attacked_correct_mismatch"] += 1
        if int(row.get("attack_success", -1)) != expected_attack_success:
            problems["attack_success_mismatch"] += 1
    return problems


def scan_logs(paths: list[Path]) -> pd.DataFrame:
    patterns = ["42901", "ssl", "connection", "warn", "error"]
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path), "exists": False, "line_number": "", "line": ""})
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            lower = line.lower()
            if any(pattern in lower for pattern in patterns):
                rows.append(
                    {
                        "path": str(path),
                        "exists": True,
                        "line_number": line_number,
                        "line": line[:500],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify 4x4 evaluation integrity.")
    parser.add_argument("--expected_rows", type=int, default=2209)
    parser.add_argument("--result_dir", default="results/01_cross_model_evaluation")
    parser.add_argument("--shared_ids", default="data/04_shared_valid_set/shared_valid_ids.csv")
    parser.add_argument("--output_dir", default="results/06_integrity_checks")
    parser.add_argument("--scan_logs", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_df = read_csv(Path(args.shared_ids))
    shared_ids = set(shared_df["id"].astype(int).tolist())

    cache_rows = []
    cell_rows = []
    fatal_problems = []
    original_caches: dict[str, pd.DataFrame] = {}

    cache_dir = result_dir / "original_predictions"
    for model in MODEL_KEYS:
        path = cache_dir / f"{model}_original_predictions.csv"
        if not path.exists():
            cache_rows.append({"model": model, "path": str(path), "exists": False})
            fatal_problems.append(f"Missing original cache: {path}")
            continue
        df = read_csv(path)
        ids = set(df["id"].astype(int).tolist()) if "id" in df.columns else set()
        duplicated = int(df["id"].duplicated().sum()) if "id" in df.columns else args.expected_rows
        missing_pred = int(df["pred_original"].isna().sum()) if "pred_original" in df.columns else args.expected_rows
        invalid_pred = invalid_values(df["pred_original"]) if "pred_original" in df.columns else ["missing_column"]
        id_set_mismatch = ids != shared_ids
        ok = (
            len(df) == args.expected_rows
            and duplicated == 0
            and missing_pred == 0
            and not invalid_pred
            and not id_set_mismatch
        )
        cache_rows.append(
            {
                "model": model,
                "path": str(path),
                "exists": True,
                "rows": len(df),
                "expected_rows": args.expected_rows,
                "duplicate_ids": duplicated,
                "missing_pred_original": missing_pred,
                "invalid_pred_original_values": ";".join(invalid_pred),
                "id_set_matches_shared": not id_set_mismatch,
                "ok": ok,
            }
        )
        if ok:
            original_caches[model] = df[["id", "pred_original"]].copy()
        if not ok:
            fatal_problems.append(f"Original cache failed integrity check: {path}")

    for generator in MODEL_KEYS:
        for target in MODEL_KEYS:
            path = result_dir / f"{generator}_as_generator" / f"to_{target}.csv"
            if not path.exists():
                cell_rows.append({"generator": generator, "target": target, "path": str(path), "exists": False})
                fatal_problems.append(f"Missing cell result: {path}")
                continue
            df = read_csv(path)
            ids = set(df["id"].astype(int).tolist()) if "id" in df.columns else set()
            duplicated = int(df["id"].duplicated().sum()) if "id" in df.columns else args.expected_rows
            missing_original = int(df["pred_original"].isna().sum()) if "pred_original" in df.columns else args.expected_rows
            missing_attacked = int(df["pred_attacked"].isna().sum()) if "pred_attacked" in df.columns else args.expected_rows
            invalid_original = invalid_values(df["pred_original"]) if "pred_original" in df.columns else ["missing_column"]
            invalid_attacked = invalid_values(df["pred_attacked"]) if "pred_attacked" in df.columns else ["missing_column"]
            if target in original_caches and "id" in df.columns and "pred_original" in df.columns:
                cache = original_caches[target].rename(columns={"pred_original": "cache_pred_original"})
                merged = df[["id", "pred_original"]].merge(cache, on="id", how="left", validate="many_to_one")
                missing_from_cache = int(merged["cache_pred_original"].isna().sum())
                original_cache_mismatch = int(
                    (
                        merged["cache_pred_original"].notna()
                        & (merged["pred_original"].astype(int) != merged["cache_pred_original"].astype(int))
                    ).sum()
                )
            else:
                missing_from_cache = args.expected_rows
                original_cache_mismatch = args.expected_rows
            consistency = check_consistency(df)
            id_set_mismatch = ids != shared_ids
            ok = (
                len(df) == args.expected_rows
                and duplicated == 0
                and missing_original == 0
                and missing_attacked == 0
                and not invalid_original
                and not invalid_attacked
                and not id_set_mismatch
                and missing_from_cache == 0
                and original_cache_mismatch == 0
                and all(value == 0 for value in consistency.values())
            )
            cell_rows.append(
                {
                    "generator": generator,
                    "target": target,
                    "path": str(path),
                    "exists": True,
                    "rows": len(df),
                    "expected_rows": args.expected_rows,
                    "duplicate_ids": duplicated,
                    "missing_pred_original": missing_original,
                    "missing_pred_attacked": missing_attacked,
                    "invalid_pred_original_values": ";".join(invalid_original),
                    "invalid_pred_attacked_values": ";".join(invalid_attacked),
                    "id_set_matches_shared": not id_set_mismatch,
                    "missing_from_original_cache": missing_from_cache,
                    "original_cache_mismatch": original_cache_mismatch,
                    **consistency,
                    "ok": ok,
                }
            )
            if not ok:
                fatal_problems.append(f"Cell failed integrity check: {path}")

    cache_report = pd.DataFrame(cache_rows)
    cell_report = pd.DataFrame(cell_rows)
    cache_report.to_csv(output_dir / "4x4_original_cache_integrity.csv", index=False, encoding="utf-8-sig")
    cell_report.to_csv(output_dir / "4x4_cell_integrity.csv", index=False, encoding="utf-8-sig")
    if args.scan_logs:
        log_report = scan_logs(
            [
                Path("logs/4x4_reeval_256_low_full.out.log"),
                Path("logs/4x4_reeval_256_low_full.err.log"),
                Path("logs/repair_clova_missing_predictions.out.log"),
                Path("logs/repair_clova_missing_predictions.err.log"),
            ]
        )
        log_report.to_csv(output_dir / "4x4_log_warning_scan.csv", index=False, encoding="utf-8-sig")

    completed_cells = int(cell_report["ok"].sum()) if "ok" in cell_report.columns else 0
    completed_caches = int(cache_report["ok"].sum()) if "ok" in cache_report.columns else 0
    cache_mismatches = int(cell_report["original_cache_mismatch"].sum()) if "original_cache_mismatch" in cell_report.columns else 0
    summary_lines = [
        "4x4 integrity verification",
        "==========================",
        f"Expected rows per cell: {args.expected_rows}",
        f"Shared valid ids: {len(shared_ids)}",
        f"Original caches OK: {completed_caches}/4",
        f"4x4 cells OK: {completed_cells}/16",
        f"Cell original-cache mismatches: {cache_mismatches}",
        f"Fatal problems: {len(fatal_problems)}",
        "",
        "Output files:",
        f"- {output_dir / '4x4_original_cache_integrity.csv'}",
        f"- {output_dir / '4x4_cell_integrity.csv'}",
    ]
    if args.scan_logs:
        summary_lines.append(f"- {output_dir / '4x4_log_warning_scan.csv'}")
    if fatal_problems:
        summary_lines.extend(["", "Problems:"])
        summary_lines.extend(f"- {problem}" for problem in fatal_problems)
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "4x4_integrity_summary.txt").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    if fatal_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
