from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import (  # noqa: E402
    CROSS_MODEL_EVAL_DIR,
    GENERATORS,
    TARGETS,
    cross_model_generator_dirname,
    model_file_key,
    target_output_filename,
)
from utils.original_cache import apply_original_cache  # noqa: E402


def main() -> None:
    cross_model_dir = ROOT / CROSS_MODEL_EVAL_DIR
    report_rows = []
    for generator in GENERATORS:
        generator_key = model_file_key(generator)
        generator_dir = cross_model_dir / cross_model_generator_dirname(generator)
        for target in TARGETS:
            target_key = model_file_key(target)
            path = generator_dir / target_output_filename(target)
            if not path.exists():
                raise FileNotFoundError(path)
            before = pd.read_csv(path, encoding="utf-8-sig")
            synced = apply_original_cache(before, cross_model_dir, target_key)
            mismatch_count = int(synced["original_cache_mismatch"].sum())
            success_before = int(before["attack_success"].astype(int).sum())
            success_after = int(synced["attack_success"].astype(int).sum())
            correct_before = int(before["original_correct"].astype(int).sum())
            correct_after = int(synced["original_correct"].astype(int).sum())
            synced = synced.drop(columns=["original_cache_mismatch"])
            synced.to_csv(path, index=False, encoding="utf-8-sig")
            report_rows.append(
                {
                    "generator": generator_key,
                    "target": target_key,
                    "path": str(path.relative_to(ROOT)),
                    "rows": len(synced),
                    "original_cache_mismatch_before_sync": mismatch_count,
                    "original_correct_before": correct_before,
                    "original_correct_after": correct_after,
                    "attack_success_before": success_before,
                    "attack_success_after": success_after,
                    "delta_original_correct": correct_after - correct_before,
                    "delta_attack_success": success_after - success_before,
                }
            )

    output_dir = ROOT / "results" / "06_integrity_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(report_rows)
    report_path = output_dir / "original_cache_sync_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {report_path}")
    print(report.to_string(index=False))
    print(f"Total original-cache mismatches fixed: {int(report['original_cache_mismatch_before_sync'].sum())}")


if __name__ == "__main__":
    main()
