from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_original_cache(cross_model_dir: str | Path, target_key: str) -> pd.DataFrame:
    path = Path(cross_model_dir) / "original_predictions" / f"{target_key}_original_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing original prediction cache: {path}")
    cache = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "label", "pred_original"}
    missing = required.difference(cache.columns)
    if missing:
        raise ValueError(f"Missing required cache columns in {path}: {sorted(missing)}")
    keep = ["id", "label", "pred_original"]
    if "raw_original" in cache.columns:
        keep.append("raw_original")
    cache = cache[keep].copy()
    rename = {
        "label": "cache_label",
        "pred_original": "cache_pred_original",
        "raw_original": "cache_raw_original",
    }
    return cache.rename(columns=rename)


def apply_original_cache(df: pd.DataFrame, cross_model_dir: str | Path, target_key: str) -> pd.DataFrame:
    cache = load_original_cache(cross_model_dir, target_key)
    out = df.copy()
    for column in ["id", "label", "pred_attacked"]:
        if column not in out.columns:
            raise ValueError(f"Missing required evaluation column: {column}")

    out = out.merge(cache, on="id", how="left", validate="many_to_one")
    if out["cache_pred_original"].isna().any():
        missing_ids = out.loc[out["cache_pred_original"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Original cache missing ids for target {target_key}: {missing_ids}")

    label_mismatch = out["label"].astype(int) != out["cache_label"].astype(int)
    if label_mismatch.any():
        bad_ids = out.loc[label_mismatch, "id"].head(10).tolist()
        raise ValueError(f"Label mismatch against original cache for target {target_key}: {bad_ids}")

    if "pred_original" in out.columns:
        out["original_cache_mismatch"] = (
            out["pred_original"].notna()
            & (out["pred_original"].astype(int) != out["cache_pred_original"].astype(int))
        ).astype(int)
    else:
        out["original_cache_mismatch"] = 0

    out["pred_original"] = out["cache_pred_original"].astype(int)
    if "cache_raw_original" in out.columns:
        out["raw_original"] = out["cache_raw_original"]

    out["label"] = out["label"].astype(int)
    out["pred_attacked"] = out["pred_attacked"].astype(int)
    out["original_correct"] = (out["pred_original"] == out["label"]).astype(int)
    out["attacked_correct"] = (out["pred_attacked"] == out["label"]).astype(int)
    out["attack_success"] = (
        (out["original_correct"] == 1) & (out["pred_attacked"] != out["label"])
    ).astype(int)

    drop_cols = [column for column in ["cache_label", "cache_pred_original", "cache_raw_original"] if column in out.columns]
    return out.drop(columns=drop_cols)
