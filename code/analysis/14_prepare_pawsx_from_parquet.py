from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}


def read_pawsx(path: Path, language: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {}
    if "id" in df.columns:
        rename["id"] = "row_idx"
    df = df.rename(columns=rename)
    if "row_idx" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_idx"})
    needed = ["row_idx", "sentence1", "sentence2", "label"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[needed].copy()
    out["language"] = language
    out["label"] = out["label"].astype(int)
    out = out.drop_duplicates(subset=["language", "row_idx"]).reset_index(drop=True)
    out = out[out["sentence1"].astype(str).str.strip().ne("")]
    out = out[out["sentence2"].astype(str).str.strip().ne("")]
    return out[["row_idx", "language", "sentence1", "sentence2", "label"]].reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, n_per_label: int, seed: int) -> pd.DataFrame:
    frames = []
    for label in [0, 1]:
        subset = df[df["label"] == label]
        take = min(n_per_label, len(subset))
        frames.append(subset.sample(n=take, random_state=seed + label))
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_prompts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prompts = []
    for _, row in out.iterrows():
        prompts.append(
            "You are evaluating semantic equivalence for a paraphrase-identification benchmark.\n"
            f"Language: {LANGUAGE_NAMES[row['language']]}\n"
            "Return only one digit: 1 if Sentence A and Sentence B have the same meaning, "
            "or 0 if they have different meanings.\n\n"
            f"Sentence A: {row['sentence1']}\n"
            f"Sentence B: {row['sentence2']}\n"
            "Label:"
        )
    out["prompt"] = prompts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PAWS-X samples from downloaded parquet files.")
    parser.add_argument("--raw_dir", default="data/05_external_benchmark/pawsx_raw")
    parser.add_argument("--output_dir", default="data/05_external_benchmark/pawsx")
    parser.add_argument("--n_per_label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    combined_samples = []
    for language in ["ko", "en"]:
        path = raw_dir / f"pawsx_{language}_test.parquet"
        full = read_pawsx(path, language)
        sample = add_prompts(balanced_sample(full, args.n_per_label, args.seed))
        full.to_csv(output_dir / f"pawsx_{language}_test_clean.csv", index=False, encoding="utf-8-sig")
        sample.to_csv(output_dir / f"pawsx_{language}_balanced_sample.csv", index=False, encoding="utf-8-sig")
        combined_samples.append(sample)
        for label, subset in full.groupby("label"):
            summary_rows.append({"language": language, "split": "test_clean", "label": int(label), "n": len(subset)})
        for label, subset in sample.groupby("label"):
            summary_rows.append({"language": language, "split": "balanced_sample", "label": int(label), "n": len(subset)})

    combined = pd.concat(combined_samples, ignore_index=True)
    combined.to_csv(output_dir / "pawsx_ko_en_balanced_sample.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "pawsx_external_benchmark_sample_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Combined sample rows: {len(combined)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
