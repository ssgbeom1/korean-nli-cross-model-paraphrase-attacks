from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_MODELS, EVAL_PARAMS  # noqa: E402
from utils.api_clients import call_model  # noqa: E402


def parse_binary_label(text: str) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    if re.fullmatch(r"[01]", cleaned):
        return int(cleaned)
    match = re.search(r"(?:^|label\s*[:=]\s*|answer\s*[:=]\s*)([01])(?:\D|$)", cleaned, flags=re.I)
    if match:
        return int(match.group(1))

    lowered = re.sub(r"\s+", " ", cleaned.lower())
    negative_patterns = [
        "not equivalent",
        "not the same",
        "not a paraphrase",
        "not paraphrases",
        "different meaning",
        "different meanings",
        "do not have the same meaning",
        "does not have the same meaning",
        "are different",
    ]
    positive_patterns = [
        "same meaning",
        "same meanings",
        "equivalent",
        "paraphrase",
        "both sentences convey",
        "both sentences mean",
        "same information",
    ]
    if any(pattern in lowered for pattern in negative_patterns):
        return 0
    if any(pattern in lowered for pattern in positive_patterns):
        return 1
    return None


def strict_prompt(sentence1: str, sentence2: str) -> str:
    return (
        "Semantic equivalence classification.\n"
        "Output exactly one character and nothing else.\n"
        "1 = Sentence A and Sentence B have the same meaning.\n"
        "0 = Sentence A and Sentence B have different meanings.\n\n"
        f"Sentence A: {sentence1}\n"
        f"Sentence B: {sentence2}\n"
        "Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAWS-X external paraphrase benchmark with an LLM target.")
    parser.add_argument("--input", required=True, help="Prepared PAWS-X CSV with a prompt column.")
    parser.add_argument("--target", required=True, choices=["clova", "openai", "gemini", "sonnet"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--parse_retries", type=int, default=2)
    args = parser.parse_args()

    target = args.target
    model = args.model or DEFAULT_MODELS[target]
    params = EVAL_PARAMS[target].copy()
    params["max_tokens"] = max(32, int(params.get("max_tokens", 16)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a prompt column. Run 14_prepare_pawsx_from_parquet.py first.")

    previous = pd.read_csv(output_path, encoding="utf-8-sig") if args.resume and output_path.exists() else pd.DataFrame()
    done = (
        set(zip(previous["language"].astype(str), previous["row_idx"].astype(int)))
        if {"language", "row_idx"}.issubset(previous.columns)
        else set()
    )
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_idx = int(row["row_idx"])
        row_key = (str(row["language"]), row_idx)
        if row_key in done:
            continue
        raw = ""
        pred = None
        try:
            raw = call_model(target, row["prompt"], model=model, **params)
            pred = parse_binary_label(raw)
            for _ in range(args.parse_retries):
                if pred is not None:
                    break
                if args.sleep:
                    time.sleep(max(args.sleep, 0.2))
                raw = call_model(
                    target,
                    strict_prompt(str(row["sentence1"]), str(row["sentence2"])),
                    model=model,
                    **params,
                )
                pred = parse_binary_label(raw)
        except Exception as exc:
            raw = f"ERROR: {repr(exc)}"
        rows.append(
            {
                "row_idx": row_idx,
                "language": row["language"],
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": int(row["label"]),
                "target": target,
                "target_model": model,
                "pred": pred,
                "raw": raw,
                "correct": int(pred == int(row["label"])) if pred is not None else 0,
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    current = pd.DataFrame(rows)
    combined = pd.concat([previous, current], ignore_index=True) if len(previous) else current
    if len(combined):
        combined = combined.drop_duplicates(subset=["language", "row_idx"], keep="last").sort_values(["language", "row_idx"])
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    n = len(combined)
    valid = combined["pred"].notna().sum() if n else 0
    acc = combined["correct"].mean() if n else 0.0
    print(f"Saved: {output_path}")
    print(f"rows={n}, valid_predictions={valid}, accuracy={acc * 100:.2f}%")


if __name__ == "__main__":
    main()
