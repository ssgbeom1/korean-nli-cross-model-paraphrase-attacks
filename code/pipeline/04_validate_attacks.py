from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
from bert_score import score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BERTSCORE_LANG, BERTSCORE_THRESHOLD
INVALID_STRINGS = {'', 'nan', 'none'}

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(subset=['hypothesis', 'attacked_hypothesis']).copy()
    for column in ('hypothesis', 'attacked_hypothesis'):
        cleaned[column] = cleaned[column].astype(str).str.strip()
    return cleaned

def main() -> None:
    parser = argparse.ArgumentParser(description='Validate paraphrases with BERTScore.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=BERTSCORE_THRESHOLD)
    parser.add_argument('--lang', type=str, default=BERTSCORE_LANG)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')
    print(f'Loading: {input_path}')
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    for column in ('hypothesis', 'attacked_hypothesis'):
        assert column in df.columns, f'Missing column: {column}'
    before = len(df)
    df = normalize_text_columns(df)
    df = df[~df['hypothesis'].str.lower().isin(INVALID_STRINGS)].copy()
    df = df[~df['attacked_hypothesis'].str.lower().isin(INVALID_STRINGS)].copy()
    df = df[df['attacked_hypothesis'].str.lower().ne('error')].copy()
    after = len(df)
    if after == 0:
        print('No valid rows after filtering.')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return
    print(f'  rows: {before} -> {after} (after removing null/empty/Error rows)')
    refs = df['hypothesis'].tolist()
    cands = df['attacked_hypothesis'].tolist()
    print(f'Computing BERTScore (lang={args.lang}, batch={args.batch_size})...')
    _, _, f1 = score(cands, refs, lang=args.lang, verbose=True, batch_size=args.batch_size)
    df['bert_score'] = [float(value) for value in f1.tolist()]
    valid_df = df[df['bert_score'] >= args.threshold].copy()
    print(f'  pass (>= {args.threshold}): {len(valid_df)}')
    print(f'  drop (<  {args.threshold}): {len(df) - len(valid_df)}')
    valid_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Saved: {output_path}')
if __name__ == '__main__':
    main()
