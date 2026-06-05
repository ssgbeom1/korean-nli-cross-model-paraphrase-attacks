from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CROSS_MODEL_CACHE_DIR, DEFAULT_MODELS, EVAL_PARAMS, NLI_EVAL_PROMPT, original_prediction_cache_filename
from utils.api_clients import call_model
from utils.common import dump_debug, now_iso, parse_label

def predict_label(target: str, prompt: str, model_id: str, eval_params: dict, debug_dir: Path | None=None, rid: int=-1, kind: str='', max_attempts: int=3) -> tuple[int | None, str | None]:
    last_exc = None
    last_txt = None
    for attempt in range(1, max_attempts + 1):
        try:
            last_txt = call_model(target, prompt, model=model_id, **eval_params)
            label = parse_label(last_txt)
            if label is not None:
                return (label, last_txt)
        except Exception as exc:
            last_exc = repr(exc)
        time.sleep(0.6 * attempt)
    tqdm.write(f'[WARN] target={target} id={rid} kind={kind} last_exc={last_exc} last_txt={repr(last_txt)[:200]}')
    if debug_dir is not None:
        dump_debug(debug_dir, {'time': now_iso(), 'id': rid, 'kind': kind, 'target': target, 'model_id': model_id, 'attempts': max_attempts, 'last_exc': last_exc, 'last_txt': last_txt, 'prompt_preview': prompt[:800]})
    return (None, last_txt)

def load_original_cache(cache_path: Path) -> dict[int, tuple[int | None, str | None]]:
    cache: dict[int, tuple[int | None, str | None]] = {}
    if cache_path and cache_path.exists():
        df = pd.read_csv(cache_path, encoding='utf-8-sig')
        for _, row in df.iterrows():
            pred = int(row['pred_original']) if pd.notna(row['pred_original']) else None
            raw = str(row['raw_original']) if pd.notna(row['raw_original']) else None
            cache[int(row['id'])] = (pred, raw)
        print(f'  Loaded original cache: {len(cache)} entries from {cache_path}')
    return cache

def merge_by_id(previous: pd.DataFrame | None, current: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if previous is not None and (not previous.empty):
        frames.append(previous)
    if current is not None and (not current.empty):
        frames.append(current)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['id'], keep='last')
    return merged.sort_values('id').reset_index(drop=True)

def save_cache_rows(cache_path: Path, rows: pd.DataFrame) -> None:
    previous = None
    if cache_path.exists():
        previous = pd.read_csv(cache_path, encoding='utf-8-sig')
    merged = merge_by_id(previous, rows)
    merged.to_csv(cache_path, index=False, encoding='utf-8-sig')

def completed_ids(df: pd.DataFrame | None, required_columns: list[str]) -> set[int]:
    if df is None or df.empty or 'id' not in df.columns:
        return set()
    mask = pd.Series(True, index=df.index)
    for column in required_columns:
        if column not in df.columns:
            return set()
        mask &= df[column].notna()
    return set(df.loc[mask, 'id'].astype(int).tolist())

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate attacked NLI examples.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--target', required=True, choices=['clova', 'openai', 'gemini', 'sonnet'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debug_dir', default='debug')
    parser.add_argument('--model', default='', help='Override model id.')
    parser.add_argument('--original_cache', default='', help='Path to an original-prediction cache CSV.')
    parser.add_argument('--save_original_cache', default='', help='Optional path to save original predictions.')
    parser.add_argument('--cache_originals', action='store_true', help='Only compute original predictions and save the cache.')
    parser.add_argument('--row_delay', type=float, default=0.0, help='Optional delay after each processed row.')
    args = parser.parse_args()
    target = args.target
    model_id = args.model or DEFAULT_MODELS[target]
    eval_params = EVAL_PARAMS[target].copy()
    input_path = Path(args.input)
    debug_dir = Path(args.debug_dir) / f'{target}_eval' if args.debug_dir else None
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    for column in ('id', 'premise', 'hypothesis', 'label', 'attacked_hypothesis'):
        assert column in df.columns, f'Missing column: {column}'
    original_cache: dict[int, tuple[int | None, str | None]] = {}
    if args.original_cache:
        original_cache = load_original_cache(Path(args.original_cache))
    if args.cache_originals:
        cache_path = Path(args.save_original_cache or Path(CROSS_MODEL_CACHE_DIR) / original_prediction_cache_filename(target))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_csv(cache_path, encoding='utf-8-sig') if cache_path.exists() else None
        existing_ids = completed_ids(existing, ['pred_original'])
        buffer = []
        print(f'[CACHE MODE] target={target} model={model_id} -> {cache_path}')
        for _, row in tqdm(df.iterrows(), total=len(df)):
            rid = int(row['id'])
            if rid in existing_ids:
                continue
            prompt = NLI_EVAL_PROMPT.format(premise=str(row['premise']), hypothesis=str(row['hypothesis']))
            pred, raw = predict_label(target, prompt, model_id, eval_params, debug_dir, rid, 'original')
            buffer.append({'id': rid, 'label': int(row['label']), 'pred_original': pred, 'raw_original': raw})
            if args.row_delay > 0:
                time.sleep(args.row_delay)
        if buffer:
            save_cache_rows(cache_path, pd.DataFrame(buffer))
        elif existing is not None:
            existing.to_csv(cache_path, index=False, encoding='utf-8-sig')
        print(f'  Saved: {cache_path}')
        return
    if not args.output:
        raise ValueError('--output is required unless --cache_originals is set.')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_results = None
    done_ids: set[int] = set()
    if args.resume and output_path.exists():
        previous_results = pd.read_csv(output_path, encoding='utf-8-sig')
        done_ids = completed_ids(previous_results, ['pred_original', 'pred_attacked'])
    rows = []
    print(f'[EVAL] target={target} model={model_id}')
    print(f'  input={input_path} | output={output_path} | cache={len(original_cache)} entries')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rid = int(row['id'])
        if rid in done_ids:
            continue
        premise = str(row['premise'])
        hypothesis = str(row['hypothesis'])
        attacked_hypothesis = str(row['attacked_hypothesis'])
        gold = int(row['label'])
        if rid in original_cache:
            pred_original, raw_original = original_cache[rid]
        else:
            prompt = NLI_EVAL_PROMPT.format(premise=premise, hypothesis=hypothesis)
            pred_original, raw_original = predict_label(target, prompt, model_id, eval_params, debug_dir, rid, 'original')
        attacked_prompt = NLI_EVAL_PROMPT.format(premise=premise, hypothesis=attacked_hypothesis)
        pred_attacked, raw_attacked = predict_label(target, attacked_prompt, model_id, eval_params, debug_dir, rid, 'attacked')
        rows.append({'id': rid, 'generator': '' if pd.isna(row.get('generator', '')) else row.get('generator', ''), 'target': target, 'target_model': model_id, 'label': gold, 'pred_original': pred_original, 'pred_attacked': pred_attacked, 'raw_original': raw_original, 'raw_attacked': raw_attacked, 'original_correct': int(pred_original == gold) if pred_original is not None else 0, 'attacked_correct': int(pred_attacked == gold) if pred_attacked is not None else 0, 'attack_success': int(pred_original == gold and pred_attacked is not None and (pred_attacked != gold))})
        if args.row_delay > 0:
            time.sleep(args.row_delay)
    current_results = pd.DataFrame(rows)
    result_df = merge_by_id(previous_results, current_results)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    if args.save_original_cache:
        cache_rows = result_df[['id', 'label', 'pred_original', 'raw_original']].copy()
        cache_path = Path(args.save_original_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_cache_rows(cache_path, cache_rows)
        print(f'  Original cache saved: {cache_path}')
    output_df = pd.read_csv(output_path, encoding='utf-8-sig')
    n_correct = int(output_df['original_correct'].sum())
    n_success = int(output_df['attack_success'].sum())
    orig_acc = float(output_df['original_correct'].mean()) if len(output_df) else 0.0
    asr = n_success / n_correct if n_correct > 0 else 0.0
    print(f'Saved: {output_path}')
    print(f'  Orig Acc: {orig_acc:.4f} | ASR: {asr:.4f} ({n_success}/{n_correct})')
if __name__ == '__main__':
    main()
