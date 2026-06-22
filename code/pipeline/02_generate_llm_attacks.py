import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PARAPHRASE_SYSTEM_PROMPT, PARAPHRASE_USER_TEMPLATE, DEFAULT_MODELS, GENERATION_PARAMS, GENERATED_ATTACK_DIR, SAMPLED_SOURCE_DIR, llm_attack_filename, sampled_source_filename
from utils.common import postprocess_one_sentence, dedup_full_repeat, now_iso, dump_debug
from utils.api_clients import call_with_retry, call_model

def generate_paraphrase(generator: str, text: str, model: str) -> str:
    params = GENERATION_PARAMS[generator].copy()
    params['model'] = model
    params['system_prompt'] = PARAPHRASE_SYSTEM_PROMPT
    user_text = PARAPHRASE_USER_TEMPLATE.format(text=text)

    def _call():
        raw = call_model(generator, user_text, **params)
        out = postprocess_one_sentence(raw)
        out = dedup_full_repeat(out)
        return out
    return call_with_retry(_call, max_retries=4)

def main():
    ap = argparse.ArgumentParser(description='Generate LLM paraphrase attacks.')
    ap.add_argument('--generator', required=True, choices=['clova', 'gemini', 'openai', 'sonnet'])
    ap.add_argument('--input', type=str, default=str(Path(SAMPLED_SOURCE_DIR) / sampled_source_filename()))
    ap.add_argument('--output', type=str, default='')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--model', type=str, default='')
    ap.add_argument('--flush_every', type=int, default=50)
    ap.add_argument('--min_interval', type=float, default=0.2, help='Minimum interval between API calls in seconds.')
    ap.add_argument('--debug_dir', type=str, default='debug')
    args = ap.parse_args()
    generator = args.generator
    model = args.model or DEFAULT_MODELS[generator]
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else Path(GENERATED_ATTACK_DIR) / llm_attack_filename(generator, args.n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) / f'{generator}_generate'
    df = pd.read_csv(in_path, encoding='utf-8-sig')
    for col in ('id', 'premise', 'hypothesis', 'label'):
        assert col in df.columns, f'Missing column: {col}'
    df = df.sort_values('id').reset_index(drop=True).head(args.n).copy()
    done_ids = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path, encoding='utf-8-sig')
        done_ids = set(prev['id'].tolist()) if 'id' in prev.columns else set()
    buf = []
    last_call_ts = 0.0
    print(f'[{generator}] model={model} | n={len(df)} -> {out_path} (resume={args.resume})')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rid = int(row['id'])
        if rid in done_ids:
            continue
        now = time.time()
        wait = args.min_interval - (now - last_call_ts)
        if wait > 0:
            time.sleep(wait)
        src = str(row['hypothesis'])
        try:
            para = generate_paraphrase(generator, src, model)
            if not para.strip():
                para = 'Error'
        except Exception as e:
            para = 'Error'
            dump_debug(debug_dir, {'time': now_iso(), 'id': rid, 'kind': 'generate_fail', 'target': generator, 'last_exc': repr(e), 'hypothesis_preview': src[:500]})
        buf.append({'id': rid, 'premise': row['premise'], 'hypothesis': row['hypothesis'], 'label': int(row['label']), 'attacked_hypothesis': para, 'generator': generator})
        last_call_ts = time.time()
        if len(buf) % args.flush_every == 0:
            _flush(buf, out_path)
    if buf:
        _flush(buf, out_path)
    print(f'Done: {out_path}')
    print(pd.read_csv(out_path, encoding='utf-8-sig').head(3))

def _flush(buf: list, out_path: Path):
    tmp = pd.DataFrame(buf)
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding='utf-8-sig')
        tmp = pd.concat([prev, tmp], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    tmp.to_csv(out_path, index=False, encoding='utf-8-sig')
if __name__ == '__main__':
    main()
