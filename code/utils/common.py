from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def postprocess_one_sentence(text: str) -> str:
    if not text:
        return ''
    cleaned = str(text).strip().strip('"').strip("'").strip()
    cleaned = cleaned.splitlines()[0].strip()
    # Korean prefixes are retained because models may echo the Korean prompt.
    for prefix in ['패러프레이즈:', '변형:', '답:', '- ', '결과:']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned

def dedup_full_repeat(text: str) -> str:
    if not text:
        return text
    if len(text) % 2 == 0:
        half = len(text) // 2
        if text[:half] == text[half:]:
            return text[:half]
    return text

def parse_label(text: str) -> int | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    match = re.search('\\b([012])\\b', cleaned)
    if match:
        return int(match.group(1))
    match = re.search('label\\s*:\\s*([012])', cleaned, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    lowered = cleaned.lower()
    if 'entail' in lowered or '함의' in lowered:
        return 0
    if 'neutral' in lowered or '중립' in lowered:
        return 1
    if 'contrad' in lowered or '모순' in lowered:
        return 2
    return None

def is_retryable_error(exc: Exception) -> bool:
    message = (repr(exc) or '').lower()
    keys = ['429', 'resource_exhausted', 'quota', 'rate', 'too many requests', '503', 'unavailable', 'deadline', 'timeout', 'timed out', '500', 'internal', 'connection', 'reset', 'econnreset']
    return any((key in message for key in keys))

def dump_debug(debug_dir: Path, payload: dict) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    rid = payload.get('id', 'NA')
    kind = payload.get('kind', 'NA')
    target = payload.get('target', 'NA')
    path = debug_dir / f'{timestamp}__{target}__id{rid}__{kind}.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path

def stable_int_from_text(text: str, modulo: int | None=None) -> int:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    value = int(digest[:16], 16)
    if modulo is not None and modulo > 0:
        return value % modulo
    return value

def majority_vote(values: list[object]) -> object | None:
    filtered = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in {'nan', 'none'}:
            continue
        filtered.append(value)
    if not filtered:
        return None
    counts = Counter(filtered)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    if len(winners) != 1 or top_count <= len(filtered) / 2:
        return None
    return winners[0]
