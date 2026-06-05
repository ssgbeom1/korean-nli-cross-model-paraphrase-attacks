import json
import os
import random
import time
import uuid
from pathlib import Path
import requests
from dotenv import load_dotenv
from utils.common import is_retryable_error, dump_debug, now_iso
load_dotenv()

def call_with_retry(fn, *, max_retries: int=4, base_backoff: float=2.0, max_backoff: float=60.0) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            if result and result.strip():
                return result
        except Exception as e:
            last_err = e
            if is_retryable_error(e):
                sleep_s = base_backoff * 2 ** (attempt - 1)
                sleep_s = min(sleep_s, max_backoff)
                sleep_s += random.uniform(0, 0.25 * sleep_s)
            else:
                sleep_s = min(0.5 * attempt, 2.0)
            time.sleep(sleep_s)
    if last_err is not None:
        raise last_err
    return ''

def call_clova(text: str, *, model: str='HCX-005', max_tokens: int=16, temperature: float=0.0, repeat_penalty: float=1.1, top_p: float | None=None, system_prompt: str | None=None, capture_lines: list | None=None, capture_limit: int=200) -> str:
    host = 'https://clovastudio.stream.ntruss.com'
    raw_key = os.getenv('CLOVA_API_KEY')
    if not raw_key:
        raise RuntimeError('CLOVA_API_KEY 환경변수를 설정하세요.')
    api_key = raw_key if raw_key.startswith('Bearer') else f'Bearer {raw_key}'
    headers = {'Authorization': api_key, 'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4()).replace('-', ''), 'Content-Type': 'application/json; charset=utf-8', 'Accept': 'text/event-stream'}
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': text})
    payload = {'messages': messages, 'maxTokens': int(max_tokens), 'temperature': float(temperature), 'repeatPenalty': float(repeat_penalty), 'stopBefore': [], 'includeAiFilters': True, 'seed': 0}
    if top_p is not None:
        payload['topP'] = float(top_p)
    url = f'{host}/v3/chat-completions/{model}'

    def push_cap(s: str):
        if capture_lines is not None and len(capture_lines) < capture_limit:
            capture_lines.append(s[:500])
    response_text = ''
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        push_cap(f'HTTP {r.status_code}')
        if r.status_code != 200:
            push_cap(r.text)
            raise RuntimeError(f'CLOVA HTTP {r.status_code}: {r.text[:500]}')
        for bline in r.iter_lines():
            if not bline:
                continue
            decoded = bline.decode('utf-8', errors='ignore').strip()
            push_cap(decoded)
            if not decoded.startswith('data:'):
                continue
            js = decoded[5:].strip()
            if not js or js == '[DONE]':
                continue
            try:
                data = json.loads(js)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and 'status' in data and isinstance(data['status'], dict):
                code = data['status'].get('code')
                msg = data['status'].get('message')
                raise RuntimeError(f'CLOVA SSE error: {code} {msg}')
            msg = data.get('message') if isinstance(data, dict) else None
            if isinstance(msg, dict):
                c = msg.get('content')
                if isinstance(c, str) and c.strip():
                    response_text = c.strip()
            result = data.get('result') if isinstance(data, dict) else None
            if isinstance(result, dict):
                rmsg = result.get('message')
                if isinstance(rmsg, dict):
                    c = rmsg.get('content')
                    if isinstance(c, str) and c.strip():
                        response_text = c.strip()
    return response_text.strip()

def call_openai(text: str, *, model: str='gpt-5.2', max_tokens: int=16, temperature: float=0.0, system_prompt: str | None=None) -> str:
    from openai import OpenAI
    client = OpenAI()
    kwargs = dict(model=model, input=text, temperature=float(temperature), max_output_tokens=int(max_tokens))
    if system_prompt:
        kwargs['instructions'] = system_prompt
    resp = client.responses.create(**kwargs)
    return (resp.output_text or '').strip()
_GEMINI_CLIENT = None

def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        if not os.getenv('GOOGLE_CLOUD_PROJECT') and os.getenv('GCP_PROJECT'):
            os.environ['GOOGLE_CLOUD_PROJECT'] = os.getenv('GCP_PROJECT')
        if not os.getenv('GOOGLE_CLOUD_LOCATION') and os.getenv('GCP_LOCATION'):
            os.environ['GOOGLE_CLOUD_LOCATION'] = os.getenv('GCP_LOCATION')
        if not os.getenv('GOOGLE_GENAI_USE_VERTEXAI'):
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
        from google import genai
        _GEMINI_CLIENT = genai.Client()
    return _GEMINI_CLIENT

def _extract_gemini_text(resp) -> str:
    t = getattr(resp, 'text', None)
    if t and str(t).strip():
        return str(t).strip()
    cands = getattr(resp, 'candidates', None) or []
    if cands:
        try:
            content = getattr(cands[0], 'content', None)
            parts = getattr(content, 'parts', None) or []
            buf = [str(getattr(p, 'text', '')) for p in parts if getattr(p, 'text', None)]
            joined = ''.join(buf).strip()
            if joined:
                return joined
        except Exception:
            pass
    return ''

def call_gemini(text: str, *, model: str='gemini-3-pro-preview', max_tokens: int=256, temperature: float=0.0, thinking: str='low', system_prompt: str | None=None) -> str:
    from google.genai import types
    client = _get_gemini_client()
    token_plan = sorted(set([int(max_tokens), 512, 1024]))
    thinking_map = {
        'none': types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
        'minimal': types.ThinkingLevel.MINIMAL,
        'low': types.ThinkingLevel.LOW,
        'medium': types.ThinkingLevel.MEDIUM,
        'high': types.ThinkingLevel.HIGH,
    }
    thinking_level = thinking_map.get(str(thinking).lower(), types.ThinkingLevel.LOW)
    for mt in token_plan:
        cfg = types.GenerateContentConfig(system_instruction=system_prompt, temperature=float(temperature), max_output_tokens=int(mt), response_modalities=['TEXT'], thinking_config=types.ThinkingConfig(thinking_level=thinking_level))
        resp = client.models.generate_content(model=model, contents=text, config=cfg)
        raw = _extract_gemini_text(resp)
        if raw.strip():
            return raw.strip()
        time.sleep(0.4)
    return ''

def call_sonnet_vertex(text: str, *, model: str='claude-sonnet-4-5@20250929', max_tokens: int=16, temperature: float=0.0, system_prompt: str | None=None) -> str:
    import anthropic
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT')
    region = os.getenv('GOOGLE_CLOUD_LOCATION') or os.getenv('GCP_LOCATION') or 'global'
    if not project_id:
        raise RuntimeError('GOOGLE_CLOUD_PROJECT or GCP_PROJECT must be set for Claude on Vertex AI.')
    vertex_model = os.getenv('SONNET_VERTEX_MODEL') or model
    client = anthropic.AnthropicVertex(project_id=project_id, region=region)
    kwargs = dict(model=vertex_model, max_tokens=int(max_tokens), temperature=float(temperature), messages=[{'role': 'user', 'content': text}])
    if system_prompt:
        kwargs['system'] = system_prompt
    resp = client.messages.create(**kwargs)
    try:
        return ''.join((blk.text for blk in resp.content if hasattr(blk, 'text'))).strip()
    except Exception:
        return str(getattr(resp, 'content', '')).strip()

def call_sonnet(text: str, *, model: str='claude-sonnet-4-5', max_tokens: int=16, temperature: float=0.0, system_prompt: str | None=None) -> str:
    use_vertex = str(os.getenv('SONNET_VIA_VERTEX', '')).lower() in {'1', 'true', 'yes', 'y'}
    has_vertex_config = bool(os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT'))
    has_anthropic_key = bool(os.getenv('ANTHROPIC_API_KEY'))
    if use_vertex or (has_vertex_config and not has_anthropic_key):
        return call_sonnet_vertex(text, model=os.getenv('SONNET_VERTEX_MODEL') or 'claude-sonnet-4-5@20250929', max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt)
    import anthropic
    client = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=int(max_tokens), temperature=float(temperature), messages=[{'role': 'user', 'content': text}])
    if system_prompt:
        kwargs['system'] = system_prompt
    resp = client.messages.create(**kwargs)
    try:
        return ''.join((blk.text for blk in resp.content if hasattr(blk, 'text'))).strip()
    except Exception:
        return str(getattr(resp, 'content', '')).strip()
CALLER_MAP = {'clova': call_clova, 'openai': call_openai, 'gemini': call_gemini, 'sonnet': call_sonnet}

def call_model(target: str, text: str, **kwargs) -> str:
    fn = CALLER_MAP.get(target)
    if fn is None:
        raise ValueError(f'Unknown target: {target}. Choose from {list(CALLER_MAP)}')
    return fn(text, **kwargs)
