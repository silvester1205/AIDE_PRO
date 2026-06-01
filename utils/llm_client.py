"""
LLM Client for interacting with OpenAI-compatible APIs.
"""

import json
from typing import Dict, List, Tuple, Any, Optional


def fetch_available_models(config: Dict[str, str]) -> Tuple[bool, List[str], str]:
    """
    Fetch available models from the API endpoint.

    Args:
        config: Dictionary with 'endpoint', 'api_key'

    Returns:
        Tuple of (success: bool, models: list, message: str)
    """
    import requests

    endpoint = config.get('endpoint', '').rstrip('/')
    api_key = config.get('api_key', '')

    if not endpoint or not api_key:
        return False, [], "请先填写 API Endpoint 和 API Key"

    # 尝试多个可能的 models 端点路径
    possible_urls = [
        f"{endpoint}/models",
        f"{endpoint.replace('/v1', '')}/models",
        f"{endpoint.replace('/v1/chat/completions', '')}/models",
        f"{endpoint.replace('/chat/completions', '')}/models",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    last_error = ""

    for url in set(possible_urls):  # 去重
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # 尝试不同的响应格式
                models = []

                # OpenAI / OpenRouter 格式
                if 'data' in data and isinstance(data['data'], list):
                    for item in data['data']:
                        if isinstance(item, dict) and 'id' in item:
                            models.append(item['id'])
                        elif isinstance(item, str):
                            models.append(item)

                # 直接是数组的格式
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'id' in item:
                            models.append(item['id'])
                        elif isinstance(item, str):
                            models.append(item)

                # 对象格式 {model_id: {...}}
                elif isinstance(data, dict) and 'data' not in data:
                    for key in data.keys():
                        if not key.startswith('_'):
                            models.append(key)

                if models:
                    # 识别免费模型并排序（免费模型排在前面）
                    models = sort_models_by_free_priority(models)
                    return True, models, f"成功获取 {len(models)} 个模型（免费模型已置顶）"

                return False, [], "API 返回了空模型列表"

            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"

        except requests.exceptions.Timeout:
            last_error = "请求超时"
        except requests.exceptions.ConnectionError:
            last_error = "连接失败"
        except Exception as e:
            last_error = str(e)

    return False, [], f"无法获取模型列表: {last_error}"


def sort_models_by_free_priority(models: List[str]) -> List[str]:
    """
    排序模型列表，将免费模型排在前面。

    免费模型的识别规则：
    - 模型名称包含 ':free' 后缀（OpenRouter 等）
    - 模型名称包含 'free' 关键词
    - 常见免费模型名称模式

    Args:
        models: 原始模型列表

    Returns:
        排序后的列表（免费模型在前）
    """
    # 去重
    models = list(set(models))

    # 免费模型标识
    free_indicators = [
        ':free',
        '/free',
        '-free',
        '.free',
    ]

    # 常见免费模型前缀/名称
    known_free_models = [
        'google/gemini-2.0-flash-exp',
        'google/gemini-exp',
        'meta-llama/llama-3.2',
        'meta-llama/llama-3.1-8b',
        'meta-llama/llama-3-8b',
        'mistralai/mistral-7b',
        'nousresearch/hermes-3',
        'huggingfaceh4/zephyr-7b',
        'microsoft/phi-3',
        'openchat/openchat',
        'gryphe/mythomax-l2',
        'undi95/toppy-m',
        'pygmalionai/mythalion',
    ]

    def is_free_model(model_id: str) -> bool:
        """判断是否为免费模型"""
        model_lower = model_id.lower()

        # 检查免费标识
        for indicator in free_indicators:
            if indicator in model_lower:
                return True

        # 检查已知免费模型
        for free_prefix in known_free_models:
            if model_lower.startswith(free_prefix.lower()):
                return True

        return False

    # 分离免费和付费模型
    free_models = []
    paid_models = []

    for model in models:
        if is_free_model(model):
            free_models.append(model)
        else:
            paid_models.append(model)

    # 各自排序
    free_models.sort()
    paid_models.sort()

    # 免费模型在前，付费在后
    return free_models + paid_models


def test_connection(config: Dict[str, str]) -> Tuple[bool, str]:
    """
    Test the API connection with a simple request.

    Args:
        config: Dictionary with 'endpoint', 'api_key', 'model'

    Returns:
        Tuple of (success: bool, message: str)
    """
    import time

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['endpoint']
        )

        # 开始计时
        start_time = time.time()

        # Make a simple request to test connection
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000

        return True, f"✅ 连接成功！延迟: {latency_ms:.0f}ms | 模型: {config['model']}"

    except Exception as e:
        return False, f"❌ 连接失败: {str(e)}"


def analyze_with_llm(
    config: Dict[str, str],
    prompts: List[str],
    pdf_text: str
) -> Tuple[bool, Any]:
    """
    Send prompts and PDF text to LLM for analysis.

    Args:
        config: Dictionary with 'endpoint', 'api_key', 'model'
        prompts: List of prompts from the coding form
        pdf_text: Extracted text from PDF (with page markers)

    Returns:
        Tuple of (success: bool, result: dict or error message)
    """
    import re
    import time
    t_start = time.time()

    try:
        from openai import OpenAI

        # Timeout: 10s connect, 120s read
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['endpoint'],
            timeout=120.0
        )

        system_prompt = build_system_prompt(prompts)

        # Truncate PDF to prevent context overflow (DeepSeek: 128K, safe limit ~60K chars)
        max_chars = 60000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[PDF truncated due to length]"

        prompt_lines = []
        for i, p in enumerate(prompts, 1):
            prompt_lines.append(f"{i}. {p}")
        prompt_str = "\n".join(prompt_lines)

        # Build format example from first 2 prompts (like web version)
        ex_count = min(2, len(prompts))
        ex_items = []
        for j in range(ex_count):
            p_esc = prompts[j].replace('"', "'")
            ex_items.append(f'  {{"prompt": "{p_esc}", "response": "<answer>", "source": "<exact text from document>", "page": "<page number>"}}')
        example_json = '{\n"responses": [\n' + ',\n'.join(ex_items) + '\n]\n}'

        user_prompt = f"""Document Text:
---
{pdf_text}
---

Prompts to answer:
{prompt_str}

Extract information for each prompt and return ONLY valid JSON.

There are {len(prompts)} prompts — your responses array MUST have exactly {len(prompts)} items, one per prompt.

Use this structure (copy and replace with your answers):
{example_json}

CRITICAL: Return ONLY the JSON object. No markdown, no code blocks, no extra text.
"""

        t_sent = time.time()

        content = None
        last_err = ""
        finish_reason = ""

        # Single attempt, no response_format (works across all models)
        try:
            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=16384,
                timeout=120,
            )
            if response.choices:
                msg = response.choices[0].message
                content = msg.content or ""
                finish_reason = response.choices[0].finish_reason or ""
        except Exception as e:
            last_err = str(e)

        t_received = time.time()

        if not content:
            detail = f"finish_reason={finish_reason}" if finish_reason else ""
            if last_err:
                detail = f"error={last_err}"
            return False, (
                f"Empty response from LLM.\n"
                f"Model: {config['model']}\n"
                f"Details: {detail}\n"
                f"PDF length: {len(pdf_text)} chars"
            )

        # Warn if output was truncated
        if finish_reason == "length":
            content += "\n\n[WARNING: Response may be truncated — max_tokens limit reached]"

        # Parse JSON
        result = _parse_json(content)
        if result is None:
            return False, (
                f"Failed to parse JSON response.\n\n"
                f"Raw (first 500 chars):\n{content[:500]}"
            )

        formatted = format_result(prompts, result)
        t_done = time.time()
        formatted['_timing'] = {
            'build': round(t_sent - t_start, 1),
            'llm': round(t_received - t_sent, 1),
            'parse': round(t_done - t_received, 1),
            'total': round(t_done - t_start, 1),
            'input_chars': len(pdf_text),
            'output_chars': len(content),
        }
        return True, formatted

    except Exception as e:
        return False, f"API error: {str(e)}"


def _parse_json(content: str):
    """Parse LLM JSON response. Handles ```json fences and text-wrapped JSON.

    Simple approach: strip fences → JSON.parse → balanced brace fallback.
    """
    import re as _re
    content = content.strip()

    # Strip markdown code fences
    cleaned = _re.sub(r'^```(?:json)?\s*', '', content)
    cleaned = _re.sub(r'\s*```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return {"responses": result}
        return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: balanced brace extraction (handles text before/after JSON)
    start = cleaned.find('{')
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(cleaned[start:i+1])
                        if isinstance(result, list):
                            return {"responses": result}
                        return result
                    except json.JSONDecodeError:
                        return None

    return None


def build_system_prompt(prompts: List[str]) -> str:
    """Build system prompt for data extraction from systematic review documents."""
    return """You are a data extraction assistant for systematic reviews and meta-analysis.

For each field you MUST provide:
1. "response": The extracted answer — include ALL relevant detail (risk judgment, numerical values, text findings).
2. "source": The EXACT original text snippet from the document that supports your answer — quote the relevant sentence, paragraph, or table row.
3. "page": The specific page number where this information was found.

Return ONLY valid JSON using this structure:
{"responses": [{"prompt": "...", "response": "...", "source": "...", "page": "..."}]}

No markdown, no code blocks, no extra text outside the JSON."""


def format_result(prompts: List[str], raw_result: Dict) -> Dict[str, Dict[str, str]]:
    """
    Format the LLM result into a standardized structure.

    Maps from array format (like web version) to field_N dict for downstream compat.

    Args:
        prompts: Original list of prompts
        raw_result: Raw JSON result from LLM

    Returns:
        Formatted dictionary with keys: prompt, response, source_quote, source_page
    """
    formatted = {}

    # Backward compat: handle old field_N format
    if isinstance(raw_result, dict) and not raw_result.get('responses'):
        if any(k.startswith('field_') for k in raw_result):
            for i, prompt in enumerate(prompts):
                field_key = f"field_{i+1}"
                field_data = raw_result.get(field_key, {})
                if isinstance(field_data, dict):
                    sp = field_data.get('source_page') or field_data.get('page')
                    if sp is not None:
                        try:
                            sp = int(float(sp)) if not isinstance(sp, (int, float)) else int(sp)
                        except (ValueError, TypeError):
                            sp = None
                    formatted[field_key] = {
                        'prompt': prompt,
                        'response': field_data.get('response', ''),
                        'source_quote': field_data.get('source_quote', '') or field_data.get('source', ''),
                        'source_page': sp,
                        'source': field_data.get('source', '')
                    }
                else:
                    formatted[field_key] = {
                        'prompt': prompt,
                        'response': 'No response',
                        'source_quote': '',
                        'source_page': None,
                        'source': ''
                    }
            return formatted

    # Normalize: list → {"responses": list}
    if isinstance(raw_result, list):
        raw_result = {"responses": raw_result}

    responses = raw_result.get('responses', []) if isinstance(raw_result, dict) else []
    if not isinstance(responses, list):
        responses = []

    for i, prompt in enumerate(prompts):
        field_key = f"field_{i+1}"
        if i < len(responses) and isinstance(responses[i], dict):
            item = responses[i]
            # Normalize source_page from 'page' or 'source_page' field
            sp = item.get('page') or item.get('source_page')
            if sp is not None:
                try:
                    sp = int(float(str(sp).strip()))
                except (ValueError, TypeError):
                    sp = None
            formatted[field_key] = {
                'prompt': prompt,
                'response': item.get('response', ''),
                'source_quote': item.get('source', '') or item.get('source_quote', ''),
                'source_page': sp,
                'source': item.get('source', '')
            }
        else:
            formatted[field_key] = {
                'prompt': prompt,
                'response': 'No response',
                'source_quote': '',
                'source_page': None,
                'source': ''
            }

    return formatted
