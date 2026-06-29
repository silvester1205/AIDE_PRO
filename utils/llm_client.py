"""
LLM Client for interacting with OpenAI-compatible APIs.
"""

import json
import re
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
    template_fields: List[Dict[str, str]],
    pdf_text: str
) -> Tuple[bool, Any]:
    """
    Send template fields and PDF text to LLM for analysis.

    Args:
        config: Dictionary with 'endpoint', 'api_key', 'model'
        template_fields: List of dicts with 'name', 'prompt', 'type', 'level'
        pdf_text: Extracted text from PDF (with page markers)

    Returns:
        Tuple of (success: bool, result: dict or error message)
    """
    import time
    t_start = time.time()

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['endpoint'],
            timeout=120.0
        )

        system_prompt = build_system_prompt(template_fields)

        # Truncate PDF to prevent context overflow
        max_chars = 60000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "\n\n[PDF truncated due to length]"

        # Build fields listing
        field_lines = []
        for i, f in enumerate(template_fields, 1):
            level_tag = f.get('level', 'study')
            field_lines.append(f"{i}. [{level_tag}] {f.get('prompt', f.get('name', ''))}")
        field_str = "\n".join(field_lines)

        # Build prompt with arm/study distinction
        study_fields = [f for f in template_fields if f.get('level', 'study') == 'study']
        arm_fields = [f for f in template_fields if f.get('level') == 'arm']

        instructions = []
        if study_fields:
            instructions.append(f"Study-level fields ({len(study_fields)} items): return a SINGLE value for each.")
        if arm_fields:
            instructions.append(
                f"Arm-level fields ({len(arm_fields)} items): "
                "MUST return an ARRAY (JSON list) of values, one per study arm. "
                "Additionally, for each arm field, ALSO return 'sources' (array of source texts, one per arm) "
                "and 'pages' (array of page numbers, one per arm). "
                "All arrays ('response', 'sources', 'pages') MUST have the SAME LENGTH (one entry per arm). "
                "If all arms share the same source/page, you may duplicate the same value across entries. "
                "Example: for 'Group N' with 3 arms, return: "
                '{{"response": [60, 58, 62], "sources": ["...", "...", "..."], "pages": [5, 6, 6]}}')

        # Build example from first study field and first arm field
        ex_items = []
        for f in template_fields[:2]:
            name_esc = f.get('name', f.get('prompt', '')).replace('"', "'")
            if f.get('level') == 'arm':
                ex_items.append(f'  {{"field": "{name_esc}", "response": ["value1", "value2"], "sources": ["source text for arm1", "source text for arm2"], "pages": [5, 5], "level": "arm"}}')
            else:
                ex_items.append(f'  {{"field": "{name_esc}", "response": "<answer>", "source": "<exact text from document>", "page": "<page number>", "level": "study"}}')
        example_json = '{\n"responses": [\n' + ',\n'.join(ex_items) + '\n]\n}'

        instruction_text = "\n".join(instructions)

        user_prompt = f"""Document Text:
---
{pdf_text}
---

Fields to extract:
{field_str}

{instruction_text}

CRITICAL: Your responses array MUST have exactly {len(template_fields)} items — one per field, in order.
For arm-level fields: "response" MUST be a JSON ARRAY (e.g., ["value1", "value2"]); also include "sources" (array of source texts) and "pages" (array of page numbers) — all same length.
For study-level fields: "response" MUST be a single string/value; include "source" (text) and "page" (number).
All arm-level arrays MUST have identical lengths (same number of arms).

Use this structure (copy and replace with your answers):
{example_json}

CRITICAL: Return ONLY the JSON object. No markdown, no code blocks, no extra text.
"""

        t_sent = time.time()

        content = None
        last_err = ""
        finish_reason = ""

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

        if finish_reason == "length":
            content += "\n\n[WARNING: Response may be truncated — max_tokens limit reached]"

        result = _parse_json(content)
        if result is None:
            return False, (
                f"Failed to parse JSON response.\n\n"
                f"Raw (first 500 chars):\n{content[:500]}"
            )

        formatted = format_result(template_fields, result)
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


def build_system_prompt(template_fields_or_prompts) -> str:
    """Build system prompt for data extraction from systematic review documents.

    Accepts either a list of template field dicts (new format) or a list of prompt strings (backward compat).
    """
    return """You are a data extraction assistant for systematic reviews and meta-analysis.

For each field you MUST provide:
1. "response": The extracted answer.
   - For study-level fields: a single value (string or number).
   - For arm-level fields: a JSON ARRAY of values, one per study arm. Example: ["arm1_value", "arm2_value", "arm3_value"].
2. "source" / "page": For study-level fields, provide a single source text and page number.
   For arm-level fields, provide "sources" (array of source texts) and "pages" (array of page numbers), one per arm.
3. "level": Either "study" or "arm" matching the field definition.

IMPORTANT: Arm-level fields MUST have "response", "sources", and "pages" all as arrays of the same length.
Study-level fields MUST have "response" as a single value, with "source" and "page" as single values.

Return ONLY valid JSON using this structure:
{"responses": [{"field": "...", "response": "...", "source": "...", "page": "...", "level": "..."}]}

For arm fields, use: {"field": "...", "response": ["v1","v2"], "sources": ["src1","src2"], "pages": [1,2], "level": "arm"}

No markdown, no code blocks, no extra text outside the JSON."""


def format_result(template_fields_or_prompts, raw_result: Dict) -> Dict[str, Dict]:
    """
    Format the LLM result into a standardized structure.

    Args:
        template_fields_or_prompts: Either list of template field dicts (new) or list of prompt strings (old)
        raw_result: Raw JSON result from LLM

    Returns:
        Formatted dictionary with field_N keys
    """
    formatted = {}

    # Detect if old format (list of strings) or new format (list of dicts)
    if template_fields_or_prompts and isinstance(template_fields_or_prompts[0], str):
        # Old format: list of prompt strings
        prompts = template_fields_or_prompts
        return _format_result_old(prompts, raw_result)

    # New format: list of template field dicts
    fields = template_fields_or_prompts

    # Backward compat: handle old field_N format
    if isinstance(raw_result, dict) and not raw_result.get('responses'):
        if any(k.startswith('field_') for k in raw_result):
            for i, f in enumerate(fields):
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
                        'prompt': f.get('prompt', f.get('name', '')),
                        'name': f.get('name', ''),
                        'type': f.get('type', 'text'),
                        'level': f.get('level', 'study'),
                        'response': field_data.get('response', ''),
                        'source_quote': field_data.get('source_quote', '') or field_data.get('source', ''),
                        'source_page': sp,
                        'source': field_data.get('source', ''),
                        'arm_sources': [],
                        'arm_pages': [],
                    }
                    if f.get('level') == 'arm':
                        resp = formatted[field_key]['response']
                        if not isinstance(resp, list):
                            formatted[field_key]['response'] = [str(resp)]
            return formatted

    # Normalize: list → {"responses": list}
    if isinstance(raw_result, list):
        raw_result = {"responses": raw_result}

    responses = raw_result.get('responses', []) if isinstance(raw_result, dict) else []
    if not isinstance(responses, list):
        responses = []

    for i, f in enumerate(fields):
        field_key = f"field_{i+1}"
        if i < len(responses) and isinstance(responses[i], dict):
            item = responses[i]
            sp = item.get('page') or item.get('source_page')
            if sp is not None:
                try:
                    sp = int(float(str(sp).strip()))
                except (ValueError, TypeError):
                    sp = None
            formatted[field_key] = {
                'prompt': f.get('prompt', f.get('name', '')),
                'name': f.get('name', ''),
                'type': f.get('type', 'text'),
                'level': f.get('level', 'study'),
                'response': item.get('response', ''),
                'source_quote': item.get('source', '') or item.get('source_quote', ''),
                'source_page': sp,
                'source': item.get('source', ''),
                'arm_sources': item.get('sources', []) if isinstance(item.get('sources'), list) else [],
                'arm_pages': item.get('pages', []) if isinstance(item.get('pages'), list) else [],
            }
            # Normalize arm-level responses
            if f.get('level') == 'arm':
                resp = formatted[field_key]['response']
                if not isinstance(resp, list):
                    formatted[field_key]['response'] = [str(resp)]
        else:
            formatted[field_key] = {
                'prompt': f.get('prompt', f.get('name', '')),
                'name': f.get('name', ''),
                'type': f.get('type', 'text'),
                'level': f.get('level', 'study'),
                'response': 'No response',
                'source_quote': '',
                'source_page': None,
                'source': '',
                'arm_sources': [],
                'arm_pages': [],
            }

    return formatted



# ── Base templates ──

_BASE_RCT = [
    {"name": "Author+Year", "prompt": "First author last name and publication year. Format: 'Smith 2023'.", "type": "text", "level": "study"},
    {"name": "Registration", "prompt": "Trial registration number (e.g., NCT01234567). If not reported, write 'Not reported'.", "type": "text", "level": "study"},
    {"name": "Total_Sample_Size", "prompt": "Total number of participants enrolled in the study.", "type": "integer", "level": "study"},
    {"name": "Group_Label", "prompt": "Name/label of each study arm.", "type": "text", "level": "arm"},
    {"name": "Group_N", "prompt": "Number of participants in this arm (sample size per group).", "type": "integer", "level": "arm"},
    {"name": "Age", "prompt": "Age per arm. Format: mean±SD (or median [IQR]).", "type": "text", "level": "arm"},
    {"name": "Sex_Female%", "prompt": "Female percentage per arm. Format: N (%). Example: '132 (55%)'.", "type": "text", "level": "arm"},
    {"name": "Intervention", "prompt": "Detailed description of intervention for this arm: drug, dose, frequency, route, duration.", "type": "text", "level": "arm"},
    {"name": "Outcome_1", "prompt": "Replace 'Outcome_1' with actual outcome name. Extract ONLY this arm's value. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "Outcome_2", "prompt": "Replace 'Outcome_2' with actual outcome name. Extract ONLY this arm's value. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "Outcome_3", "prompt": "Replace 'Outcome_3' with actual outcome name. Extract ONLY this arm's value. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "D1_Randomization", "prompt": "ROB2 Domain 1 — Risk of bias arising from randomization process. Assess: random sequence generation (was the allocation sequence random?), allocation concealment (was the allocation sequence concealed until participants were enrolled?), baseline differences between groups. Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
    {"name": "D2_Deviations", "prompt": "ROB2 Domain 2 — Risk of bias due to deviations from intended interventions (effect of assignment). Assess: blinding of participants and personnel, whether the analysis was appropriate to estimate the effect of assignment (ITT analysis), deviations from the intended intervention that arose because of the trial context. Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
    {"name": "D3_MissingData", "prompt": "ROB2 Domain 3 — Risk of bias due to missing outcome data. Assess: total dropout rate per group, reasons for dropout, whether missing data were balanced across groups, how missing data were handled in the analysis (e.g., LOCF, multiple imputation, complete case). Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
    {"name": "D4_Measurement", "prompt": "ROB2 Domain 4 — Risk of bias in measurement of the outcome. Assess: whether outcome assessors were blinded, whether the outcome measurement method was appropriate, whether the measurement differed between groups, whether knowledge of intervention status could have influenced outcome assessment. Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
    {"name": "D5_Selection", "prompt": "ROB2 Domain 5 — Risk of bias in selection of the reported result. Assess: whether the trial was pre-registered, whether the analysis plan was pre-specified, whether results were reported according to the pre-specified plan, evidence of selective reporting (e.g., outcomes measured but not reported, reporting only favorable results). Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
    {"name": "Overall_ROB", "prompt": "Overall ROB2 judgment based on Domains 1-5 ONLY. Rules: (1) If ANY domain is High risk → Overall = High risk. (2) If NO domain is High risk but ANY domain is Some concerns → Overall = Some concerns. (3) If ALL 5 domains are Low risk → Overall = Low risk. Do NOT upgrade to High risk just because multiple domains have Some concerns — multiple Some concerns still results in Some concerns, NOT High risk. Valid values: Low risk / Some concerns / High risk", "type": "categorical", "level": "study"},
]

_BASE_COHORT = [
    {"name": "Author+Year", "prompt": "First author last name and publication year. Format: 'Smith 2023'.", "type": "text", "level": "study"},
    {"name": "Study_Design", "prompt": "Specific study design (e.g., 'retrospective cohort', 'prospective cohort').", "type": "text", "level": "study"},
    {"name": "Total_Sample_Size", "prompt": "Total number of participants.", "type": "integer", "level": "study"},
    {"name": "Group_Label", "prompt": "Name/label of each group (e.g., 'Exposed', 'Unexposed').", "type": "text", "level": "arm"},
    {"name": "Group_N", "prompt": "Number of participants in this group.", "type": "integer", "level": "arm"},
    {"name": "Age", "prompt": "Age for this group. Format: mean±SD (or median [IQR]).", "type": "text", "level": "arm"},
    {"name": "Sex_Female%", "prompt": "Female percentage for this group. Format: N (%). Example: '132 (55%)'.", "type": "text", "level": "arm"},
    {"name": "Follow_up_duration", "prompt": "Duration of follow-up.", "type": "text", "level": "study"},
    {"name": "Effect_Estimate", "prompt": "Adjusted effect estimate (HR/OR/RR) with 95% CI. Format: 'HR=1.25 (95%CI 1.10-1.42)'.", "type": "text", "level": "study"},
    {"name": "Confounders_Adjusted", "prompt": "List of confounders adjusted for in the analysis.", "type": "text", "level": "study"},
    {"name": "Outcome_1", "prompt": "Replace with actual outcome name. Extract ONLY this group's result. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "Outcome_2", "prompt": "Replace with actual outcome name. Extract ONLY this group's result. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "Outcome_3", "prompt": "Replace with actual outcome name. Extract ONLY this group's result. Format: 'value (mean±SD or N(%))'.", "type": "text", "level": "arm"},
    {"name": "NOS_Selection", "prompt": "NOS Selection domain (max 4 stars). Assess: (1) Representativeness of the exposed cohort — truly representative (★) / somewhat representative (★) / selected group / no description; (2) Selection of the non-exposed cohort — drawn from same community (★) / different source / no description; (3) Ascertainment of exposure — secure record (★) / structured interview (★) / self report / no description; (4) Demonstration outcome not present at start — yes (★) / no. One star per item.", "type": "categorical", "level": "study"},
    {"name": "NOS_Comparability", "prompt": "NOS Comparability domain (max 2 stars). Assess: ★ Study controls for the most important confounding factor; ★ Study controls for any additional factor (list them). If cohorts are not comparable on design or analysis → 0 stars.", "type": "categorical", "level": "study"},
    {"name": "NOS_Outcome", "prompt": "NOS Outcome domain (max 3 stars). Assess: (1) Assessment of outcome — independent blind assessment (★) / record linkage (★) / self report / no description; (2) Follow-up long enough for outcomes to occur — yes (★) / no; (3) Adequacy of follow-up — complete follow-up all subjects accounted (★) / loss ≤20% unlikely to introduce bias (★) / follow-up <80% no description / no statement. One star per item.", "type": "categorical", "level": "study"},
    {"name": "NOS_Total", "prompt": "NOS quality rating based on domain stars (AHRQ standards): Good = Selection 3-4 stars AND Comparability 1-2 stars AND Outcome 2-3 stars; Fair = Selection 2 stars AND Comparability 1-2 stars AND Outcome 2-3 stars; Poor = Selection 0-1 stars OR Comparability 0 stars OR Outcome 0-1 stars. Provide the star counts per domain and the quality rating.", "type": "categorical", "level": "study"},
]

_BASE_DIAGNOSTIC = [
    {"name": "Author+Year", "prompt": "First author last name and publication year. Format: 'Smith 2023'.", "type": "text", "level": "study"},
    {"name": "Sample_Size", "prompt": "Total number of participants/eyes/lesions.", "type": "integer", "level": "study"},
    {"name": "Setting", "prompt": "Study setting and population.", "type": "text", "level": "study"},
    {"name": "Index_Test", "prompt": "Index test name and details.", "type": "text", "level": "study"},
    {"name": "Reference_Standard", "prompt": "Reference standard used.", "type": "text", "level": "study"},
    {"name": "TP", "prompt": "True positives.", "type": "integer", "level": "study"},
    {"name": "FP", "prompt": "False positives.", "type": "integer", "level": "study"},
    {"name": "FN", "prompt": "False negatives.", "type": "integer", "level": "study"},
    {"name": "TN", "prompt": "True negatives.", "type": "integer", "level": "study"},
    {"name": "Sensitivity", "prompt": "Sensitivity with 95%CI.", "type": "text", "level": "study"},
    {"name": "Specificity", "prompt": "Specificity with 95%CI.", "type": "text", "level": "study"},
    {"name": "AUC", "prompt": "Area under the ROC curve with 95%CI.", "type": "text", "level": "study"},
    {"name": "QUADAS_PatientSelection", "prompt": "QUADAS-2 Domain 1 — Patient Selection. Signalling questions: (1.1) Was a consecutive or random sample of patients enrolled? (1.2) Was a case-control design avoided? (1.3) Did the study avoid inappropriate exclusions? Risk of Bias: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_Applicability_PatientSelection", "prompt": "QUADAS-2 Domain 1 — Applicability Concerns for Patient Selection. Does the study population match the review question in terms of setting, target condition, and inclusion criteria? Valid values: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_IndexTest", "prompt": "QUADAS-2 Domain 2 — Index Test. Signalling questions: (2.1) Were the index test results interpreted without knowledge of the reference standard results? (2.2) Did the study prespecify the threshold for a positive result? Risk of Bias: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_Applicability_IndexTest", "prompt": "QUADAS-2 Domain 2 — Applicability Concerns for Index Test. Does the index test, its conduct, and interpretation match the review question? Valid values: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_ReferenceStandard", "prompt": "QUADAS-2 Domain 3 — Reference Standard. Signalling questions: (3.1) Is the reference standard likely to correctly classify the target condition? (3.2) Were the reference standard results interpreted without knowledge of the index test results? Risk of Bias: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_Applicability_ReferenceStandard", "prompt": "QUADAS-2 Domain 3 — Applicability Concerns for Reference Standard. Does the reference standard and its conduct match the review question? Valid values: Low / High / Unclear", "type": "categorical", "level": "study"},
    {"name": "QUADAS_FlowTiming", "prompt": "QUADAS-2 Domain 4 — Flow and Timing. Signalling questions: (4.1) Was there an appropriate interval between index test and reference standard? (4.2) Did all patients receive the same reference standard? (4.3) Were all patients included in the analysis? Risk of Bias: Low / High / Unclear. (Applicability not assessed for Domain 4)", "type": "categorical", "level": "study"},
    {"name": "Overall_QUADAS", "prompt": "Overall QUADAS-2 risk of bias judgment based on Domains 1-4. Rules: (1) If ANY domain is High → Overall = High. (2) If NO domain is High but ANY domain is Unclear → Overall = Unclear. (3) If ALL 4 domains are Low → Overall = Low. Valid values: Low / High / Unclear", "type": "categorical", "level": "study"},
]

_BASE_CASE_CONTROL = [
    {"name": "Author+Year", "prompt": "First author last name and publication year. Format: 'Smith 2023'.", "type": "text", "level": "study"},
    {"name": "Total_Sample_Size", "prompt": "Total number of participants (cases + controls).", "type": "integer", "level": "study"},
    {"name": "Case_Definition", "prompt": "Case definition: how were cases identified and diagnosed?", "type": "text", "level": "study"},
    {"name": "Control_Source", "prompt": "Source of controls (e.g., hospital-based, population-based, neighborhood).", "type": "text", "level": "study"},
    {"name": "Exposure_Measurement", "prompt": "How was the exposure/risk factor measured? (e.g., medical records, interview, questionnaire).", "type": "text", "level": "study"},
    {"name": "Exposed_Cases", "prompt": "Number of cases with the exposure. Format: 'N (%)'.", "type": "text", "level": "study"},
    {"name": "Exposed_Controls", "prompt": "Number of controls with the exposure. Format: 'N (%)'.", "type": "text", "level": "study"},
    {"name": "Odds_Ratio", "prompt": "Odds ratio (OR) with 95% CI. If adjusted, specify. Format: 'OR=2.5 (95%CI 1.8-3.5)'.", "type": "text", "level": "study"},
    {"name": "Confounders_Adjusted", "prompt": "List of confounders adjusted for in the analysis.", "type": "text", "level": "study"},
    {"name": "NOS_Selection", "prompt": "NOS Selection domain for case-control (max 4 stars). Assess: (1) Adequacy of case definition — yes (★) / no; (2) Representativeness of cases — consecutive or obviously representative series (★) / potential selection bias / no description; (3) Selection of controls — community controls (★) / hospital controls (★) / no description; (4) Definition of controls — no history of disease (★) / no description. One star per item.", "type": "categorical", "level": "study"},
    {"name": "NOS_Comparability", "prompt": "NOS Comparability domain (max 2 stars). Assess: ★ Study controls for the most important confounding factor; ★ Study controls for any additional factor (list them). If cases and controls are not comparable → 0 stars.", "type": "categorical", "level": "study"},
    {"name": "NOS_Exposure", "prompt": "NOS Exposure domain (max 3 stars). Assess: (1) Ascertainment of exposure — secure record (★) / structured interview blind to case/control status (★) / interview not blinded / self report or medical record only / no description; (2) Same method of ascertainment for cases and controls — yes (★) / no; (3) Non-response rate — same rate for both groups (★) / non-respondents described / rate different and no designation. One star per item.", "type": "categorical", "level": "study"},
    {"name": "NOS_Total", "prompt": "NOS quality rating based on domain stars (AHRQ standards): Good = Selection 3-4 stars AND Comparability 1-2 stars AND Exposure 2-3 stars; Fair = Selection 2 stars AND Comparability 1-2 stars AND Exposure 2-3 stars; Poor = Selection 0-1 stars OR Comparability 0 stars OR Exposure 0-1 stars. Provide the star counts per domain and the quality rating.", "type": "categorical", "level": "study"},
]


def generate_template(config: Dict[str, str], topic: str, prompt_count: int = 0) -> Tuple[bool, Any]:
    """
    Use LLM to refine a base template based on the research topic.

    First identifies study type, selects appropriate base template,
    then asks LLM to customize it for the specific topic.

    Args:
        config: Dictionary with 'endpoint', 'api_key', 'model'
        topic: Research topic or abstract text

    Returns:
        Tuple of (success, result) where result is a list of field dicts or error message
    """
    import time
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['endpoint'],
            timeout=60.0
        )

        # Step 1: Identify study type
        study_type_prompt = f"""Read this research topic/abstract and determine the study design.
Return ONLY one word: RCT / Cohort / CaseControl / Diagnostic / CrossSectional / CaseSeries / Other
For retrospective case-control studies, return 'CaseControl'. For retrospective cohort, return 'Cohort'.

{topic[:500]}"""

        study_type = None
        raw = ""
        try:
            st_resp = client.chat.completions.create(
                model=config['model'],
                messages=[{"role": "user", "content": study_type_prompt}],
                temperature=0, max_tokens=4096, timeout=30,
                extra_body={"reasoning_effort": "low"},
            )
            raw = (st_resp.choices[0].message.content or "").strip()
            print(f"[AIDE DEBUG] study_type raw='{raw}' finish_reason={st_resp.choices[0].finish_reason}")
            raw_lower = raw.lower()
            study_map = {"rct": "RCT", "cohort": "Cohort", "casecontrol": "CaseControl",
                         "diagnostic": "Diagnostic", "crosssectional": "CrossSectional",
                         "caseseries": "CaseSeries"}
            for t in study_map:
                if t in raw_lower:
                    study_type = study_map[t]
                    break
        except Exception as e:
            return False, f"Failed to identify study type: {e}"

        if not study_type:
            return False, f"Could not identify study type. LLM returned: '{raw}'"

        # Step 2: Select base template
        if study_type == "RCT":
            base_fields = _BASE_RCT
        elif study_type == "Diagnostic":
            base_fields = _BASE_DIAGNOSTIC
        elif study_type == "CaseControl":
            base_fields = _BASE_CASE_CONTROL
        else:  # Cohort, CrossSectional, CaseSeries, Other
            base_fields = _BASE_COHORT

        # Step 3: Ask LLM to customize the base template
        base_json = json.dumps(base_fields, ensure_ascii=False, indent=2)
        system_msg = "You are an expert in systematic review methodology. Customize extraction templates for specific studies."

        user_msg = f"""Research topic / abstract:
---
{topic}
---

Study type: {study_type}

Below is a base extraction template for this study type. Customize it for the SPECIFIC topic:

{base_json}

### Instructions:
1. Use the **PICOS framework** to identify extraction fields: Population, Intervention, Comparison, Outcomes, Study design
2. Keep ONLY fields that are relevant to THIS specific study
3. For **outcomes**: REPLACE generic outcome fields with one field PER outcome. E.g., instead of "Outcome_1", create "FEV1" (arm), "Exacerbations" (arm), "Mortality" (arm) — each as a separate arm-level field
4. ADD disease-specific baseline measures as separate fields (e.g., for COPD: add FEV1, FVC, SGRQ; for diabetes: add HbA1c, fasting glucose)
5. Adjust prompts to reference the specific interventions and outcomes of THIS study
6. Reorder fields: basic info → baseline → outcomes → quality assessment
7. DO NOT change existing field names that are still relevant
8. Quality assessment fields (ROB2/NOS/QUADAS) MUST be kept as-is, they are the core evaluation tool
9. You can change the "level" of any field (except quality assessment) if appropriate for this study
10. Total fields: 12-25 depending on study complexity

Return ONLY valid JSON:
{{"fields": [
  {{"name": "...", "prompt": "...", "type": "text|integer|float|categorical", "level": "study|arm"}},
  ...
]}}

No markdown, no code blocks, no extra text.
"""

        response = client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=8192,
            timeout=60,
        )

        content = response.choices[0].message.content or ""
        if not content:
            return False, "Empty response"

        # Strip code fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', content)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        fields = result.get('fields', []) if isinstance(result, dict) else result

        if not fields or not isinstance(fields, list):
            return False, "No fields generated"

        return True, fields

    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}\n\nRaw:\n{content[:500] if 'content' in dir() else 'N/A'}"
    except Exception as e:
        return False, f"API error: {str(e)}"


def _format_result_old(prompts: list, raw_result: Dict) -> Dict[str, Dict]:
    """Old format result formatter for backward compatibility."""
    formatted = {}

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
                        'name': prompt,
                        'type': 'text',
                        'level': 'study',
                        'response': field_data.get('response', ''),
                        'source_quote': field_data.get('source_quote', '') or field_data.get('source', ''),
                        'source_page': sp,
                        'source': field_data.get('source', '')
                    }
                else:
                    formatted[field_key] = {
                        'prompt': prompt, 'name': prompt, 'type': 'text', 'level': 'study',
                        'response': 'No response', 'source_quote': '', 'source_page': None, 'source': ''
                    }
            return formatted

    if isinstance(raw_result, list):
        raw_result = {"responses": raw_result}

    responses = raw_result.get('responses', []) if isinstance(raw_result, dict) else []
    if not isinstance(responses, list):
        responses = []

    for i, prompt in enumerate(prompts):
        field_key = f"field_{i+1}"
        if i < len(responses) and isinstance(responses[i], dict):
            item = responses[i]
            sp = item.get('page') or item.get('source_page')
            if sp is not None:
                try:
                    sp = int(float(str(sp).strip()))
                except (ValueError, TypeError):
                    sp = None
            formatted[field_key] = {
                'prompt': prompt, 'name': prompt, 'type': 'text', 'level': 'study',
                'response': item.get('response', ''),
                'source_quote': item.get('source', '') or item.get('source_quote', ''),
                'source_page': sp, 'source': item.get('source', '')
            }
        else:
            formatted[field_key] = {
                'prompt': prompt, 'name': prompt, 'type': 'text', 'level': 'study',
                'response': 'No response', 'source_quote': '', 'source_page': None, 'source': ''
            }

    return formatted
