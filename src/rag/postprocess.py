import json
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ParsedResponse:
    text: str
    result: str
    reasoning: Optional[str]
    has_finding: Optional[bool] = None
    organ: Optional[str] = None
    modality: Optional[str] = None


def parse_llm_response(content: str, expect_json: bool) -> ParsedResponse:
    content = content.strip()
    if not expect_json:
        return ParsedResponse(text=content, result=content, reasoning=None)

    def _strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```json"):
            t = t[len("```json"):]
        elif t.startswith("```"):
            t = t[len("```"):]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

    def _extract_json_substring(text: str) -> Optional[str]:
        start = -1
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return text[start:i+1]
        return None

    def _coerce_json_like(s: str) -> str:
        s2 = s
        # Replace single-quoted strings with double quotes cautiously
        s2 = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', s2)
        # Ensure keys are double-quoted
        s2 = re.sub(r"(\{|,)(\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*):", r"\1\2""\3""\4:", s2)
        # Python booleans/None -> JSON
        s2 = re.sub(r"\bTrue\b", "true", s2)
        s2 = re.sub(r"\bFalse\b", "false", s2)
        s2 = re.sub(r"\bNone\b", "null", s2)
        # Remove trailing commas
        s2 = re.sub(r",\s*([}\]])", r"\1", s2)
        return s2

    raw = _strip_code_fences(content)

    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        block = _extract_json_substring(raw)
        if block is None:
            block = _strip_code_fences(content)
        coerced = _coerce_json_like(block)
        try:
            obj = json.loads(coerced)
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to parse JSON from LLM response; returning raw text. Snippet: %s",
                content[:100].replace("\n", " ")
            )
            return ParsedResponse(text=content, result=content, reasoning=None)

    result = str(obj.get("result", "")).strip() or content
    reasoning = obj.get("reasoning")
    has_finding = obj.get("has_finding")
    organ = obj.get("organ")
    modality = obj.get("modality")

    return ParsedResponse(
        text=content,
        result=result,
        reasoning=reasoning if isinstance(reasoning, str) else None,
        has_finding=bool(has_finding) if isinstance(has_finding, bool) or has_finding is not None else None,
        organ=str(organ) if organ is not None else None,
        modality=str(modality) if modality is not None else None,
    )


