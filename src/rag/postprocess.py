import json
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

    raw = content
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: try to extract result line
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


