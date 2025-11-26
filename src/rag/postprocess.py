import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel  # type: ignore
from langchain_core.output_parsers import JsonOutputParser  # type: ignore


class _LangchainParsedModel(BaseModel):  # type: ignore
    result: Any = None
    reasoning: Optional[str] = None
    has_finding: Optional[bool] = None
    organ: Optional[str] = None
    modality: Optional[str] = None


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

    try:
        parser = JsonOutputParser(pydantic_object=_LangchainParsedModel)  # type: ignore
        parsed_obj = parser.parse(content)
        if hasattr(parsed_obj, "model_dump"):
            obj = parsed_obj.model_dump()  # pydantic v2
        elif hasattr(parsed_obj, "dict"):
            obj = parsed_obj.dict()  # pydantic v1
        elif isinstance(parsed_obj, dict):
            obj = parsed_obj
        else:
            obj = json.loads(parsed_obj)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "LangChain JSON parsing failed: %s. Returning raw text. Snippet: %s",
            e,
            content[:100].replace("\n", " ") + "..."
        )
        return ParsedResponse(text=content, result=content, reasoning=None)

    result = str(obj.get("result", "")).strip() or content
    reasoning = obj.get("reasoning", "")
    has_finding = obj.get("has_finding", "")
    organ = obj.get("organ", "")
    modality = obj.get("modality", "")

    return ParsedResponse(
        text=content,
        result=result,
        reasoning=reasoning,
        has_finding=bool(has_finding),
        organ=str(organ),
        modality=str(modality),
    )


