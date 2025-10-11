import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from openai import OpenAI


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    temperature: float = 0.2
    max_tokens: int = 512


class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig) -> None:
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be provided in env or config")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.cfg = cfg

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


