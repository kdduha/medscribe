import re
import os
import json
import logging
from openai import OpenAI
from dataclasses import dataclass, field
from src.logger import setup_logger
from typing import Pattern

setup_logger()
LOG = logging.getLogger(__name__)
LABEL_MAP: dict[str, str] = {
    "NAME": "[NAME]",
    "DATE": "[DATE]",
    "DOB": "[DATE]",
    "STUDY_DATE": "[DATE]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ID": "[ID]",
    "POLICY": "[ID]",
    "SNILS": "[ID]",
    "ADDRESS": "[ADDRESS]",
    "ORG": "[ORG]",
    "LOCATION": "[ADDRESS]",
    "COMBO": "[PII_COMBO]",
}


@dataclass
class AnonymizationResult:
    text: str
    # each tuple: (start_char, end_char, label, original_text)
    replacements: list[tuple[int, int, str, str]] = field(default_factory=list)

    def to_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for _, _, label, _ in self.replacements:
            summary[label] = summary.get(label, 0) + 1
        return {"counts": summary, "total_replacements": len(self.replacements)}


class BaseEngine:
    def anonymize(self, text: str) -> AnonymizationResult:
        raise NotImplementedError


class RegexEngine(BaseEngine):

    def __init__(self, label_map: dict[str, str] | None = None, extra_patterns: list[tuple[str, str]] = None):
        self.label_map = label_map or LABEL_MAP

        # combine default with extras
        self.patterns: list[tuple[str, Pattern]] = list(self._build_default_patterns())
        if extra_patterns:
            for label, pat in extra_patterns:
                self.patterns.append((label, re.compile(pat, re.IGNORECASE)))

    @staticmethod
    def _build_default_patterns() -> list[tuple[str, Pattern]]:
        return [
            ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
            ("PHONE", re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?|0)?\d{3,4}[\s-]?\d{2,3}[\s-]?\d{2,3}")),
            ("DATE", re.compile(r"\b(?:\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|(?:янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\s+\d{1,2},?\s*\d{4})\b")),
            ("DOB", re.compile(r"(Дата\s+рождения|дата\s+рожд\.?)[^\n:;\d]{0,5}[:\s]*\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4}")),
            ("POLICY", re.compile(r"(полис|полиса|полис ОМС|№ полиса)[:\s]*[A-Z0-9\-\s]{4,30}")),
            ("SNILS", re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}\b")),
            ("NAME", re.compile(
                r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]\.[А-ЯЁ]\.|\s+[А-ЯЁ]\.\s+[А-ЯЁ]\.)\b"
            )),
            ("ADDRESS", re.compile(r"\b(?:ул\.|улица|проспект|просп\.|пл\.|площадь|пер\.|перевулок|г\.|город)\s+[\w\d\-\.]+")),
            ("ORG", re.compile(r"\b(?:ФГБУ|ООО|ЗАО|ОАО|Больница|Клиника|Поликлиника|Лаборатория)\b[\w\s,\-\.]*")),
            ("LOCATION", re.compile(r"\b(город|г\.|г\s)[\s\S]{1,40}?\b[А-ЯЁ][а-яё]+")),
        ]

    def anonymize(self, text: str) -> AnonymizationResult:
        result = AnonymizationResult(text=text)
        replacements: list[tuple[int, int, str, str]] = []

        for label, pattern in self.patterns:
            for m in pattern.finditer(text):
                s, e = m.start(), m.end()
                original = text[s:e]
                replacements.append((s, e, label, original))

        replacements.sort(key=lambda x: x[0])
        merged: list[tuple[int, int, str, str]] = []
        last_end = -1
        for s, e, label, orig in replacements:
            if s >= last_end:
                merged.append((s, e, label, orig))
                last_end = e
            else:
                if merged and e > merged[-1][1]:
                    merged[-1] = (merged[-1][0], e, merged[-1][2], merged[-1][3] + "|" + orig)
                last_end = e

        out = text
        for s, e, label, orig in reversed(merged):
            token = self.label_map.get(label, f"[{label}]")
            out = out[:s] + token + out[e:]

        result.text = out
        result.replacements = merged
        return result


class OpenAIEngine(BaseEngine):
    def __init__(self, max_tokens: int = 2048):
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL")
        )

    def anonymize(self, text: str) -> AnonymizationResult:
        chunks = self._chunk_text(text)
        full_text = ""
        replacements: list[tuple[int, int, str, str]] = []
        offset = 0

        for chunk in chunks:
            chunk_result = self._anonymize_chunk(chunk)
            for start, end, label, original in chunk_result.replacements:
                replacements.append((start + offset, end + offset, label, original))
            full_text += chunk_result.text
            offset += len(chunk)

        result = AnonymizationResult(text=full_text, replacements=replacements)
        return result

    @staticmethod
    def _chunk_text(text: str, size: int = 500) -> list[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _anonymize_chunk(self, chunk: str) -> AnonymizationResult:
        prompt = self._build_prompt(chunk)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "Ты — очень строгий анонимизатор медицинских текстов. Дословно следуй моим инструкциям."},
                      {"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        content = resp.choices[0].message.content.strip()
        parsed = self._parse_response(content)
        text = parsed.get("text", chunk)
        spans = parsed.get("spans", [])
        replacements = [
            (s["start"], s["end"], s["label"], s.get("text", text[s["start"]:s["end"]]))
            for s in spans
        ]
        return AnonymizationResult(text=text, replacements=replacements)

    @staticmethod
    def _build_prompt(chunk: str) -> str:
        few_shot_examples = """
    Пример 1:
    Текст: "Пациент Иванов Иван Иванович пришел на обследование 22.04.2015"
    Ответ JSON:
    {
      "text": "Пациент [NAME] пришел на обследование [DATE]",
      "spans": [
        {"start": 8, "end": 27, "label": "NAME", "text": "Иванов Иван Иванович"},
        {"start": 45, "end": 55, "label": "DATE", "text": "22.04.2015"}
      ]
    }

    Пример 2:
    Текст: "Ф.И.О.      Кузовенкова Н.Н.\nДата рождения  1979 Москва  исследования   Органы грудной клетки"
    Ответ JSON:
    {
      "text": "Ф.И.О.      [NAME]\nДата рождения  [DATE] [ADDRESS]  исследования   Органы грудной клетки",
      "spans": [
        {"start": 7, "end": 24, "label": "NAME", "text": "Кузовенкова Н.Н."},
        {"start": 36, "end": 40, "label": "DATE", "text": "1979"},
        {"start": 41, "end": 47, "label": "ADDRESS", "text": "Москва"}
      ]
    }

    Пример 3:
    Текст: "20 июня 2007 Москва: ________Попов П.А."
    Ответ JSON:
    {
      "text": "[DATE] [ADDRESS]: ________[NAME]",
      "spans": [
        {"start": 0, "end": 11, "label": "DATE", "text": "20 июня 2007"},
        {"start": 12, "end": 18, "label": "ADDRESS", "text": "Москва"},
        {"start": 21, "end": 30, "label": "NAME", "text": "Попов П.А."}
      ]
    }
    """
        return (
            "Задача: заменить все персональные данные на токены из label_map: [NAME], [DATE], [ADDRESS], [PHONE], [POLICY], [SNILS], [EMAIL]. "
            "Данные о местоположении и имени больницы тоже считаются как критичные и попадают в NAME/ADDRESS. "
            "Ты должен возвращать JSON с полями 'text' (анонимизированный текст) и 'spans' (список объектов с start, end, label, text), пример формата см. ниже.\n\n"
            f"{few_shot_examples}\n"
            "Если сомневаешься, отмечай потенциальные персональные данные. Не надо переходить в **reasoning** режим."
            "Если в тексте нет данных для анонимизации, верни исходный текст как 'text' и пустой список 'spans'.\n"
            f"**Текст для анонимизации**:\n\n{chunk}\n"
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content, "spans": []}