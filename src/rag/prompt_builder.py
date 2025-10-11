from dataclasses import dataclass
from typing import List

from .retriever import RetrievedExample


@dataclass
class PromptConfig:
    json_output: bool = True
    cot: bool = False
    max_examples: int = 5


def _format_examples(examples: List[RetrievedExample]) -> str:
    lines: List[str] = ["Примеры:"]
    for i, ex in enumerate(examples, start=1):
        lines.append(
            f"Пример {i}:\nНаходка: {ex.finding}\nЗаключение: {ex.result}\n"
        )
    return "\n".join(lines)


def build_prompt(organ: str, finding: str, examples: List[RetrievedExample], cfg: PromptConfig) -> str:
    ctx = _format_examples(examples[: cfg.max_examples]) if examples else "Примеры: (нет подходящих примеров)"
    if cfg.cot and cfg.json_output:
        return (
            "Ты — опытный радиолог. \n"
            "Проанализируй находку и рассуждай шаг за шагом перед формированием заключения. \n"
            "Используй приведённые примеры как ориентиры по стилю и структуре.\n\n"
            f"{ctx}\n\n"
            "Задача:\n"
            f"Орган: {organ}\n"
            f"Находка: {finding}\n\n"
            "Ответ верни строго в формате JSON со следующими полями:\n"
            "{\n"
            "  \"reasoning\": \"...\",\n"
            "  \"result\": \"...\",\n"
            "  \"has_finding\": true,  \n"
            "  \"organ\": \"...\",      \n"
            "  \"modality\": \"МРТ|КТ\" \n"
            "}\n\n"
            "Требования: has_finding=true/false; organ — анатомическая область; modality — только 'МРТ' или 'КТ'."
        )

    if cfg.cot and not cfg.json_output:
        return (
            "Ты — опытный радиолог. Рассуждай шаг за шагом и сформулируй итоговое заключение.\n\n"
            f"{ctx}\n\n"
            f"Орган: {organ}\n"
            f"Находка: {finding}\n\n"
            "Пошаговое рассуждение и итоговое заключение:"
        )

    if not cfg.cot and cfg.json_output:
        return (
            "Ты — медицинская LLM. Сформируй краткое заключение по находке.\n\n"
            f"{ctx}\n\n"
            f"Орган: {organ}\n"
            f"Находка: {finding}\n\n"
            "Ответ верни строго в формате JSON со следующими полями:\n"
            "{\n"
            "  \"result\": \"...\",\n"
            "  \"has_finding\": true,\n"
            "  \"organ\": \"...\",\n"
            "  \"modality\": \"МРТ|КТ\"\n"
            "}\n\n"
            "Требования: has_finding=true/false; organ — анатомическая область; modality — только 'МРТ' или 'КТ'."
        )

    return (
        "Ты — медицинская LLM. Сформируй краткое заключение по находке.\n\n"
        f"{ctx}\n\n"
        f"Орган: {organ}\n"
        f"Находка: {finding}\n\n"
        "Ответ:"
    )


