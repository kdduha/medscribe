import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path

def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Present: {list(df.columns)}")


def _split_dataframe(df: pd.DataFrame, simple_frac: float, medium_frac: float, seed: int) -> List[pd.DataFrame]:
    if not (0.0 < simple_frac < 1.0) or not (0.0 < medium_frac < 1.0) or simple_frac + medium_frac >= 1.0:
        raise ValueError("simple_frac and medium_frac must be in (0,1) and sum to < 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(df)
    n_simple = int(round(n * simple_frac))
    n_medium = int(round((n - n_simple) * medium_frac))

    idx_simple = indices[:n_simple]
    idx_medium = indices[n_simple:n_simple + n_medium]
    idx_cot = indices[n_simple + n_medium:]

    return [df.iloc[idx_simple].reset_index(drop=True), df.iloc[idx_medium].reset_index(drop=True), df.iloc[idx_cot].reset_index(drop=True)]


def make_simple_prompt(row: pd.Series, json_output: bool = False) -> str:
    if json_output:
        return (
            f"Ты — медицинская LLM. "
            f"По следующей находке сформируй краткое заключение. "
            f"Ответ верни в формате JSON с ключом 'result'.\n\n"
            f"Орган: {row['organ']}\n"
            f"Находка: {row['finding_text']}\n\n"
            f"Формат ответа:\n{{'result': '...'}}"
        )
    return (
        f"Ты — медицинская LLM. "
        f"По следующей находке сформируй краткое заключение:\n\n"
        f"Орган: {row['organ']}\n"
        f"Находка: {row['finding_text']}\n\n"
        f"Ответ:"
    )


def make_medium_prompt(row: pd.Series, json_output: bool = False) -> str:
    modality_hint = "МРТ" if "МР" in str(row.get("finding_text", "")) else "КТ"
    if json_output:
        return (
            f"Ты — нейросеть, помогающая радиологу писать заключения. "
            f"Тебе дана находка по исследованию {modality_hint} для органа {row['organ']}. "
            f"Сформируй полное, логически связанное заключение. "
            f"Ответ верни в формате JSON с ключом 'result'.\n\n"
            f"Находка: {row['finding_text']}\n\n"
            f"Формат:\n{{'result': '...'}}"
        )
    return (
        f"Ты — нейросеть, помогающая радиологу писать заключения. "
        f"Исследование: {modality_hint}. Орган: {row['organ']}. "
        f"Находка: {row['finding_text']}\n\n"
        f"Сформируй итоговое заключение:"
    )


def make_cot_prompt(row: pd.Series, json_output: bool = False) -> str:
    if json_output:
        return (
            f"Ты — опытный радиолог. "
            f"Проанализируй следующую находку, рассуждай шаг за шагом, а затем сформулируй финальное заключение. "
            f"Ответ верни в JSON с ключом 'reasoning' и 'result'.\n\n"
            f"Орган: {row['organ']}\n"
            f"Находка: {row['finding_text']}\n\n"
            f"Формат:\n{{'reasoning': '...', 'result': '...'}}"
        )
    return (
        f"Ты — опытный радиолог. "
        f"Проанализируй находку и рассуждай шаг за шагом перед ответом.\n\n"
        f"Орган: {row['organ']}\n"
        f"Находка: {row['finding_text']}\n\n"
        f"Пошаговое рассуждение и итоговое заключение:"
    )


def _build_prompt_df(df: pd.DataFrame, prompt_func, tag: str) -> pd.DataFrame:
    out_json = df.copy()
    out_json["result_json"] = df.apply(lambda x: f"{{'result': '{x['result_text']}'}}", axis=1)
    out_json["prompt_json"] = df.apply(lambda r: prompt_func(r, json_output=True), axis=1)
    out_json["prompt_text"] = df.apply(lambda r: prompt_func(r, json_output=False), axis=1)
    out_json["type"] = tag
    return out_json


def run(cfg: Any) -> int:
    input_file = to_absolute_path(str(cfg.get("input_file", "datasets/compiled.csv")))
    output_jsonl = to_absolute_path(str(cfg.get("output_jsonl", "datasets/train_prompts.jsonl")))
    seed = int(cfg.get("seed", 42))
    simple_frac = float(cfg.get("simple_frac", 0.2))
    medium_frac = float(cfg.get("medium_frac", 0.4))

    columns = cfg.get("source_columns", {}) or {}
    col_organ = columns.get("organ", "organ")
    col_finding = columns.get("finding", "finding")
    col_result = columns.get("result", "result")

    if not osp.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    lower = input_file.lower()
    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    _validate_columns(df, [col_organ, col_finding, col_result])

    df = df.rename(columns={
        col_organ: "organ_abbr",
        col_finding: "finding_text",
        col_result: "result_text",
    })

    organ_mapper_cfg = cfg.get("organ_mapper")
    if isinstance(organ_mapper_cfg, dict) and organ_mapper_cfg:
        df["organ"] = df["organ_abbr"].map(organ_mapper_cfg).fillna(df["organ_abbr"])
    else:
        df["organ"] = df["organ_abbr"]

    df_simple, df_medium, df_cot = _split_dataframe(df, simple_frac=simple_frac, medium_frac=medium_frac, seed=seed)

    simple_prompts = _build_prompt_df(df_simple, make_simple_prompt, "simple")
    medium_prompts = _build_prompt_df(df_medium, make_medium_prompt, "medium")
    cot_prompts = _build_prompt_df(df_cot, make_medium_prompt, "medium")
    # cot_prompts = _build_prompt_df(df_cot, make_cot_prompt, "cot")

    all_prompts = pd.concat([simple_prompts, medium_prompts, cot_prompts]).reset_index(drop=True)

    train_df = all_prompts[[
        "type",
        "organ",
        "finding_text",
        "result_text",
        "prompt_text",
        "result_json",
        "prompt_json",
    ]]

    os.makedirs(osp.dirname(output_jsonl) or ".", exist_ok=True)
    train_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)

    print(train_df.head(3))
    print(f"Сохранено {len(train_df)} примеров для fine-tuning в {output_jsonl}.")
    return len(train_df)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pipeline.py to run this module via Hydra.")


