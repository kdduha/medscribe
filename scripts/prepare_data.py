import os
import os.path as osp
import json
from typing import Iterable, List, Dict

from hydra import main
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset, DatasetDict

from src.data_preparation.download_processed_from_drive import run as run_download
from src.data_preparation.compile_datasets import run as run_compile
from src.data_preparation.compile_prompts import run as run_prompts
from src.data_preparation.prepare_for_sft import run as run_sft_prep


def _normalize_steps(steps: Iterable[str]) -> list:
    return [s.strip().lower() for s in steps if s and str(s).strip()]


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _build_messages_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        prompt = str(r.get("prompt_text", "")).strip()
        answer = str(r.get("result_text", "")).strip()
        if not prompt or not answer:
            continue
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        out.append({"messages": messages})
    return out


@main(config_path="../configs", config_name="prepare_data", version_base="1.3")
def run(cfg: DictConfig) -> None:
    merged: DictConfig = cfg

    steps = _normalize_steps(merged.get("steps", ["download", "compile", "prompts"]))
    run_download_step = "download" in steps
    run_compile_step = "compile" in steps
    run_prompts_step = "prompts" in steps

    if run_download_step:
        print("=== Step 1: Download processed CSVs (optional) ===")
        num_downloaded = run_download(merged.data)
        print(f"Downloaded: {num_downloaded}")

    if run_compile_step:
        print("\n=== Step 2: Compile datasets into a single CSV ===")
        num_rows = run_compile(merged.compile)
        print(f"Rows written: {num_rows}")

    if run_prompts_step:
        print("\n=== Step 3: Build training prompts ===")
        num_prompts = run_prompts(merged.prompts)
        print(f"Prompts saved: {num_prompts}")

        # Optional SFT prep delegated to module
        sft_prep = merged.get("sft_prep")
        if sft_prep and bool(sft_prep.get("enabled", True)):
            print("\n=== Step 4: Prepare SFT datasets (messages format) ===")
            # Merge default source_jsonl without mutating struct config
            sft_cfg = sft_prep
            if not sft_prep.get("source_jsonl"):
                sft_cfg = OmegaConf.merge(sft_prep, {"source_jsonl": merged.prompts.output_jsonl})
            sizes = run_sft_prep(sft_cfg)
            print(f"SFT datasets saved: train={sizes.get('train', 0)} eval={sizes.get('eval', 0)}")


if __name__ == "__main__":
    run()
