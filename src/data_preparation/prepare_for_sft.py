import os
import os.path as osp
import json
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict
from hydra.utils import to_absolute_path


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


def run(cfg: Any) -> Dict[str, int]:
    """
    Prepare SFT datasets from prompts JSONL:
      - Convert to chat messages format
      - Optional train/val split
      - Save JSONL and HF dataset dir

    cfg fields:
      - source_jsonl: str (defaults to prompts.output_jsonl)
      - split.enabled: bool
      - split.test_size: float
      - split.seed: int
      - outputs.train_jsonl / outputs.eval_jsonl / outputs.hf_dir
    """
    source_jsonl = to_absolute_path(str(cfg.get("source_jsonl", "datasets/train_prompts.jsonl")))
    if not osp.exists(source_jsonl):
        raise FileNotFoundError(f"Source JSONL not found: {source_jsonl}")

    rows = _read_jsonl(source_jsonl)
    msg_rows = _build_messages_rows(rows)
    if not msg_rows:
        raise RuntimeError("No valid prompt/response pairs to prepare for SFT")

    if cfg.get("split", {}).get("enabled", True):
        test_size = float(cfg.split.get("test_size", 0.1))
        seed = int(cfg.split.get("seed", 42))
        ds_full = Dataset.from_list(msg_rows)
        split = ds_full.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        ds_train = split["train"]
        ds_eval = split.get("test")
    else:
        ds_train = Dataset.from_list(msg_rows)
        ds_eval = None

    out_train = to_absolute_path(str(cfg.outputs.get("train_jsonl", "datasets/sft_train.jsonl")))
    out_eval = to_absolute_path(str(cfg.outputs.get("eval_jsonl", "datasets/sft_eval.jsonl")))
    out_hf_dir = to_absolute_path(str(cfg.outputs.get("hf_dir", "datasets/sft_dataset")))

    train_rows = ds_train.to_list()
    _write_jsonl(out_train, train_rows)
    if ds_eval is not None and len(ds_eval) > 0:
        eval_rows = ds_eval.to_list()
        _write_jsonl(out_eval, eval_rows)
    else:
        if osp.exists(out_eval):
            try:
                os.remove(out_eval)
            except OSError:
                pass

    dsd = {"train": ds_train}
    if ds_eval is not None and len(ds_eval) > 0:
        dsd["eval"] = ds_eval
    ds = DatasetDict(dsd)
    os.makedirs(out_hf_dir, exist_ok=True)
    ds.save_to_disk(out_hf_dir)

    return {"train": len(ds_train), "eval": (len(ds_eval) if ds_eval is not None else 0)}


