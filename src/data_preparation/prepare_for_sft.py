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


def _build_messages_rows_with_index(rows: List[Dict]) -> Dict[str, Any]:
    """
    Build chat messages and keep mapping to original prompt rows to enable test split export.
    Returns dict with keys: messages_rows (list), source_rows (list[Dict]), organ_by_source_idx (list[str]).
    """
    messages_rows: List[Dict] = []
    source_rows: List[Dict] = []
    organ_by_idx: List[str] = []
    for r in rows:
        prompt = str(r.get("prompt_text", "")).strip()
        answer = str(r.get("result_text", "")).strip()
        if not prompt or not answer:
            continue
        organ = str(r.get("organ", r.get("organ_abbr", "")))
        source_rows.append(r)
        organ_by_idx.append(organ)
        idx = len(source_rows) - 1
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        messages_rows.append({"messages": messages, "orig_index": idx})
    return {"messages_rows": messages_rows, "source_rows": source_rows, "organ_by_idx": organ_by_idx}


gdef _infer_modality(text: str, fallback: str = "") -> str:
    t = (text or "").lower()
    if "мр" in t or "мрт" in t or "mri" in t:
        return "МРТ"
    if "кт" in t or "ct" in t:
        return "КТ"
    return fallback


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
    built = _build_messages_rows_with_index(rows)
    msg_rows = built["messages_rows"]
    source_rows = built["source_rows"]
    organ_by_idx = built["organ_by_idx"]
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
    out_test = to_absolute_path(str(cfg.outputs.get("test_jsonl", "datasets/sft_test_rag.jsonl")))
    out_hf_dir = to_absolute_path(str(cfg.outputs.get("hf_dir", "datasets/sft_dataset")))

    # Write SFT train/eval chat JSONLs (strip helper field)
    train_rows = [{"messages": r["messages"]} for r in ds_train.to_list()]
    _write_jsonl(out_train, train_rows)
    if ds_eval is not None and len(ds_eval) > 0:
        eval_list = ds_eval.to_list()
        eval_rows = [{"messages": r["messages"]} for r in eval_list]
        _write_jsonl(out_eval, eval_rows)
        # Also write test JSONL for RAG inference compatibility using the same eval samples
        test_rows: List[Dict] = []
        for r in eval_list:
            idx = int(r.get("orig_index", -1))
            if 0 <= idx < len(source_rows):
                src = source_rows[idx]
                # modality preference: if present in source, use it; otherwise infer from finding_text
                modality_val = src.get("modality")
                if not modality_val:
                    modality_val = _infer_modality(str(src.get("finding_text", "")))
                test_rows.append({
                    "id": idx,
                    "organ": organ_by_idx[idx],
                    "finding_text": src.get("finding_text", ""),
                    "result_text": src.get("result_text", ""),
                    "modality": modality_val,
                })
        _write_jsonl(out_test, test_rows)
    else:
        if osp.exists(out_eval):
            try:
                os.remove(out_eval)
            except OSError:
                pass
        if osp.exists(out_test):
            try:
                os.remove(out_test)
            except OSError:
                pass

    dsd = {"train": ds_train}
    if ds_eval is not None and len(ds_eval) > 0:
        dsd["eval"] = ds_eval
    ds = DatasetDict(dsd)
    os.makedirs(out_hf_dir, exist_ok=True)
    ds.save_to_disk(out_hf_dir)

    return {"train": len(ds_train), "eval": (len(ds_eval) if ds_eval is not None else 0)}


