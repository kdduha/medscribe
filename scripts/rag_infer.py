import os
import os.path as osp
import json
import csv
import logging
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from src.rag.retriever import FaissRetriever
from src.rag.prompt_builder import build_prompt, PromptConfig
from src.rag.llm_client import OpenAICompatClient, LLMConfig
from src.rag.postprocess import parse_llm_response
from src.validation.metrics import compute_all_metrics, compute_classification_accuracy, normalize_modality
from src.logger import setup_logger


def _ensure_dir(path: str) -> None:
    d = osp.dirname(path)
    if d and not osp.exists(d):
        os.makedirs(d, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_rows(path: str, fmt: str, columns: Dict[str, str]) -> list[dict[str, str]]:
    fmt_l = (fmt or "auto").lower()
    if fmt_l == "auto":
        if path.lower().endswith(".jsonl"):
            fmt_l = "jsonl"
        else:
            fmt_l = "csv"

    rows: list[dict[str, str]] = []
    if fmt_l == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(obj)
    else:
        import csv as _csv
        with open(path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for r in reader:
                rows.append(dict(r))
    # normalize fields using columns mapping
    out: list[dict[str, str]] = []
    col_id = columns.get("id", "id")
    col_organ = columns.get("organ", "organ")
    col_finding = columns.get("finding", "finding")
    col_ref = columns.get("reference", "result_text")
    col_mod = columns.get("modality", "modality")
    def _infer_modality(text: str) -> str:
        return normalize_modality(text)
    for i, r in enumerate(rows):
        mod_val = str(r.get(col_mod, ""))
        if not mod_val:
            mod_val = _infer_modality(str(r.get(col_finding, r.get("finding_text", ""))))
        out.append({
            "id": str(r.get(col_id, i)),
            "organ": str(r.get(col_organ, r.get("organ_abbr", ""))),
            "finding": str(r.get(col_finding, r.get("finding_text", ""))),
            "reference": str(r.get(col_ref, r.get("result", r.get("result_text", "")))),
            "modality": mod_val,
        })
    return out


@main(config_path="../configs", config_name="rag_infer", version_base="1.3")
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logger()
    LOG = logging.getLogger(__name__)

    index_dir = str(cfg.get("index_dir", "artifacts/rag_index"))
    organ = str(cfg.get("organ", ""))
    finding = str(cfg.get("finding", ""))
    reference = str(cfg.get("reference", ""))
    top_k = int(cfg.get("top_k", 5))
    cot = bool(cfg.get("cot", False))
    json_output = bool(cfg.get("json_output", True))

    # optional batch input
    input_cfg = cfg.get("input", {})
    input_file = str(input_cfg.get("file", "")) if input_cfg is not None else ""
    input_fmt = str(input_cfg.get("format", "auto")) if input_cfg is not None else "auto"
    input_columns = dict(input_cfg.get("columns", {})) if input_cfg is not None else {}

    model = str(cfg.llm.get("model", os.environ.get("OPENAI_MODEL", "gpt-5-mini")))
    api_key = cfg.llm.get("api_key")
    base_url = cfg.llm.get("base_url")
    temperature = float(cfg.llm.get("temperature", 0.2))
    max_tokens = int(cfg.llm.get("max_tokens", 512))

    out_csv = str(cfg.outputs.get("csv", "outputs/team_name_results.csv"))
    logs_dir = str(cfg.outputs.get("logs", "outputs/logs"))
    metrics_json = str(cfg.outputs.get("metrics", "outputs/metrics.json"))

    LOG.info("Loading retriever index from %s", index_dir)
    retriever = FaissRetriever()
    retriever.load(index_dir)
    LOG.info("Initializing LLM client: model=%s base_url=%s", model, str(base_url or ""))
    llm = OpenAICompatClient(LLMConfig(api_key=api_key, base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens))

    # prepare CSV
    _ensure_dir(out_csv)
    write_header = not osp.exists(out_csv) or os.path.getsize(out_csv) == 0
    f_csv = open(out_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["id", "finding", "gt_modality", "gt_organ", "gt_result", "pred_modality", "pred_organ", "pred_result"])

    def infer_one(row_id: str, org: str, fnd: str, ref: str, gt_mod: str) -> Dict[str, Any]:
        LOG.info("Infer id=%s", row_id)
        LOG.info("Input: organ=%s", org)
        LOG.info("Input finding:\n%s", fnd)
        examples = retriever.search(fnd, top_k=top_k)
        prompt_cfg = PromptConfig(json_output=json_output, cot=cot, max_examples=top_k)
        prompt = build_prompt(org, fnd, examples, prompt_cfg)
        LOG.info("Prompt:\n%s", prompt)
        content = llm.chat(prompt)
        parsed = parse_llm_response(content, expect_json=json_output)
        predicted_local = parsed.result
        LOG.info("Raw output:\n%s", content)
        LOG.info("Parsed result: %s", predicted_local)

        metrics = {"bleu": 0.0, "rouge": 0.0, "meteor": 0.0, "levenshtein": 0.0, "exact_match": 0.0}
        if ref:
            metrics = compute_all_metrics(ref, predicted_local)
        # classification metrics (if we have ground truth hints)
        class_metrics = compute_classification_accuracy(
            true_has_finding=bool(len(fnd.strip()) > 0),
            pred_has_finding=parsed.has_finding,
            true_organ=(org or ""),
            pred_organ=parsed.organ,
            true_modality_hint=fnd,
            pred_modality=parsed.modality,
        )

        LOG.info(
            "Metrics[id=%s]: EM=%.3f BLEU=%.3f ROUGE=%.3f METEOR=%.3f Lev=%.3f | has=%s organ_ok=%s modality_ok=%s",
            row_id,
            metrics.get("exact_match", None),
            metrics.get("bleu", None),
            metrics.get("rouge", None),
            metrics.get("meteor", None),
            metrics.get("levenshtein", None),
            bool(class_metrics.get("acc_has_finding", None) == 1.0),
            bool(class_metrics.get("acc_organ", None) == 1.0),
            bool(class_metrics.get("acc_modality", None) == 1.0),
        )

        writer.writerow([
            row_id,
            fnd,
            gt_mod,
            org,
            ref,
            (parsed.modality or ""),
            (parsed.organ or ""),
            predicted_local,
        ])

        return {
            "predicted": predicted_local,
            "metrics": metrics,
            "class_metrics": class_metrics,
            "examples": [{"finding": e.finding, "result": e.result, "score": e.score} for e in examples],
            "prompt": prompt,
            "raw_response": content,
            "reasoning": parsed.reasoning,
            "has_finding": parsed.has_finding,
            "organ_pred": parsed.organ,
            "modality_pred": parsed.modality,
        }

    # run single or batch
    if input_file:
        rows = _read_rows(input_file, input_fmt, input_columns)
        for r in rows:
            rid = str(r.get("id"))
            org = str(r.get("organ", ""))
            fnd = str(r.get("finding", ""))
            ref = str(r.get("reference", ""))
            if not fnd:
                continue
            gt_mod = str(r.get("modality", "")) or normalize_modality(fnd)
            out = infer_one(rid, org or organ, fnd, ref, gt_mod)
            now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            log_obj = {
                "timestamp": now,
                "input": {"id": rid, "organ": org or organ, "finding": fnd},
                **out,
            }
            _ensure_dir(logs_dir)
            _write_json(osp.join(logs_dir, f"log_{now}_{rid}.json"), log_obj)
    else:
        gt_mod = normalize_modality(finding)
        out = infer_one("single", organ, finding, reference, gt_mod)
        now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        log_obj = {
            "timestamp": now,
            "input": {"organ": organ, "finding": finding},
            **out,
        }
        _ensure_dir(logs_dir)
        _write_json(osp.join(logs_dir, f"log_{now}.json"), log_obj)
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # save logs
    # close CSV file
    f_csv.close()


if __name__ == "__main__":
    run()


