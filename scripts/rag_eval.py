import os
import os.path as osp
import json
import csv
import logging
from typing import List, Dict

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from src.rag.retriever import FaissRetriever
from src.rag.prompt_builder import build_prompt, PromptConfig
from src.rag.llm_client import OpenAICompatClient, LLMConfig
from src.rag.postprocess import parse_llm_response
from src.validation.metrics import compute_all_metrics, compute_classification_accuracy, normalize_modality
from src.logger import setup_logger


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@main(config_path="../configs", config_name="rag_eval", version_base="1.3")
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logger()
    LOG = logging.getLogger(__name__)

    index_dir = str(cfg.get("index_dir", "artifacts/rag_index"))
    input_jsonl = str(cfg.get("input_jsonl"))
    top_k = int(cfg.get("top_k", 5))
    cot = bool(cfg.get("cot", False))
    json_output = bool(cfg.get("json_output", True))
    out_csv = str(cfg.outputs.get("csv", "outputs/team_name_results.csv"))
    metrics_json = str(cfg.outputs.get("metrics", "outputs/metrics.json"))

    model = str(cfg.llm.get("model", os.environ.get("OPENAI_MODEL", "gpt-5-mini")))
    api_key = cfg.llm.get("api_key")
    base_url = cfg.llm.get("base_url")
    temperature = float(cfg.llm.get("temperature", 0.2))
    max_tokens = int(cfg.llm.get("max_tokens", 512))

    retriever = FaissRetriever()
    retriever.load(index_dir)

    llm = OpenAICompatClient(LLMConfig(api_key=api_key, base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens))
    rows = read_jsonl(input_jsonl)

    # Prepare CSV
    os.makedirs(osp.dirname(out_csv) or ".", exist_ok=True)
    write_header = not osp.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "finding", "gt_modality", "gt_organ", "gt_result", "pred_modality", "pred_organ", "pred_result", "exact_match", "bleu", "rouge", "meteor", "levenshtein", "acc_has_finding", "acc_organ", "acc_modality"])

        acc_vals = []
        bleu_vals = []
        rouge_vals = []
        meteor_vals = []
        lev_vals = []
        acc_has_vals = []
        acc_org_vals = []
        acc_mod_vals = []

        for i, r in enumerate(rows):
            modality = str(r.get("modality", ""))
            organ = str(r.get("organ", r.get("organ_abbr", "")))
            finding = str(r.get("finding", r.get("finding_text", "")))
            reference = str(r.get("result", r.get("result_text", "")))
            gt_modality = str(r.get("modality", "")) or normalize_modality(finding)
            if not finding:
                continue

            examples = retriever.search(finding, top_k=top_k)
            prompt_cfg = PromptConfig(json_output=json_output, cot=cot, max_examples=top_k)
            prompt = build_prompt(organ, finding, examples, prompt_cfg)
            content = llm.chat(prompt)
            parsed = parse_llm_response(content, expect_json=json_output)
            predicted = parsed.result

            m = compute_all_metrics(reference, predicted)
            acc_vals.append(m["exact_match"])  # accuracy equals exact_match here
            bleu_vals.append(m["bleu"])
            rouge_vals.append(m["rouge"])
            meteor_vals.append(m["meteor"])
            lev_vals.append(m["levenshtein"])

            cm = compute_classification_accuracy(
                true_has_finding=bool(len(finding.strip()) > 0),
                pred_has_finding=parsed.has_finding,
                true_organ=organ,
                pred_organ=parsed.organ,
                true_modality=modality,
                pred_modality=parsed.modality,
            )
            acc_has_vals.append(cm["acc_has_finding"])
            acc_org_vals.append(cm["acc_organ"])
            acc_mod_vals.append(cm["acc_modality"])

            writer.writerow([
                i,
                finding,
                gt_modality,
                organ,
                reference,
                (parsed.modality or ""),
                (parsed.organ or ""),
                predicted,
                int(m["exact_match"] == 1.0),
                f"{m['bleu']:.4f}",
                f"{m['rouge']:.4f}",
                f"{m['meteor']:.4f}",
                f"{m['levenshtein']:.4f}",
                int(cm["acc_has_finding"] == 1.0),
                int(cm["acc_organ"] == 1.0),
                int(cm["acc_modality"] == 1.0),
            ])
            
            LOG.info(
                "\n--------------------------------"
                "\n[id: %s]"
                "\nFinding: %s"
                "\nGROUND TRUTH | Modality: %s, Organ: %s, Result: %s"
                "\nPREDICTION | Modality: %s, Organ: %s, Result: %s"
                "\nMetrics| EM=%.3f BLEU=%.3f ROUGE=%.3f METEOR=%.3f Lev=%.3f has_finding=%s organ_ok=%s modality_ok=%s"
                "\n--------------------------------",
                i, finding, modality, organ, reference, parsed.modality, parsed.organ, parsed.result,
                m.get("exact_match", None),
                m.get("bleu", None),
                m.get("rouge", None),
                m.get("meteor", None),
                m.get("levenshtein", None),
                bool(cm.get("acc_has_finding", None) == 1.0),
                bool(cm.get("acc_organ", None) == 1.0),
                bool(cm.get("acc_modality", None) == 1.0),
                )

            # LOG.info(
            #     "Metrics| EM=%.3f BLEU=%.3f ROUGE=%.3f METEOR=%.3f Lev=%.3f has_finding=%s organ_ok=%s modality_ok=%s",
            #     m.get("exact_match", None),
            #     m.get("bleu", None),
            #     m.get("rouge", None),
            #     m.get("meteor", None),
            #     m.get("levenshtein", None),
            #     bool(cm.get("acc_has_finding", None) == 1.0),
            #     bool(cm.get("acc_organ", None) == 1.0),
            #     bool(cm.get("acc_modality", None) == 1.0),
            # )

    # aggregate
    n = max(1, len(acc_vals))
    metrics = {
        "accuracy": float(sum(acc_vals) / n),
        "bleu": float(sum(bleu_vals) / n),
        "rougeL": float(sum(rouge_vals) / n),
        "meteor": float(sum(meteor_vals) / n),
        "levenshtein": float(sum(lev_vals) / n),
        "acc_has_finding": float(sum(acc_has_vals) / max(1, len(acc_has_vals))),
        "acc_organ": float(sum(acc_org_vals) / max(1, len(acc_org_vals))),
        "acc_modality": float(sum(acc_mod_vals) / max(1, len(acc_mod_vals))),
    }
    os.makedirs(osp.dirname(metrics_json) or ".", exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Aggregated metrics:", metrics)


if __name__ == "__main__":
    run()


