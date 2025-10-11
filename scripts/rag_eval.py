import os
import os.path as osp
import json
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


# ---------- COSINE SIMILARITY HELPER ----------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity (0–1) between two vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


_EMBEDDER = SentenceTransformer('deepvk/USER-bge-m3')


def embed_texts(texts: List[str]) -> np.ndarray:
    return _EMBEDDER.encode(texts, normalize_embeddings=True, show_progress_bar=False)


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

    model = str(cfg.llm.get("model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
    api_key = cfg.llm.get("api_key")
    base_url = cfg.llm.get("base_url")
    temperature = float(cfg.llm.get("temperature", 0.2))
    max_tokens = int(cfg.llm.get("max_tokens", 512))

    retriever = FaissRetriever()
    retriever.load(index_dir)

    llm = OpenAICompatClient(
        LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )

    rows = read_jsonl(input_jsonl)

    # Prepare CSV
    os.makedirs(osp.dirname(out_csv) or ".", exist_ok=True)
    write_header = not osp.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "id", "modality", "organ", "finding", "predicted_result",
                "is_exact_match", "bleu", "rouge", "meteor",
                "acc_has_finding", "acc_organ_exact", "acc_organ_cosine", "acc_modality"
            ])

        acc_vals = []
        bleu_vals = []
        rouge_vals = []
        meteor_vals = []

        acc_has_vals = []
        acc_organ_exact_vals = []
        acc_organ_cosine_vals = []
        acc_mod_vals = []

        for i, r in enumerate(rows):
            modality = str(r.get("modality", ""))
            organ = str(r.get("organ", r.get("organ_abbr", "")))
            finding = str(r.get("finding", r.get("finding_text", "")))
            reference = str(r.get("result", r.get("result_text", "")))
            modality = str(r.get("modality", r.get("modality", "")))

            if not finding.strip():
                continue

            examples = retriever.search(finding, top_k=top_k)
            prompt_cfg = PromptConfig(json_output=json_output, cot=cot, max_examples=top_k)
            prompt = build_prompt(organ, finding, examples, prompt_cfg)
            content = llm.chat(prompt)
            parsed = parse_llm_response(content.reasoning_content, expect_json=json_output)
            predicted = parsed.result

            m = compute_all_metrics(reference, predicted)
            acc_vals.append(m["exact_match"])
            bleu_vals.append(m["bleu"])
            rouge_vals.append(m["rouge"])
            meteor_vals.append(m["meteor"])

            cm = compute_classification_accuracy(
                true_has_finding=bool(len(finding.strip()) > 15),
                pred_has_finding=parsed.has_finding,
                true_organ=organ,
                pred_organ=parsed.organ,
                true_modality=modality,
                pred_modality=parsed.modality,
            )

            acc_has_vals.append(cm["acc_has_finding"])
            acc_organ_exact_vals.append(cm["acc_organ"])
            acc_mod_vals.append(cm["acc_modality"])

            true_vec = embed_texts([organ])[0]
            pred_vec = embed_texts([parsed.organ or ""])[0]
            organ_cosine = cosine_similarity(true_vec, pred_vec)
            acc_organ_cosine_vals.append(organ_cosine)

            writer.writerow([
                i,
                modality,
                organ,
                finding,
                parsed,
                int(m["exact_match"]),
                f"{m['bleu']:.4f}",
                f"{m['rouge']:.4f}",
                f"{m['meteor']:.4f}",
                int(cm["acc_has_finding"]),
                int(cm["acc_organ"]),
                f"{organ_cosine:.4f}",
                int(cm["acc_modality"]),
            ])

            LOG.info(
                "ORGAN true: %s | pred: %s | exact: %s | cosine: %.3f",
                organ, parsed.organ, bool(cm["acc_organ"]), organ_cosine
            )

            LOG.info(
                "MODALITY true: %s | pred: %s | exact: %s",
                modality, parsed.modality, bool(cm["acc_modality"] == 1.0)
            )

            LOG.info(
                "Metrics[id=%s]: EM=%.3f BLEU=%.3f ROUGE=%.3f METEOR=%.3f | "
                "has=%s organ_exact=%s organ_cosine=%.3f modality=%s",
                i,
                m.get("exact_match"),
                m.get("bleu"),
                m.get("rouge"),
                m.get("meteor"),
                bool(cm.get("acc_has_finding")),
                bool(cm.get("acc_organ")),
                organ_cosine,
                bool(cm.get("acc_modality")),
            )

    n = max(1, len(acc_vals))
    metrics = {
        "accuracy": float(sum(acc_vals) / n),
        "bleu": float(sum(bleu_vals) / n),
        "rougeL": float(sum(rouge_vals) / n),
        "meteor": float(sum(meteor_vals) / n),
        "acc_has_finding": float(sum(acc_has_vals) / max(1, len(acc_has_vals))),
        "acc_organ_exact": float(sum(acc_organ_exact_vals) / max(1, len(acc_organ_exact_vals))),
        "acc_organ_cosine": float(sum(acc_organ_cosine_vals) / max(1, len(acc_organ_cosine_vals))),  # NEW
        "acc_modality": float(sum(acc_mod_vals) / max(1, len(acc_mod_vals))),
    }

    os.makedirs(osp.dirname(metrics_json) or ".", exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Aggregated metrics:", metrics)


if __name__ == "__main__":
    run()
