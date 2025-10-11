from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from rapidfuzz.distance import Levenshtein
try:
    from nltk.translate.meteor_score import meteor_score  # type: ignore
    _HAS_NLTK_METEOR = True
except Exception:
    _HAS_NLTK_METEOR = False
try:
    import nltk  # type: ignore
except Exception:
    nltk = None  # type: ignore


def compute_bleu(references: List[str], predictions: List[str]) -> float:
    bleu = BLEU(effective_order=True)
    return float(bleu.corpus_score(predictions, [references]).score)


def compute_rouge_l(references: List[str], predictions: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    scores = []
    for ref, pred in zip(references, predictions):
        s = scorer.score(ref, pred)
        scores.append(s["rougeLsum"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


def compute_meteor(references: List[str], predictions: List[str]) -> float:
    if not _HAS_NLTK_METEOR:
        return 0.0
    # Ensure required NLTK resources are available (punkt, wordnet)
    if nltk is not None:
        try:
            nltk.data.find('tokenizers/punkt')
        except Exception:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find('corpora/wordnet')
        except Exception:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except Exception:
                pass
    scores = []
    for ref, pred in zip(references, predictions):
        try:
            scores.append(meteor_score([ref], pred))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_levenshtein(references: List[str], predictions: List[str]) -> float:
    # Use normalized_similarity directly (range [0,1]) for stability
    sims = [Levenshtein.normalized_similarity(r or "", p or "") for r, p in zip(references, predictions)]
    return float(np.mean(sims)) if sims else 0.0


def compute_exact_match(references: List[str], predictions: List[str]) -> float:
    matches = [1.0 if r.strip() == p.strip() else 0.0 for r, p in zip(references, predictions)]
    return float(np.mean(matches)) if matches else 0.0


def compute_all_metrics(reference: str, prediction: str) -> Dict[str, float]:
    refs = [reference]
    preds = [prediction]
    return {
        "bleu": compute_bleu(refs, preds),
        "rouge": compute_rouge_l(refs, preds),
        "meteor": compute_meteor(refs, preds),
        "levenshtein": compute_levenshtein(refs, preds),
        "exact_match": compute_exact_match(refs, preds),
    }


def normalize_modality(text: str) -> str:
    t = (text or "").strip().lower()
    if "мр" in t or "mri" in t or "мрт" in t:
        return "МРТ"
    if "кт" in t or "ct" in t:
        return "КТ"
    return t.upper()


def compute_classification_accuracy(true_has_finding: bool, pred_has_finding: Optional[bool],
                                    true_organ: str, pred_organ: Optional[str],
                                    true_modality_hint: str, pred_modality: Optional[str]) -> Dict[str, float]:
    
    acc_has = 1.0 if pred_has_finding is not None and bool(pred_has_finding) == bool(true_has_finding) else 0.0
    acc_org = 1.0 if (pred_organ or "").strip().lower() == (true_organ or "").strip().lower() and pred_organ is not None else 0.0
    true_mod = normalize_modality(true_modality_hint)
    pred_mod = normalize_modality(pred_modality or "") if pred_modality else ""
    acc_mod = 1.0 if pred_mod and true_mod and pred_mod == true_mod else 0.0
    return {
        "acc_has_finding": acc_has,
        "acc_organ": acc_org,
        "acc_modality": acc_mod,
    }


