import os
import os.path as osp
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    faiss = None  # lazy check later


@dataclass
class RetrievedExample:
    finding: str
    result: str
    score: float


class FaissRetriever:
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.corpus_findings: List[str] = []
        self.corpus_results: List[str] = []

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=self.normalize, show_progress_bar=False)
        return emb.astype("float32")

    # ---- Building index ----
    def build_from_jsonl(self, jsonl_path: str, finding_key: str = "finding_text", result_key: str = "result_text") -> int:
        if faiss is None:
            raise ImportError("faiss-cpu is required for retrieval. Please install faiss-cpu.")
        if not osp.exists(jsonl_path):
            raise FileNotFoundError(jsonl_path)
        findings: List[str] = []
        results: List[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                fnd = str(obj.get(finding_key, "")).strip()
                res = str(obj.get(result_key, "")).strip()
                if fnd and res:
                    findings.append(fnd)
                    results.append(res)

        if not findings:
            raise RuntimeError("No (finding,result) pairs found to index")

        emb = self._encode(findings)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        if not self.normalize:
            # if not normalized, use L2 index as fallback
            index = faiss.IndexFlatL2(dim)
        index.add(emb)

        self.index = index
        self.corpus_findings = findings
        self.corpus_results = results
        return len(findings)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index is not built")
        # save FAISS
        faiss.write_index(self.index, osp.join(dir_path, "index.faiss"))
        # save meta
        meta = {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "findings": self.corpus_findings,
            "results": self.corpus_results,
        }
        with open(osp.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, dir_path: str) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required for retrieval. Please install faiss-cpu.")
        index_path = osp.join(dir_path, "index.faiss")
        meta_path = osp.join(dir_path, "meta.json")
        if not osp.exists(index_path) or not osp.exists(meta_path):
            raise FileNotFoundError("Missing index.faiss or meta.json in retriever dir")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name = meta.get("model_name", self.model_name)
        self.normalize = bool(meta.get("normalize", True))
        self.corpus_findings = list(meta.get("findings", []))
        self.corpus_results = list(meta.get("results", []))

    # ---- Query ----
    def search(self, finding: str, top_k: int = 5) -> List[RetrievedExample]:
        if self.index is None:
            raise RuntimeError("Index is not loaded/built")
        q = self._encode([finding])  # (1, d)
        scores, idx = self.index.search(q, top_k)
        out: List[RetrievedExample] = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
            if i < 0:
                continue
            out.append(RetrievedExample(
                finding=self.corpus_findings[i],
                result=self.corpus_results[i],
                score=float(s),
            ))
        return out


