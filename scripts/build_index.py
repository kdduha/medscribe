import os
import os.path as osp
from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from src.rag.retriever import FaissRetriever


@main(config_path="../configs", config_name="rag_build_index", version_base="1.3")
def run(cfg: DictConfig) -> None:
    load_dotenv()

    jsonl = str(cfg.get("jsonl", "datasets/train_prompts.jsonl"))
    out_dir = str(cfg.get("out_dir", "artifacts/rag_index"))
    model = str(cfg.get("model", "intfloat/e5-base-v2"))
    device = cfg.get("device")
    normalize = bool(cfg.get("normalize", True))

    retriever = FaissRetriever(model_name=model, device=device, normalize=normalize)
    count = retriever.build_from_jsonl(jsonl)
    os.makedirs(out_dir, exist_ok=True)
    retriever.save(out_dir)
    print(f"Indexed {count} examples and saved to {out_dir}")


if __name__ == "__main__":
    run()


