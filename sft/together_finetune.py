import json
import os
import time
from typing import Any, Dict, Optional

from hydra import main
from omegaconf import DictConfig, OmegaConf
from together import Together
from together.utils import check_file


def _get_api_key(cfg: DictConfig) -> str:
    key = os.environ.get("TOGETHER_API_KEY") or str(cfg.get("api_key", ""))
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set and api_key missing in config")
    return key


def _upload_if_path(client, path_or_id: str) -> str:
    # If it already looks like a Together file id, pass-through
    if isinstance(path_or_id, str) and path_or_id.startswith("file-"):
        return path_or_id
    # Otherwise upload local file
    resp = client.files.upload(file=path_or_id)
    return resp.id if hasattr(resp, "id") else resp["id"]


def _inject_api_params_from_cfg(params: Dict[str, Any], cfg: DictConfig) -> None:
    # Top-level numeric/string params per API
    if cfg.training.get("epochs") is not None:
        params["n_epochs"] = int(cfg.training.epochs)
    if cfg.job.get("n_checkpoints") is not None:
        params["n_checkpoints"] = int(cfg.job.n_checkpoints)
    if cfg.job.get("n_evals") is not None:
        params["n_evals"] = int(cfg.job.n_evals)
    if cfg.training.get("batch_size") is not None:
        params["batch_size"] = cfg.training.batch_size
    if cfg.training.get("learning_rate") is not None:
        params["learning_rate"] = float(cfg.training.learning_rate)
    # if cfg.training.get("lr_scheduler_type"):
        # params["lr_scheduler"] = {"type": str(cfg.training.lr_scheduler_type)}
    if cfg.training.get("warmup_ratio") is not None:
        params["warmup_ratio"] = float(cfg.training.warmup_ratio)
    if cfg.training.get("max_grad_norm") is not None:
        params["max_grad_norm"] = float(cfg.training.max_grad_norm)
    if cfg.training.get("weight_decay") is not None:
        params["weight_decay"] = float(cfg.training.weight_decay)
    if cfg.job.get("suffix"):
        params["suffix"] = str(cfg.job.suffix)
    # Optional W&B passthroughs
    if cfg.get("wandb", {}).get("api_key"):
        params["wandb_api_key"] = str(cfg.wandb.api_key)
    if cfg.get("wandb", {}).get("project"):
        params["wandb_project_name"] = str(cfg.wandb.project)
    if cfg.get("wandb", {}).get("name"):
        params["wandb_name"] = str(cfg.wandb.name)
    if cfg.get("wandb", {}).get("base_url"):
        params["wandb_base_url"] = str(cfg.wandb.base_url)
    # Resume/continue options
    if cfg.job.get("from_checkpoint"):
        params["from_checkpoint"] = str(cfg.job.from_checkpoint)
    if cfg.job.get("from_hf_model"):
        params["from_hf_model"] = str(cfg.job.from_hf_model)
    if cfg.job.get("hf_model_revision"):
        params["hf_model_revision"] = str(cfg.job.hf_model_revision)
    if cfg.job.get("hf_api_token"):
        params["hf_api_token"] = str(cfg.job.hf_api_token)
    if cfg.job.get("hf_output_repo_name"):
        params["hf_output_repo_name"] = str(cfg.job.hf_output_repo_name)

    wandb = cfg.get("wandb")
    if wandb and wandb.get("enabled", True):
        params["wandb_api_key"] = wandb.get("api_key", None) or os.environ.get("WANDB_API_KEY")
        if wandb.get("base_url"):
            params["wandb_base_url"] = str(wandb.base_url)
        params["wandb_project_name"] = str(wandb.project)
        params["wandb_name"] = str(wandb.name)

    lora = cfg.get("lora")
    if lora and lora.get("enabled", True):
        params["lora"] = "true"
        params["lora_r"] = int(lora.get("r", 64))
        params["lora_alpha"] = int(lora.get("alpha", 128))
        params["lora_dropout"] = float(lora.get("dropout", 0.1))
        params["lora_trainable_modules"] = lora.get("target_modules")
        
    elif cfg.job.get("training_type"):
        # allow forcing full training or other type via config
        params["training_type"] = dict(cfg.job.training_type)


def _maybe_wait_for_completion(client, job_id: str, poll_seconds: int = 15) -> Dict[str, Any]:
    while True:
        job = client.fine_tuning.retrieve(job_id)
        status = getattr(job, "status", None) or job.get("status")
        if status in {"completed", "error", "cancelled"}:
            return job
        time.sleep(poll_seconds)


@main(config_path="configs", config_name="together", version_base="1.3")
def run(cfg: DictConfig) -> None:
    # print("Config:\n" + OmegaConf.to_yaml(cfg))
    api_key = _get_api_key(cfg)
    client = Together(api_key=api_key)

    # Upload or reuse files
    train_file = _upload_if_path(client, str(cfg.data.train_path))
    val_file = None
    if cfg.data.get("eval_path"):
        val_file = _upload_if_path(client, str(cfg.data.eval_path))

    # sft_report = check_file(train_file)
    # print(json.dumps(sft_report, indent=2))
    # assert sft_report["is_check_passed"] == True

    params = {
        "model": str(cfg.model.name_or_path),
        "training_file": train_file,
    }
    if val_file:
        params["validation_file"] = val_file
    # Populate API params from config (flattened per API schema)
    _inject_api_params_from_cfg(params, cfg)

    # Create job
    print(params)
    job = client.fine_tuning.create(**params)
    job_id = getattr(job, "id", None) or job.get("id")
    output_name = getattr(job, "output_name", None) or job.get("output_name")
    print(f"Started Together fine-tuning job: {job_id}")
    if output_name:
        print(f"Output model (when finished): {output_name}")

    if bool(cfg.job.get("wait", True)):
        print("Waiting for job completion...")
        job = _maybe_wait_for_completion(client, job_id)
        print(f"Job finished with status: {getattr(job, 'status', None) or job.get('status')}")
        output_name = getattr(job, "output_name", None) or job.get("output_name")
        if output_name:
            print(f"Final output model: {output_name}")

    # Persist job metadata if requested
    out_meta = cfg.job.get("save_job_json")
    if out_meta:
        os.makedirs(os.path.dirname(out_meta) or ".", exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "output_name": output_name}, f, ensure_ascii=False, indent=2)
        print(f"Saved job metadata to {out_meta}")


if __name__ == "__main__":
    run()


