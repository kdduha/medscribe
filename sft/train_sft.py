import os
from typing import Any, Dict, Optional

from hydra import main
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def _maybe_init_wandb(cfg: DictConfig) -> None:
    enabled = bool(cfg.get("wandb", {}).get("enabled", False))
    if enabled:
        import wandb
        os.environ["WANDB_DISABLED"] = "false"
        if cfg.wandb.get("project"):
            os.environ["WANDB_PROJECT"] = str(cfg.wandb.project)
        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = str(cfg.wandb.entity)
    else:
        os.environ["WANDB_DISABLED"] = "true"


def _load_splits(data_cfg: DictConfig) -> Dict[str, Dataset]:
    out: Dict[str, Dataset] = {}
    if data_cfg.get("hf_dir"):
        dsd = load_from_disk(str(data_cfg.hf_dir))
        # Align keys
        if "train" in dsd:
            out["train"] = dsd["train"]
        if "eval" in dsd:
            out["eval"] = dsd["eval"]
    else:
        if data_cfg.get("train_path"):
            ds = load_dataset("json", data_files=str(data_cfg.train_path))
            out["train"] = ds["train"]
        if data_cfg.get("eval_path"):
            ds_eval = load_dataset("json", data_files=str(data_cfg.eval_path))
            out["eval"] = ds_eval["train"]
    return out


def _format_dataset(ds: Dataset, input_field: str, target_field: str, template: str, use_messages: bool) -> Dataset:
    def builder(example: Dict[str, Any]) -> Dict[str, Any]:
        if use_messages and "messages" in example:
            return {"messages": example["messages"]}
        prompt = str(example.get(input_field, "")).strip()
        response = str(example.get(target_field, "")).strip()
        text = template.format(prompt=prompt, response=response)
        return {"text": text}

    # keep only examples that have both fields
    if use_messages:
        ds = ds.filter(lambda e: "messages" in e and e["messages"] is not None)
        return ds.map(builder, remove_columns=[c for c in ds.column_names if c != "messages"])
    ds = ds.filter(lambda e: input_field in e and target_field in e and e[input_field] is not None and e[target_field] is not None)
    return ds.map(builder, remove_columns=[c for c in ds.column_names if c != "text"])


def _build_lora(cfg: DictConfig) -> Optional[LoraConfig]:
    lora = cfg.get("lora")
    if not lora or not lora.get("enabled", True):
        return None
    target_modules = lora.get("target_modules")
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(target_modules) if target_modules else None,
        bias="none",
        task_type="CAUSAL_LM",
    )


@main(config_path="configs", config_name="sft", version_base="1.3")
def run(cfg: DictConfig) -> None:
    _maybe_init_wandb(cfg)
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # tokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        use_fast=bool(cfg.model.get("use_fast", True)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = cfg.model.get("padding_side", "right")

    # model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )

    # data
    splits = _load_splits(cfg.data)
    use_messages = bool(cfg.data.get("hf_dir"))  # HF dir produced by prepare_data contains messages
    input_field = str(cfg.data.get("input_field", "prompt_text"))
    target_field = str(cfg.data.get("target_field", "result_text"))
    template = str(cfg.data.get("template", "{prompt}\n{response}"))

    train_ds = _format_dataset(splits["train"], input_field, target_field, template, use_messages) if "train" in splits else None
    eval_ds = _format_dataset(splits["eval"], input_field, target_field, template, use_messages) if "eval" in splits else None
    if train_ds is None:
        raise ValueError("data.train_path is required")

    # LoRA
    peft_cfg = _build_lora(cfg)

    # training
    training_args = SFTConfig(**cfg.training)
    # If messages available and tokenizer supports chat templates, pass dataset_text_field accordingly
    dataset_text_field = "messages" if use_messages else "text"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field=dataset_text_field,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    tok.save_pretrained(cfg.training.output_dir)


if __name__ == "__main__":
    run()


