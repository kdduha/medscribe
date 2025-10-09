import os
import os.path as osp
from typing import Iterable

from hydra import main
from omegaconf import DictConfig

from src.data_preparation.download_processed_from_drive import run as run_download
from src.data_preparation.compile_datasets import run as run_compile
from src.data_preparation.compile_prompts import run as run_prompts


def _normalize_steps(steps: Iterable[str]) -> list:
    return [s.strip().lower() for s in steps if s and str(s).strip()]


@main(config_path="../configs", config_name="pipeline", version_base="1.3")
def run(cfg: DictConfig) -> None:
    merged: DictConfig = cfg

    steps = _normalize_steps(merged.get("steps", ["download", "compile", "prompts"]))
    run_download_step = "download" in steps
    run_compile_step = "compile" in steps
    run_prompts_step = "prompts" in steps

    if run_download_step:
        print("=== Step 1: Download processed CSVs (optional) ===")
        num_downloaded = run_download(merged.data)
        print(f"Downloaded: {num_downloaded}")

    if run_compile_step:
        print("\n=== Step 2: Compile datasets into a single CSV ===")
        num_rows = run_compile(merged.compile)
        print(f"Rows written: {num_rows}")

    if run_prompts_step:
        print("\n=== Step 3: Build training prompts ===")
        num_prompts = run_prompts(merged.prompts)
        print(f"Prompts saved: {num_prompts}")


if __name__ == "__main__":
    run()

# import os
# import os.path as osp
# from typing import Any, Iterable

# from hydra import main
# from omegaconf import DictConfig, OmegaConf

# from src.data_preparation.download_processed_from_drive import run as run_download
# from src.data_preparation.compile_datasets import run as run_compile
# from src.data_preparation.compile_prompts import run as run_prompts


# def _normalize_steps(steps: Iterable[str]) -> list:
#     return [s.strip().lower() for s in steps if s and str(s).strip()]


# @main(config_path="../configs", config_name="pipeline", version_base="1.3")
# def run(cfg: DictConfig) -> None:
#     merged: DictConfig = cfg

#     steps = _normalize_steps(merged.get("steps", ["download", "compile", "prompts"]))
#     run_download_step = "download" in steps
#     run_compile_step = "compile" in steps
#     run_prompts_step = "prompts" in steps

#     if run_download_step:
#         print("=== Step 1: Download processed CSVs (optional) ===")
#         num_downloaded = run_download(merged.data)
#         print(f"Downloaded: {num_downloaded}")

#     if run_compile_step:
#         print("\n=== Step 2: Compile datasets into a single CSV ===")
#         num_rows = run_compile(merged.compile)
#         print(f"Rows written: {num_rows}")

#     if run_prompts_step:
#         print("\n=== Step 3: Build training prompts ===")
#         num_prompts = run_prompts(merged.prompts)
#         print(f"Prompts saved: {num_prompts}")


# if __name__ == "__main__":
#     run()


