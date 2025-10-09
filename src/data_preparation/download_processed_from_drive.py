import os
import os.path as osp
from typing import List

import gdown
from hydra.utils import to_absolute_path


def list_drive_tree(url: str, output_root: str):
    files = gdown.download_folder(
        url=url,
        output=output_root,
        quiet=True,
        skip_download=True,
        remaining_ok=True,
    )
    return files or []


def ensure_parent_dir(path: str) -> None:
    parent = osp.dirname(path)
    if parent and not osp.exists(parent):
        os.makedirs(parent, exist_ok=True)


def download_selected(files, patterns: List[str], output_root: str, resume: bool) -> List[str]:
    normalized_patterns = [p.lower() for p in patterns]
    downloaded: List[str] = []

    for item in files:
        filename = osp.basename(item.local_path)
        if filename.lower() not in normalized_patterns:
            continue

        target_path = item.local_path
        ensure_parent_dir(target_path)

        local_path = gdown.download(
            url=f"https://drive.google.com/uc?id={item.id}",
            output=target_path,
            quiet=False,
            use_cookies=True,
            resume=resume,
        )
        if local_path:
            downloaded.append(local_path)

    return downloaded


def run(cfg) -> int:
    url = str(cfg.get("drive_folder", "")).strip()
    output_root = to_absolute_path(str(cfg.get("output_root", "datasets")))
    resume = bool(cfg.get("resume", True))
    patterns = [str(p) for p in (cfg.get("patterns") or ["processed_result.csv", "processed_results.csv"])]

    if not url:
        print("No drive_folder provided; skipping download step.")
        return 0

    if not url.startswith("http"):
        url = f"https://drive.google.com/drive/folders/{url}"

    files = list_drive_tree(url=url, output_root=output_root)
    downloaded = download_selected(files=files, patterns=patterns, output_root=output_root, resume=resume)

    print(f"Downloaded {len(downloaded)} files to {output_root}:")
    for p in downloaded:
        print(p)
    return len(downloaded)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pipeline.py to run this module via Hydra.")


