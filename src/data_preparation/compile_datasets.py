import csv
import os
import os.path as osp
from typing import Dict, Iterable, List, Set

from hydra.utils import to_absolute_path


TARGET_FILENAMES = {"processed_result.csv", "processed_results.csv"}


def find_processed_csvs(root: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower() in TARGET_FILENAMES:
                matches.append(osp.join(dirpath, name))
    return matches


def read_rows_with_origin(path: str) -> Iterable[Dict[str, str]]:
    origin_folder = osp.basename(osp.dirname(path))
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row) if row is not None else {}
            row["origin_folder"] = origin_folder
            yield row


def collect_all_fieldnames(files: List[str]) -> List[str]:
    fieldnames_set: Set[str] = set(["origin_folder"])  # ensure column order includes origin
    for path in files:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name and name.strip():
                        fieldnames_set.add(name)
    # Prefer to place origin_folder first, then the rest in stable sorted order
    other = sorted([n for n in fieldnames_set if n != "origin_folder"])
    return ["origin_folder"] + other


def write_combined_csv(files: List[str], output_csv: str) -> int:
    if not files:
        # ensure directory exists even if nothing to write
        os.makedirs(osp.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return 0

    fieldnames = collect_all_fieldnames(files)
    os.makedirs(osp.dirname(output_csv) or ".", exist_ok=True)
    written = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path in files:
            for row in read_rows_with_origin(path):
                full_row = {name: row.get(name, "") for name in fieldnames}
                writer.writerow(full_row)
                written += 1
    return written


def run(cfg) -> int:
    input_root = to_absolute_path(str(cfg.get("input_root", "datasets")))
    output_csv = to_absolute_path(str(cfg.get("output_csv", "datasets/compiled.csv")))

    files = find_processed_csvs(input_root)
    count = write_combined_csv(files, output_csv)
    print(f"Found {len(files)} CSV files. Wrote {count} rows to {output_csv}.")
    return count


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pipeline.py to run this module via Hydra.")


