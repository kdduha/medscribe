import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from src.anonymizer.core import Anonymizer
from src.anonymizer.engine import AnonymizationResult

load_dotenv()


def process_file(an: Anonymizer, in_path: Path, out_path: Path) -> AnonymizationResult:
    ext = in_path.suffix.lower()
    if ext == ".txt":
        return an.process_txt(str(in_path), str(out_path))
    elif ext == ".pdf":
        return an.process_pdf(str(in_path), str(out_path))
    elif ext == ".doc":
        return an.process_doc(str(in_path), str(out_path))
    elif ext == ".docx":
        return an.process_docx(str(in_path), str(out_path))
    elif ext == ".rtf":
        return an.process_rtf(str(in_path), str(out_path))
    else:
        print(f"[SKIP] Unsupported file type: {in_path}")
        return AnonymizationResult(text="")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python anonymize.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    an = Anonymizer()
    results = []
    fails = []

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for f in files:
            path = root_path / f
            rel_path = path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".txt")

            print(f"[INFO] Processing {path} -> {out_path}")
            time.sleep(0.2)

            try:
                res = process_file(an, path, out_path)
                results.append(res)
            except Exception as exc:
                print(f"[ERROR] {exc}")
                fails.append((path, exc))

    summary = an.generate_summary_report(results)
    with open(output_dir / "anonymization_summary.txt", "w", encoding="utf-8") as f:
        f.write("Anonymization summary:\n")
        f.write(f"Files processed: {summary['files_processed']}\n")
        f.write(f"Total replacements: {summary['total_replacements']}\n")
        f.write("Counts by label:\n")
        for label, count in summary["counts"].items():
            f.write(f"  {label}: {count}\n")

    with open(output_dir / "fails_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total failed files: {len(fails)}")
        for path, exc in fails:
            f.write(f"- {path} | {exc}")

    print(f"\n[INFO] Done. Summary saved")
