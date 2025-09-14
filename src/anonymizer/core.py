import os
import logging
import subprocess
from src.anonymizer.engine import RegexEngine, OpenAIEngine, AnonymizationResult
from src.logger import setup_logger

setup_logger()
LOG = logging.getLogger(__name__)


class Anonymizer:
    ENGINE_CLASSES = {
        "openai": OpenAIEngine,
        "regex": RegexEngine,
    }

    def __init__(self, engines: list[tuple[str, dict[str, any]]] | None = None):
        if engines is None:
            self.engines = {name: cls() for name, cls in self.ENGINE_CLASSES.items()}
        else:
            self.engines = {name: self.ENGINE_CLASSES[name](**params) for name, params in engines}

    def anonymize(self, text: str) -> AnonymizationResult:
        result = AnonymizationResult(text=text)

        for name, engine in self.engines.items():
            LOG.info("Start anon with engine %s", name)
            res = engine.anonymize(result.text)
            result.replacements.extend(res.replacements)
            result.text = res.text
            LOG.info("End anon with engine %s", name)

        return result

    @staticmethod
    def generate_summary_report(results: list[AnonymizationResult]) -> dict[str, any]:
        total_summary: dict[str, int] = {}
        total_replacements = 0

        for res in results:
            s = res.to_summary()
            total_replacements += s.get("total_replacements", 0)
            for label, count in s.get("counts", {}).items():
                total_summary[label] = total_summary.get(label, 0) + count

        return {"counts": total_summary, "total_replacements": total_replacements, "files_processed": len(results)}

    # ---------- high-level file processors ----------

    def process_txt(self, in_path: str, out_txt_path: str, anonymize_before_write: bool = True) -> AnonymizationResult:
        text = self._read_txt(in_path)
        return self._process_text_and_write(text, out_txt_path, anonymize_before_write)

    def process_docx(self, in_path: str, out_txt_path: str, anonymize_before_write: bool = True) -> AnonymizationResult:
        text = self._read_docx(in_path)
        return self._process_text_and_write(text, out_txt_path, anonymize_before_write)

    def process_pdf(self, in_path: str, out_txt_path: str, anonymize_before_write: bool = True) -> AnonymizationResult:
        text = self._read_pdf(in_path)
        return self._process_text_and_write(text, out_txt_path, anonymize_before_write)

    def process_rtf(self, in_path: str, out_txt_path: str, anonymize_before_write: bool = True) -> AnonymizationResult:
        text = self._read_rtf(in_path)
        return self._process_text_and_write(text, out_txt_path, anonymize_before_write)

    # ---------- internals ----------

    def _process_text_and_write(self, text: str, out_txt_path: str, anonymize_before_write: bool) -> AnonymizationResult:
        if anonymize_before_write:
            res = self.anonymize(text)
        else:
            res = AnonymizationResult(text=text)

        self._write_txt(out_txt_path, res.text)
        return res

    @staticmethod
    def _ensure_dir(path: str) -> None:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _read_txt(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def _read_docx(path: str) -> str:
        try:
            import docx
        except Exception as e:
            raise ImportError("python-docx is required for .docx support: pip install python-docx") from e

        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            from PyPDF2 import PdfReader
        except Exception as e:
            raise ImportError("PyPDF2 is required for .pdf support: pip install PyPDF2") from e
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            text = p.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    @staticmethod
    def _read_rtf(path: str) -> str:
        try:
            import pypandoc
        except Exception as e:
            raise ImportError("pypandoc is required for .rtf support: pip install pypandoc") from e
        try:
            result = subprocess.run(
                ["unrtf", "--html", path],
                capture_output=True,
                text=True,
                check=True
            )
            html_text = result.stdout

            # HTML → plain text
            # https://qna.habr.com/q/703118?ysclid=mfk4jlw1m3877812478
            text = pypandoc.convert_text(html_text, "plain", format="html")
            return text

        except subprocess.CalledProcessError as exc:
            LOG.exception("Failed to convert RTF via unrtf: %s", exc)
        except Exception as exc:
            LOG.exception("Unexpected error reading RTF %s: %s", path, exc)

    def _write_txt(self, path: str, text: str) -> None:
        self._ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
