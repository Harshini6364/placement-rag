"""
ingestion/parser.py
Docling parser with lightweight config (no OCR, no layout models).
Falls back to pdfplumber for table extraction.
"""
import logging
from core.interfaces import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):

    def parse(self, path: str) -> list[dict]:
        sections = []
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = False
            pipeline_options.generate_picture_images = False

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            result = converter.convert(path)
            doc = result.document

            for element, _level in doc.iterate_items():
                text = self._safe_extract(element)
                if text and len(text) > 20:
                    sections.append({
                        "type": self._classify(text),
                        "text": text,
                        "page": 0,
                    })

            if not sections:
                raise ValueError("Docling returned 0 usable sections")

            logger.info(f"Docling parsed {len(sections)} sections")

        except Exception as e:
            logger.warning(f"Docling failed ({e}), using pdfplumber fallback")
            sections = self._pdfplumber_parse(path)

        return sections

    def _safe_extract(self, element) -> str:
        """Try every known text-export method across Docling versions."""
        for method in ["export_to_markdown", "export_to_text"]:
            if hasattr(element, method):
                try:
                    text = getattr(element, method)()
                    if text and text.strip():
                        return text.strip()
                except Exception:
                    continue
        if hasattr(element, "text") and element.text:
            return str(element.text).strip()
        if hasattr(element, "content") and element.content:
            return str(element.content).strip()
        try:
            return str(element).strip()
        except Exception:
            return ""

    def _pdfplumber_parse(self, path: str) -> list[dict]:
        import pdfplumber
        sections = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract tables first (structured data)
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue
                    raw_headers = table[0]
                    headers = [
                        str(h).strip() if h else f"col{j}"
                        for j, h in enumerate(raw_headers)
                    ]
                    for row in table[1:]:
                        row_dict = {
                            headers[j]: str(v).strip() if v else ""
                            for j, v in enumerate(row)
                            if j < len(headers)
                        }
                        text = self._row_to_text(row_dict)
                        if text:
                            sections.append({
                                "type": "table_row",
                                "text": text,
                                "page": i + 1,
                                "raw_row": row_dict,
                            })
                # Extract free text
                text = page.extract_text()
                if text and text.strip():
                    sections.append({
                        "type": self._classify(text),
                        "text": text.strip(),
                        "page": i + 1,
                    })
        logger.info(f"pdfplumber parsed {len(sections)} sections")
        return sections

    def _row_to_text(self, row: dict) -> str:
        return " | ".join(
            f"{k}: {v}" for k, v in row.items() if v and v.strip()
        )

    def _classify(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["min cgpa", "package", "bond", "backlogs"]):
            return "eligibility_table"
        if any(w in t for w in ["round 1", "round 2", "technical interview", "hr round"]):
            return "interview_experience"
        if any(w in t for w in ["2021", "2022", "2023", "2024", "trend"]):
            return "temporal_trend"
        if any(w in t for w in ["conflicting", "portal", "official source", "cgpa (official)"]):
            return "conflict_record"
        if any(w in t for w in ["sde", "analyst", "officer", "intern", "hiring distribution"]):
            return "hiring_distribution"
        if any(w in t for w in ["adversarial", "out-of-corpus", "fallback"]):
            return "adversarial"
        return "text"