"""
ingestion/parser.py
Docling-based PDF parser with pdfplumber table fallback.
"""
import logging
from pathlib import Path
from core.interfaces import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """
    Primary parser using Docling for layout-aware extraction.
    Falls back to pdfplumber for table-heavy pages.
    """

    def parse(self, path: str) -> list[dict]:
        sections = []
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(path)
            doc = result.document

            for element in doc.body.children:
                text = element.export_to_text().strip()
                if text:
                    sections.append({
                        "type": self._classify(text),
                        "text": text,
                        "page": getattr(element, "page_no", 0),
                    })
            logger.info(f"Docling parsed {len(sections)} sections from {path}")
        except Exception as e:
            logger.warning(f"Docling failed ({e}), falling back to pdfplumber")
            sections = self._pdfplumber_parse(path)

        return sections

    def _pdfplumber_parse(self, path: str) -> list[dict]:
        import pdfplumber
        sections = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract tables as structured dicts
                for table in page.extract_tables():
                    if table and len(table) > 1:
                        headers = table[0]
                        for row in table[1:]:
                            row_dict = dict(zip(headers, row))
                            sections.append({
                                "type": "table_row",
                                "text": self._row_to_text(row_dict),
                                "page": i + 1,
                                "raw_row": row_dict,
                            })
                # Extract remaining text
                text = page.extract_text()
                if text:
                    sections.append({
                        "type": "text",
                        "text": text.strip(),
                        "page": i + 1,
                    })
        logger.info(f"pdfplumber parsed {len(sections)} sections")
        return sections

    def _row_to_text(self, row: dict) -> str:
        """Convert a table row dict to natural language for embedding."""
        parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
        return " | ".join(parts)

    def _classify(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["min cgpa", "package", "bond", "backlogs"]):
            return "eligibility_table"
        if any(w in t for w in ["round 1", "round 2", "technical interview", "hr round"]):
            return "interview_experience"
        if any(w in t for w in ["2021", "2022", "2023", "2024", "lpa", "trend"]):
            return "temporal_trend"
        if any(w in t for w in ["conflicting", "portal", "official", "cgpa (official)"]):
            return "conflict_record"
        if any(w in t for w in ["sde", "analyst", "officer", "intern", "hiring"]):
            return "hiring_distribution"
        if any(w in t for w in ["adversarial", "out-of-corpus", "fallback"]):
            return "adversarial"
        return "text"