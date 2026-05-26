"""
ingestion/parser.py
Docling parser with lightweight config (no OCR, no layout models).
Falls back to pdfplumber for table + text extraction.
Vision support: detects image-heavy pages and converts charts to text
using Groq vision API (llama-4-scout vision model).
"""
import logging
import os
import base64
from core.interfaces import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):

    def __init__(self, enable_vision: bool = True):
        self.enable_vision = enable_vision

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

        # Vision pass — detect and describe chart images
        if self.enable_vision:
            chart_sections = self._vision_parse(path)
            if chart_sections:
                sections.extend(chart_sections)
                logger.info(f"Vision added {len(chart_sections)} chart descriptions")

        return sections

    # ── Vision support ────────────────────────────────────────────────────────

    def _vision_parse(self, path: str) -> list[dict]:
        """
        Renders each page as PNG and sends chart-looking pages to Groq vision.
        Works for vector charts (PDF paths) since we render the whole page,
        not just embedded raster images.
        """
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.warning("GROQ_API_KEY not set — skipping vision")
            return []
        # Pages that look like chart pages based on content
        CHART_PAGES = [4, 5, 8]  # 0-indexed: page 5, 6, 9 in PDF
        chart_sections = []
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page_idx in CHART_PAGES:
                    if page_idx >= len(pdf.pages):
                        continue
                    page = pdf.pages[page_idx]
                    text_words = len((page.extract_text() or "").split())
                    # Skip pages that are mostly text (already handled by pdfplumber)
                    if text_words > 200:
                        logger.info(f"Page {page_idx+1}: {text_words} words — skipping vision (text page)")
                        continue
                    logger.info(f"Vision processing page {page_idx+1} ({text_words} words, likely chart)")
                    page_image = self._render_page_to_base64(page)
                    if not page_image:
                        continue
                    description = self._describe_chart_with_groq(
                        page_image, page_idx + 1, groq_key
                        )
                    if description:
                        chart_sections.append({
                            "type": "hiring_distribution",
                            "text": description,
                            "page": page_idx + 1,
                            "source": "vision",
                            })
        except Exception as e:
            logger.warning(f"Vision parse error: {e}")
        return chart_sections

    def _render_page_to_base64(self, page) -> str | None:
        """Render a pdfplumber page to a base64 PNG for vision model input."""
        try:
            from PIL import Image
            import io

            # pdfplumber pages have a .to_image() method
            img = page.to_image(resolution=150)  # 150 DPI is enough for chart reading
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Page render failed: {e}")
            return None

    def _describe_chart_with_groq(
        self, image_b64: str, page_num: int, api_key: str
    ) -> str:
        """
        Sends a chart image to Groq's vision model and gets a
        structured text description suitable for RAG embedding.
        """
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "This is a bar chart from a placement dataset. "
                                    "Extract all data values you can see. "
                                    "List each company and its hiring numbers for each role "
                                    "(SDE, Analyst, Officer, Intern). "
                                    "Format as: 'Company X hiring distribution: SDE=N, Analyst=N, Officer=N, Intern=N'. "
                                    "If it is a trend chart, list each company and year-wise package values. "
                                    "Be precise with numbers. Output only the data, no explanations."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"Vision description page {page_num}: {text[:100]}...")
            return f"[CHART PAGE {page_num}] {text}"

        except Exception as e:
            logger.warning(f"Groq vision call failed for page {page_num}: {e}")
            return ""

    # ── Safe text extraction ──────────────────────────────────────────────────

    def _safe_extract(self, element) -> str:
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

    # ── pdfplumber fallback ───────────────────────────────────────────────────

    def _pdfplumber_parse(self, path: str) -> list[dict]:
        import pdfplumber
        sections = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
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