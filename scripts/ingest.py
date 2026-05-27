"""
scripts/ingest.py
Run once to parse, chunk, deduplicate, and index the PDF.
"""
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from version import print_banner
print_banner()

from ingestion.parser import DoclingParser
from ingestion.chunker import PlacementChunker
from ingestion.deduplicator import TFIDFDeduplicator
from ingestion.embedder import HybridEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

PDF_PATH = "data/Placement_RAG_Dataset_Enhanced.pdf"


def main():
    logger.info("=== INGESTION PIPELINE START ===")

    parser   = DoclingParser(enable_vision=True)
    sections = parser.parse(PDF_PATH)
    logger.info(
        f"Parsed {len(sections)} sections "
        f"({sum(1 for s in sections if s.get('source') == 'vision')} vision)"
    )

    chunker = PlacementChunker()
    chunks  = chunker.chunk(sections)
    logger.info(f"Created {len(chunks)} chunks")

    deduper = TFIDFDeduplicator(threshold=0.85)
    chunks  = deduper.deduplicate(chunks)
    logger.info(f"After dedup: {len(chunks)} chunks")

    from collections import Counter
    section_counts = Counter(c.section for c in chunks)
    logger.info(f"Chunk breakdown: {dict(section_counts)}")

    embedder = HybridEmbedder(
        model_name  =os.getenv("EMBED_MODEL",        "all-MiniLM-L6-v2"),
        faiss_path  =os.getenv("FAISS_INDEX_PATH",   "data/faiss_index"),
        bm25_path   =os.getenv("BM25_PATH",          "data/bm25_store.pkl"),
        chunks_path =os.getenv("CHUNKS_PATH",        "data/chunks.pkl"),
    )
    embedder.build_index(chunks)

    logger.info("=== INGESTION COMPLETE ===")
    target_ok = 80 <= len(chunks) <= 150
    logger.info(
        f"Final chunk count: {len(chunks)} "
        f"({'target: 80-150 ✓' if target_ok else 'outside target range'})"
    )


if __name__ == "__main__":
    main()