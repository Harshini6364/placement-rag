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

from ingestion.parser import DoclingParser
from ingestion.chunker import PlacementChunker
from ingestion.deduplicator import TFIDFDeduplicator
from ingestion.embedder import HybridEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_PATH = "data/Placement_RAG_Dataset_Enhanced.pdf"


def main():
    logger.info("=== INGESTION PIPELINE START ===")

    # Parse
    parser = DoclingParser()
    sections = parser.parse(PDF_PATH)
    logger.info(f"Parsed {len(sections)} sections")

    # Chunk
    chunker = PlacementChunker()
    chunks = chunker.chunk(sections)
    logger.info(f"Created {len(chunks)} chunks")

    # Deduplicate
    deduper = TFIDFDeduplicator(threshold=0.85)
    chunks = deduper.deduplicate(chunks)
    logger.info(f"After dedup: {len(chunks)} chunks")

    # Embed + Index
    embedder = HybridEmbedder(
        model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        bm25_path=os.getenv("BM25_PATH", "data/bm25_store.pkl"),
        chunks_path=os.getenv("CHUNKS_PATH", "data/chunks.pkl"),
    )
    embedder.index(chunks)

    logger.info("=== INGESTION COMPLETE ===")
    logger.info(f"Final chunk count: {len(chunks)} (target: 80-120)")


if __name__ == "__main__":
    main()