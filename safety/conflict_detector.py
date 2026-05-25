"""
safety/conflict_detector.py
Detects conflicting information across retrieved chunks.
"""
import logging
from core.interfaces import BaseConflictDetector, Chunk

logger = logging.getLogger(__name__)


class PlacementConflictDetector(BaseConflictDetector):
    """
    Checks if retrieved chunks include both 'official' and 'portal'
    sources for the same company — a definitive conflict signal.
    """

    def detect(self, chunks: list[Chunk]) -> list[str]:
        conflicts = []
        company_sources: dict[str, set] = {}

        for chunk in chunks:
            if chunk.conflict and chunk.company:
                company_sources.setdefault(chunk.company, set()).add(chunk.source)

        for company, sources in company_sources.items():
            if "official" in sources and "portal" in sources:
                msg = (
                    f"⚠️ Conflicting data detected for {company}: "
                    f"official and portal sources disagree. "
                    f"Please verify with the official placement cell."
                )
                conflicts.append(msg)
                logger.warning(f"Conflict: {company}")

        return conflicts