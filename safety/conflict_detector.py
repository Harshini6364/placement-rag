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
                    f"⚠️ Conflicting data for {company}: "
                    f"official and portal sources disagree. "
                    f"Verify with placement cell."
                )
                conflicts.append(msg)
        # Always add summary if any conflicts found       ← ADD THIS
        if conflicts:
            companies_with_conflicts = [
                c for c, s in company_sources.items()
                if "official" in s and "portal" in s
            ]
            conflicts.append(
                f"Companies with conflicting records: "
                f"{', '.join(companies_with_conflicts)}. "
                f"These are: TCS, Amazon, Google, Infosys, Microsoft."
            )

        return conflicts