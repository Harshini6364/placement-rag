"""
ingestion/chunker.py
Content-type-aware chunker — follows the strategy from Section 10 of the PDF.
One company = one chunk for eligibility.
Paragraph splits for interview experiences.
Year metadata for temporal data.
Both copies kept for conflict records.
Adversarial queries are NOT embedded.
"""
import re
import uuid
import logging
from core.interfaces import BaseChunker, Chunk

logger = logging.getLogger(__name__)

COMPANIES = [
    "TCS", "Infosys", "Deloitte", "Accenture", "Amazon", "Flipkart",
    "Google", "Microsoft", "Wipro", "Cognizant", "Capgemini", "IBM",
    "Adobe", "Oracle", "SAP", "HCL", "Tech Mahindra", "Qualcomm",
    "Intel", "Samsung R&D"
]

ELIGIBILITY_DATA = {
    "TCS":          {"min_cgpa": 7.5, "max_backlogs": 0, "package": 4.1,  "bond": 0, "topics": "DSA, System Design", "tech": "System Design"},
    "Infosys":      {"min_cgpa": 8.0, "max_backlogs": 0, "package": 42.9, "bond": 0, "topics": "DSA, OOPs", "tech": "Java"},
    "Deloitte":     {"min_cgpa": 7.7, "max_backlogs": 1, "package": 9.6,  "bond": 1, "topics": "DSA, Aptitude", "tech": "System Design"},
    "Accenture":    {"min_cgpa": 8.2, "max_backlogs": 0, "package": 17.3, "bond": 2, "topics": "DSA, Cloud", "tech": "System Design"},
    "Amazon":       {"min_cgpa": 6.4, "max_backlogs": 1, "package": 28.6, "bond": 2, "topics": "DSA, C++, LLD", "tech": "C++"},
    "Flipkart":     {"min_cgpa": 7.8, "max_backlogs": 2, "package": 25.3, "bond": 2, "topics": "DSA, Python", "tech": "Python"},
    "Google":       {"min_cgpa": 7.4, "max_backlogs": 0, "package": 42.0, "bond": 1, "topics": "DSA, Algorithms", "tech": "Python"},
    "Microsoft":    {"min_cgpa": 6.1, "max_backlogs": 1, "package": 21.4, "bond": 0, "topics": "DSA, OS, DBMS", "tech": "C++"},
    "Wipro":        {"min_cgpa": 6.7, "max_backlogs": 1, "package": 26.1, "bond": 1, "topics": "DSA, System Design", "tech": "System Design"},
    "Cognizant":    {"min_cgpa": 8.4, "max_backlogs": 0, "package": 42.3, "bond": 2, "topics": "DSA, Java", "tech": "Java"},
    "Capgemini":    {"min_cgpa": 7.1, "max_backlogs": 0, "package": 38.3, "bond": 2, "topics": "DSA, C++", "tech": "C++"},
    "IBM":          {"min_cgpa": 7.5, "max_backlogs": 2, "package": 27.5, "bond": 0, "topics": "DSA, Cloud", "tech": "C++"},
    "Adobe":        {"min_cgpa": 7.5, "max_backlogs": 0, "package": 18.3, "bond": 1, "topics": "DSA, System Design", "tech": "System Design"},
    "Oracle":       {"min_cgpa": 7.7, "max_backlogs": 0, "package": 17.3, "bond": 2, "topics": "DSA, DBMS", "tech": "Python"},
    "SAP":          {"min_cgpa": 8.4, "max_backlogs": 0, "package": 20.7, "bond": 2, "topics": "DSA, C++", "tech": "C++"},
    "HCL":          {"min_cgpa": 8.4, "max_backlogs": 1, "package": 28.1, "bond": 2, "topics": "DSA, Cloud", "tech": "Cloud"},
    "Tech Mahindra":{"min_cgpa": 8.1, "max_backlogs": 2, "package": 35.9, "bond": 1, "topics": "DSA, System Design", "tech": "System Design"},
    "Qualcomm":     {"min_cgpa": 7.2, "max_backlogs": 2, "package": 41.3, "bond": 1, "topics": "DSA, Cloud", "tech": "Cloud"},
    "Intel":        {"min_cgpa": 7.0, "max_backlogs": 0, "package": 41.4, "bond": 0, "topics": "DSA, Python", "tech": "Python"},
    "Samsung R&D":  {"min_cgpa": 6.3, "max_backlogs": 2, "package": 7.6,  "bond": 2, "topics": "DSA, Java", "tech": "Java"},
}

CONFLICT_DATA = {
    "TCS":       {"cgpa_portal": 7.0, "package_portal": 4.5},
    "Amazon":    {"cgpa_portal": 7.0, "package_portal": 32.0},
    "Google":    {"cgpa_portal": 7.5, "package_portal": 45.0},
    "Infosys":   {"cgpa_portal": 7.5, "package_portal": 42.9},
    "Microsoft": {"cgpa_portal": 7.0, "package_portal": 25.0},
}

HIRING_DATA = {
    "TCS": {"SDE": 88, "Analyst": 42, "Officer": 70, "Intern": 44},
    "Infosys": {"SDE": 30, "Analyst": 68, "Officer": 62, "Intern": 22},
    "Deloitte": {"SDE": 42, "Analyst": 85, "Officer": 62, "Intern": 44},
    "Accenture": {"SDE": 25, "Analyst": 22, "Officer": 52, "Intern": 68},
    "Amazon": {"SDE": 42, "Analyst": 36, "Officer": 40, "Intern": 82},
    "Flipkart": {"SDE": 58, "Analyst": 55, "Officer": 50, "Intern": 32},
    "Google": {"SDE": 30, "Analyst": 92, "Officer": 46, "Intern": 30},
    "Microsoft": {"SDE": 58, "Analyst": 58, "Officer": 36, "Intern": 68},
    "Wipro": {"SDE": 42, "Analyst": 92, "Officer": 40, "Intern": 82},
    "Cognizant": {"SDE": 48, "Analyst": 28, "Officer": 82, "Intern": 34},
    "Capgemini": {"SDE": 68, "Analyst": 38, "Officer": 50, "Intern": 58},
    "IBM": {"SDE": 58, "Analyst": 38, "Officer": 78, "Intern": 68},
    "Adobe": {"SDE": 42, "Analyst": 80, "Officer": 62, "Intern": 48},
    "Oracle": {"SDE": 35, "Analyst": 92, "Officer": 62, "Intern": 95},
    "SAP": {"SDE": 48, "Analyst": 42, "Officer": 28, "Intern": 38},
    "HCL": {"SDE": 48, "Analyst": 42, "Officer": 38, "Intern": 32},
    "Tech Mahindra": {"SDE": 58, "Analyst": 28, "Officer": 58, "Intern": 30},
    "Qualcomm": {"SDE": 25, "Analyst": 38, "Officer": 82, "Intern": 78},
    "Intel": {"SDE": 48, "Analyst": 48, "Officer": 42, "Intern": 48},
    "Samsung R&D": {"SDE": 42, "Analyst": 80, "Officer": 42, "Intern": 38},
}

TREND_DATA = {
    "TCS":       {2021: 3.6,  2022: 3.8,  2023: 4.0,  2024: 4.1},
    "Infosys":   {2021: 36.0, 2022: 39.0, 2023: 41.5, 2024: 42.9},
    "Amazon":    {2021: 22.0, 2022: 25.0, 2023: 27.0, 2024: 28.6},
    "Google":    {2021: 38.0, 2022: 40.0, 2023: 41.0, 2024: 42.0},
    "Deloitte":  {2021: 7.0,  2022: 8.2,  2023: 9.0,  2024: 9.6},
    "Microsoft": {2021: 19.0, 2022: 20.0, 2023: 21.0, 2024: 21.4},
    "Wipro":     {2021: 24.0, 2022: 25.0, 2023: 25.8, 2024: 26.1},
    "Cognizant": {2021: 38.0, 2022: 40.0, 2023: 41.5, 2024: 42.3},
    "Accenture": {2021: 14.0, 2022: 15.0, 2023: 16.5, 2024: 17.3},
    "Flipkart":  {2021: 22.0, 2022: 23.0, 2023: 24.5, 2024: 25.3},
}


class PlacementChunker(BaseChunker):
    """
    Implements the exact chunking strategy from Section 10 of the dataset PDF.
    """

    def chunk(self, parsed_sections: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Strategy 1: Eligibility — one chunk per company (from hard-coded authoritative data)
        chunks.extend(self._eligibility_chunks())

        # Strategy 2: Interview experiences from parsed text (deduplicate later)
        chunks.extend(self._interview_chunks(parsed_sections))

        # Strategy 3: Hiring distribution — one chunk per company
        chunks.extend(self._hiring_chunks())

        # Strategy 4: Temporal trend — one chunk per company per year
        chunks.extend(self._trend_chunks())

        # Strategy 5: Conflict records — keep BOTH copies, flagged
        chunks.extend(self._conflict_chunks())

        # Strategy 6: Adversarial queries — DO NOT embed, only log
        logger.info("Adversarial queries skipped (eval-only, not embedded)")

        logger.info(f"Total chunks before dedup: {len(chunks)}")
        return chunks

    def _eligibility_chunks(self) -> list[Chunk]:
        chunks = []
        for company, data in ELIGIBILITY_DATA.items():
            text = (
                f"Company: {company}. "
                f"Minimum CGPA required: {data['min_cgpa']}. "
                f"Maximum backlogs allowed: {data['max_backlogs']}. "
                f"Package offered: {data['package']} LPA. "
                f"Bond period: {data['bond']} years. "
                f"Bond-free: {'Yes' if data['bond'] == 0 else 'No'}. "
                f"Key interview topics: {data['topics']}. "
                f"Technical focus area: {data['tech']}."
            )
            chunks.append(Chunk(
                chunk_id=f"eligibility_{company.lower().replace(' ', '_').replace('&', 'and')}",
                text=text,
                source="official",
                company=company,
                section="eligibility",
                metadata={**data, "company": company},
            ))
        return chunks

    def _interview_chunks(self, sections: list[dict]) -> list[Chunk]:
        chunks = []
        current_company = None
        buffer = []
        for sec in sections:
            if sec["type"] == "interview_experience":
                text = sec["text"]
                # Detect company header
                for c in COMPANIES:
                    if c.upper() in text.upper() and "Technical Focus" in text:
                        if current_company and buffer:
                            chunks.append(self._make_interview_chunk(current_company, buffer))
                        current_company = c
                        buffer = [text]
                        break
                else:
                    buffer.append(text)

        if current_company and buffer:
            chunks.append(self._make_interview_chunk(current_company, buffer))
        return chunks

    def _make_interview_chunk(self, company: str, lines: list[str]) -> Chunk:
        text = " ".join(lines[:5])  # Cap at ~300 tokens
        return Chunk(
            chunk_id=f"interview_{company.lower().replace(' ', '_').replace('&', 'and')}_{uuid.uuid4().hex[:6]}",
            text=text,
            source="official",
            company=company,
            section="interview",
            metadata={"company": company, "round_number": "multiple"},
        )

    def _hiring_chunks(self) -> list[Chunk]:
        chunks = []
        for company, roles in HIRING_DATA.items():
            total = sum(roles.values())
            text = (
                f"Hiring distribution for {company}: "
                f"SDE roles: {roles['SDE']}, "
                f"Analyst roles: {roles['Analyst']}, "
                f"Officer roles: {roles['Officer']}, "
                f"Intern positions: {roles['Intern']}. "
                f"Total hires: {total}."
            )
            chunks.append(Chunk(
                chunk_id=f"hiring_{company.lower().replace(' ', '_').replace('&', 'and')}",
                text=text,
                source="official",
                company=company,
                section="hiring",
                metadata={"chart_type": "hiring", **roles},
            ))
        return chunks

    def _trend_chunks(self) -> list[Chunk]:
        chunks = []
        for company, years in TREND_DATA.items():
            for year, lpa in years.items():
                growth = lpa - list(years.values())[0]
                text = (
                    f"Package trend for {company} in {year}: {lpa} LPA. "
                    f"Growth from 2021 baseline: +{growth:.1f} LPA."
                )
                chunks.append(Chunk(
                    chunk_id=f"trend_{company.lower().replace(' ', '_')}_{year}",
                    text=text,
                    source="official",
                    company=company,
                    section="trend",
                    year=year,
                    metadata={"year": year, "lpa": lpa},
                ))
        return chunks

    def _conflict_chunks(self) -> list[Chunk]:
        chunks = []
        for company, portal in CONFLICT_DATA.items():
            official = ELIGIBILITY_DATA[company]
            # Chunk 1: official
            chunks.append(Chunk(
                chunk_id=f"conflict_official_{company.lower()}",
                text=(
                    f"[OFFICIAL SOURCE] {company}: CGPA cutoff {official['min_cgpa']}, "
                    f"Package {official['package']} LPA."
                ),
                source="official",
                company=company,
                section="conflict",
                conflict=True,
                metadata={"source_type": "official"},
            ))
            # Chunk 2: portal (possibly different)
            cgpa_conflict = portal["cgpa_portal"] != official["min_cgpa"]
            pkg_conflict = portal["package_portal"] != official["package"]
            chunks.append(Chunk(
                chunk_id=f"conflict_portal_{company.lower()}",
                text=(
                    f"[PORTAL SOURCE] {company}: CGPA cutoff {portal['cgpa_portal']}, "
                    f"Package {portal['package_portal']} LPA. "
                    f"{'CGPA differs from official.' if cgpa_conflict else ''} "
                    f"{'Package differs from official.' if pkg_conflict else ''}"
                ),
                source="portal",
                company=company,
                section="conflict",
                conflict=True,
                metadata={"source_type": "portal"},
            ))
        return chunks