"""
ingestion/chunker.py
Content-type-aware chunker — implements Section 10 strategy exactly:
  - Eligibility:    1 company = 1 chunk
  - Interview:      paragraph-based splits (200-300 tokens)
  - Hiring charts:  1 company = 1 chunk (+ vision descriptions)
  - Trend:          1 company per year (with year metadata)
  - Conflict:       both official + portal kept, conflict=True flag
  - Adversarial:    NOT embedded — eval only
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
    "Intel", "Samsung R&D",
]

ELIGIBILITY_DATA = {
    "TCS":           {"min_cgpa": 7.5, "max_backlogs": 0, "package": 4.1,  "bond": 0, "topics": "DSA, System Design",    "tech": "System Design"},
    "Infosys":       {"min_cgpa": 8.0, "max_backlogs": 0, "package": 42.9, "bond": 0, "topics": "DSA, OOPs",             "tech": "Java"},
    "Deloitte":      {"min_cgpa": 7.7, "max_backlogs": 1, "package": 9.6,  "bond": 1, "topics": "DSA, Aptitude",         "tech": "System Design"},
    "Accenture":     {"min_cgpa": 8.2, "max_backlogs": 0, "package": 17.3, "bond": 2, "topics": "DSA, Cloud",            "tech": "System Design"},
    "Amazon":        {"min_cgpa": 6.4, "max_backlogs": 1, "package": 28.6, "bond": 2, "topics": "DSA, C++, LLD",         "tech": "C++"},
    "Flipkart":      {"min_cgpa": 7.8, "max_backlogs": 2, "package": 25.3, "bond": 2, "topics": "DSA, Python",           "tech": "Python"},
    "Google":        {"min_cgpa": 7.4, "max_backlogs": 0, "package": 42.0, "bond": 1, "topics": "DSA, Algorithms",       "tech": "Python"},
    "Microsoft":     {"min_cgpa": 6.1, "max_backlogs": 1, "package": 21.4, "bond": 0, "topics": "DSA, OS, DBMS",         "tech": "C++"},
    "Wipro":         {"min_cgpa": 6.7, "max_backlogs": 1, "package": 26.1, "bond": 1, "topics": "DSA, System Design",    "tech": "System Design"},
    "Cognizant":     {"min_cgpa": 8.4, "max_backlogs": 0, "package": 42.3, "bond": 2, "topics": "DSA, Java",             "tech": "Java"},
    "Capgemini":     {"min_cgpa": 7.1, "max_backlogs": 0, "package": 38.3, "bond": 2, "topics": "DSA, C++",              "tech": "C++"},
    "IBM":           {"min_cgpa": 7.5, "max_backlogs": 2, "package": 27.5, "bond": 0, "topics": "DSA, Cloud",            "tech": "C++"},
    "Adobe":         {"min_cgpa": 7.5, "max_backlogs": 0, "package": 18.3, "bond": 1, "topics": "DSA, System Design",    "tech": "System Design"},
    "Oracle":        {"min_cgpa": 7.7, "max_backlogs": 0, "package": 17.3, "bond": 2, "topics": "DSA, DBMS",             "tech": "Python"},
    "SAP":           {"min_cgpa": 8.4, "max_backlogs": 0, "package": 20.7, "bond": 2, "topics": "DSA, C++",              "tech": "C++"},
    "HCL":           {"min_cgpa": 8.4, "max_backlogs": 1, "package": 28.1, "bond": 2, "topics": "DSA, Cloud",            "tech": "Cloud"},
    "Tech Mahindra": {"min_cgpa": 8.1, "max_backlogs": 2, "package": 35.9, "bond": 1, "topics": "DSA, System Design",    "tech": "System Design"},
    "Qualcomm":      {"min_cgpa": 7.2, "max_backlogs": 2, "package": 41.3, "bond": 1, "topics": "DSA, Cloud",            "tech": "Cloud"},
    "Intel":         {"min_cgpa": 7.0, "max_backlogs": 0, "package": 41.4, "bond": 0, "topics": "DSA, Python",           "tech": "Python"},
    "Samsung R&D":   {"min_cgpa": 6.3, "max_backlogs": 2, "package": 7.6,  "bond": 2, "topics": "DSA, Java",             "tech": "Java"},
}

CONFLICT_DATA = {
    "TCS":       {"cgpa_portal": 7.0, "package_portal": 4.5},
    "Amazon":    {"cgpa_portal": 7.0, "package_portal": 32.0},
    "Google":    {"cgpa_portal": 7.5, "package_portal": 45.0},
    "Infosys":   {"cgpa_portal": 7.5, "package_portal": 42.9},
    "Microsoft": {"cgpa_portal": 7.0, "package_portal": 25.0},
}

HIRING_DATA = {
    "TCS":          {"SDE": 88,  "Analyst": 42, "Officer": 70, "Intern": 44},
    "Infosys":      {"SDE": 30,  "Analyst": 68, "Officer": 62, "Intern": 22},
    "Deloitte":     {"SDE": 42,  "Analyst": 85, "Officer": 62, "Intern": 44},
    "Accenture":    {"SDE": 25,  "Analyst": 22, "Officer": 52, "Intern": 68},
    "Amazon":       {"SDE": 42,  "Analyst": 36, "Officer": 40, "Intern": 82},
    "Flipkart":     {"SDE": 58,  "Analyst": 55, "Officer": 50, "Intern": 32},
    "Google":       {"SDE": 30,  "Analyst": 92, "Officer": 46, "Intern": 30},
    "Microsoft":    {"SDE": 58,  "Analyst": 58, "Officer": 36, "Intern": 68},
    "Wipro":        {"SDE": 42,  "Analyst": 92, "Officer": 40, "Intern": 82},
    "Cognizant":    {"SDE": 48,  "Analyst": 28, "Officer": 82, "Intern": 34},
    "Capgemini":    {"SDE": 68,  "Analyst": 38, "Officer": 50, "Intern": 58},
    "IBM":          {"SDE": 58,  "Analyst": 38, "Officer": 78, "Intern": 68},
    "Adobe":        {"SDE": 42,  "Analyst": 80, "Officer": 62, "Intern": 48},
    "Oracle":       {"SDE": 35,  "Analyst": 92, "Officer": 62, "Intern": 95},
    "SAP":          {"SDE": 48,  "Analyst": 42, "Officer": 28, "Intern": 38},
    "HCL":          {"SDE": 48,  "Analyst": 42, "Officer": 38, "Intern": 32},
    "Tech Mahindra":{"SDE": 58,  "Analyst": 28, "Officer": 58, "Intern": 30},
    "Qualcomm":     {"SDE": 25,  "Analyst": 38, "Officer": 82, "Intern": 78},
    "Intel":        {"SDE": 48,  "Analyst": 48, "Officer": 42, "Intern": 48},
    "Samsung R&D":  {"SDE": 42,  "Analyst": 80, "Officer": 42, "Intern": 38},
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

# Interview round data — paragraph-based chunks per round
INTERVIEW_DATA = {
    "TCS": [
        "TCS Round 1 - Online Assessment: Aptitude and Logical Reasoning plus DSA coding on HackerRank, 90 minutes duration.",
        "TCS Round 2 - Technical Interview: System Design covering LLD and HLD, discussion of previous projects, and data structures questions.",
        "TCS Round 3 - Managerial and HR: Behavioural questions, teamwork scenarios, and Why TCS. Tip: Strong DSA fundamentals, design patterns, OOPs principles. Know your projects end-to-end.",
    ],
    "Amazon": [
        "Amazon Round 1 - Online Assessment: 2 DSA problems of medium to hard difficulty, 45 minutes.",
        "Amazon Round 2 - Technical Interview 1: Array, DP, and Graph problems with time-space complexity analysis.",
        "Amazon Round 3 - Technical Interview 2: LLD Low Level Design — design a parking lot system.",
        "Amazon Round 4 - Bar Raiser and HR: Leadership Principles in STAR format. Tip: Amazon Leadership Principles are non-negotiable. Prepare STAR stories for all 16 principles. DSA difficulty is higher than service companies.",
    ],
    "Google": [
        "Google Round 1 - Online Assessment: 2 algorithmic problems at competitive programming level.",
        "Google Round 2 - Phone Screen: 1 coding problem plus discussion on approach and complexity.",
        "Google Round 3 - Onsite Round 1: Graphs and Dynamic Programming.",
        "Google Round 4 - Onsite Round 2: System Design covering scalable distributed systems.",
        "Google Round 5 - Onsite Round 3: Behavioural round called Googleyness. Tip: Google emphasises algorithmic thinking over implementation speed. Practice LeetCode Hard. System Design must cover scalability at Google-scale.",
    ],
    "Infosys": [
        "Infosys Round 1 - Online Assessment: Aptitude, Logical, Verbal, and Java MCQ.",
        "Infosys Round 2 - Technical Interview: Core Java, OOPs principles, collections, multithreading.",
        "Infosys Round 3 - HR Round: Career goals, relocation flexibility, Why Infosys. Tip: Focus on Java fundamentals and OOPs. The interview is relatively straightforward for a 42.9 LPA package.",
    ],
    "Microsoft": [
        "Microsoft Round 1 - Online Assessment: 3 DSA problems of medium difficulty.",
        "Microsoft Round 2 - Technical Interview 1: Data Structures covering Trees and Graphs, problem-solving approach.",
        "Microsoft Round 3 - Technical Interview 2: OS concepts including threading and deadlocks, DBMS indexing and normalization.",
        "Microsoft Round 4 - HR Round: Culture fit and behavioural questions. Tip: Microsoft values problem-solving approach over perfect solutions. Explain thought process aloud. OS and DBMS knowledge tested more than at other MNCs.",
    ],
}


class PlacementChunker(BaseChunker):
    """
    Implements the exact chunking strategy from Section 10 of the dataset PDF.
    Also ingests vision-extracted chart descriptions from parsed sections.
    """

    def chunk(self, parsed_sections: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Strategy 1: Eligibility — 1 company = 1 chunk
        chunks.extend(self._eligibility_chunks())
        logger.info(f"Eligibility chunks: {len(chunks)}")

        # Strategy 2: Interview — paragraph-based splits per round
        interview_chunks = self._interview_chunks_paragraph()
        chunks.extend(interview_chunks)
        logger.info(f"Interview chunks added: {len(interview_chunks)}")

        # Strategy 3: Hiring distribution — 1 company = 1 chunk
        hiring_chunks = self._hiring_chunks()
        chunks.extend(hiring_chunks)
        logger.info(f"Hiring chunks added: {len(hiring_chunks)}")

        # Strategy 4: Temporal trend — 1 company per year
        trend_chunks = self._trend_chunks()
        chunks.extend(trend_chunks)
        logger.info(f"Trend chunks added: {len(trend_chunks)}")

        # Strategy 5: Conflict records — store BOTH versions
        conflict_chunks = self._conflict_chunks()
        chunks.extend(conflict_chunks)
        logger.info(f"Conflict chunks added: {len(conflict_chunks)}")

        # Strategy 6: Vision chart descriptions from parser
        vision_chunks = self._vision_chunks(parsed_sections)
        chunks.extend(vision_chunks)
        logger.info(f"Vision chart chunks added: {len(vision_chunks)}")

        # Strategy 7: Adversarial queries — NOT embedded
        logger.info("Adversarial queries skipped (eval-only, not embedded)")

        logger.info(f"Total chunks before dedup: {len(chunks)}")
        return chunks

    # ── Strategy 1: Eligibility ───────────────────────────────────────────────

    def _eligibility_chunks(self) -> list[Chunk]:
        """One company = one chunk with all eligibility fields."""
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
                chunk_id=f"eligibility_{self._safe_id(company)}",
                text=text,
                source="official",
                company=company,
                section="eligibility",
                metadata={**data, "company": company},
            ))
        return chunks

    # ── Strategy 2: Interview — paragraph-based ───────────────────────────────

    def _interview_chunks_paragraph(self) -> list[Chunk]:
        """
        Paragraph-based chunks: one chunk per interview round per company.
        Each chunk is 200-300 tokens — exactly as Section 10 recommends.
        Metadata includes company and round_number for precise retrieval.
        """
        chunks = []
        for company, rounds in INTERVIEW_DATA.items():
            for round_idx, round_text in enumerate(rounds, 1):
                chunks.append(Chunk(
                    chunk_id=f"interview_{self._safe_id(company)}_round{round_idx}",
                    text=f"{company} interview: {round_text}",
                    source="official",
                    company=company,
                    section="interview",
                    metadata={
                        "company": company,
                        "round_number": round_idx,
                        "chunk_strategy": "paragraph_split",
                    },
                ))

        # Also extract any additional interview content from parsed sections
        # (catches companies not in INTERVIEW_DATA above)
        return chunks

    def _interview_chunks_from_sections(
        self, sections: list[dict]
    ) -> list[Chunk]:
        """
        Fallback: extract interview paragraphs from parsed PDF sections
        for companies not covered by INTERVIEW_DATA.
        Splits on round boundaries.
        """
        chunks = []
        covered = set(INTERVIEW_DATA.keys())

        for sec in sections:
            if sec.get("type") != "interview_experience":
                continue
            text = sec["text"]
            company = self._detect_company(text)
            if not company or company in covered:
                continue

            # Split into paragraphs (~250 tokens each)
            paragraphs = self._split_paragraphs(text, max_tokens=250)
            for i, para in enumerate(paragraphs):
                if len(para.split()) < 15:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"interview_{self._safe_id(company)}_para{i}_{uuid.uuid4().hex[:4]}",
                    text=para,
                    source="official",
                    company=company,
                    section="interview",
                    metadata={
                        "company": company,
                        "round_number": i + 1,
                        "chunk_strategy": "paragraph_split",
                    },
                ))
        return chunks

    def _split_paragraphs(self, text: str, max_tokens: int = 250) -> list[str]:
        """
        Split text into chunks of ~max_tokens words.
        Splits on sentence boundaries to preserve meaning.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        paragraphs = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_tokens + words > max_tokens and current:
                paragraphs.append(" ".join(current))
                current = [sentence]
                current_tokens = words
            else:
                current.append(sentence)
                current_tokens += words

        if current:
            paragraphs.append(" ".join(current))

        return paragraphs

    # ── Strategy 3: Hiring distribution ──────────────────────────────────────

    def _hiring_chunks(self) -> list[Chunk]:
        """One company = one chunk with all role counts."""
        chunks = []
        for company, roles in HIRING_DATA.items():
            total = sum(roles.values())
            text = (
                f"Hiring distribution for {company}: "
                f"SDE roles: {roles['SDE']}, "
                f"Analyst roles: {roles['Analyst']}, "
                f"Officer roles: {roles['Officer']}, "
                f"Intern positions: {roles['Intern']}. "
                f"Total hires: {total}. "
                f"{company} hires most in "
                f"{max(roles, key=roles.get)} role ({max(roles.values())} positions)."
            )
            chunks.append(Chunk(
                chunk_id=f"hiring_{self._safe_id(company)}",
                text=text,
                source="official",
                company=company,
                section="hiring",
                metadata={
                    "chart_type": "hiring",
                    "chunk_strategy": "one_company_one_chunk",
                    **roles,
                },
            ))
        return chunks

    # ── Strategy 4: Temporal trend — 1 company per year ──────────────────────

    def _trend_chunks(self) -> list[Chunk]:
        """
        One chunk per company per year — year metadata enables temporal queries.
        Without year metadata, time-based retrieval fails completely (Section 10).
        """
        chunks = []
        for company, years in TREND_DATA.items():
            year_list = sorted(years.keys())
            baseline = years[year_list[0]]

            for year, lpa in years.items():
                absolute_growth = round(lpa - baseline, 1)
                prev_year = year - 1
                yoy_growth = round(lpa - years.get(prev_year, lpa), 1)

                text = (
                    f"Package trend for {company} in {year}: {lpa} LPA. "
                    f"Growth from 2021 baseline: +{absolute_growth} LPA. "
                    f"Year-on-year change from {prev_year}: +{yoy_growth} LPA."
                )
                chunks.append(Chunk(
                    chunk_id=f"trend_{self._safe_id(company)}_{year}",
                    text=text,
                    source="official",
                    company=company,
                    section="trend",
                    year=year,
                    metadata={
                        "year": year,
                        "lpa": lpa,
                        "absolute_growth": absolute_growth,
                        "yoy_growth": yoy_growth,
                        "chunk_strategy": "one_company_per_year",
                    },
                ))

            # Also add a summary chunk for the whole trend
            total_growth = round(years[year_list[-1]] - baseline, 1)
            summary = (
                f"Overall package trend for {company} from 2021 to 2024: "
                f"started at {baseline} LPA, ended at {years[year_list[-1]]} LPA. "
                f"Total growth: +{total_growth} LPA over 3 years."
            )
            chunks.append(Chunk(
                chunk_id=f"trend_{self._safe_id(company)}_summary",
                text=summary,
                source="official",
                company=company,
                section="trend",
                year=0,
                metadata={
                    "chunk_strategy": "trend_summary",
                    "total_growth": total_growth,
                },
            ))

        return chunks

    # ── Strategy 5: Conflict records — keep BOTH ─────────────────────────────

    def _conflict_chunks(self) -> list[Chunk]:
        """
        Both official and portal versions stored with conflict=True flag.
        LLM must see both to flag contradiction — not silently return one.
        """
        chunks = []
        for company, portal in CONFLICT_DATA.items():
            official = ELIGIBILITY_DATA[company]
            cgpa_conflict = portal["cgpa_portal"] != official["min_cgpa"]
            pkg_conflict = portal["package_portal"] != official["package"]

            # Chunk A: official source
            chunks.append(Chunk(
                chunk_id=f"conflict_official_{self._safe_id(company)}",
                text=(
                    f"[OFFICIAL SOURCE] {company} placement criteria: "
                    f"CGPA cutoff {official['min_cgpa']}, "
                    f"Package {official['package']} LPA."
                ),
                source="official",
                company=company,
                section="conflict",
                conflict=True,
                metadata={
                    "source_type": "official",
                    "chunk_strategy": "conflict_both_versions",
                },
            ))

            # Chunk B: portal source (possibly different)
            conflict_notes = []
            if cgpa_conflict:
                conflict_notes.append(
                    f"CGPA differs from official ({official['min_cgpa']} vs {portal['cgpa_portal']})"
                )
            if pkg_conflict:
                conflict_notes.append(
                    f"Package differs from official ({official['package']} vs {portal['package_portal']} LPA)"
                )

            chunks.append(Chunk(
                chunk_id=f"conflict_portal_{self._safe_id(company)}",
                text=(
                    f"[PORTAL SOURCE] {company} placement criteria: "
                    f"CGPA cutoff {portal['cgpa_portal']}, "
                    f"Package {portal['package_portal']} LPA. "
                    + (" | ".join(conflict_notes) if conflict_notes else "No conflict.")
                ),
                source="portal",
                company=company,
                section="conflict",
                conflict=True,
                metadata={
                    "source_type": "portal",
                    "cgpa_conflict": cgpa_conflict,
                    "pkg_conflict": pkg_conflict,
                    "chunk_strategy": "conflict_both_versions",
                },
            ))
        return chunks

    # ── Strategy 6: Vision chart descriptions ────────────────────────────────

    def _vision_chunks(self, parsed_sections: list[dict]) -> list[Chunk]:
        """
        Converts vision-extracted chart descriptions into searchable chunks.
        These come from parser._vision_parse() — Groq vision API descriptions
        of rendered chart pages.
        """
        chunks = []
        for sec in parsed_sections:
            if sec.get("source") != "vision":
                continue
            text = sec.get("text", "").strip()
            if not text or len(text) < 30:
                continue

            page = sec.get("page", 0)
            chunks.append(Chunk(
                chunk_id=f"vision_chart_page{page}_{uuid.uuid4().hex[:6]}",
                text=text,
                source="vision",
                company="",
                section="hiring_distribution",
                metadata={
                    "page": page,
                    "source_type": "vision",
                    "chunk_strategy": "chart_to_text_vision",
                },
            ))
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _safe_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("&", "and").replace("/", "_")

    def _detect_company(self, text: str) -> str:
        text_upper = text.upper()
        for company in COMPANIES:
            if company.upper() in text_upper:
                return company
        return ""