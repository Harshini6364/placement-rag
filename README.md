# Placement Intelligence Assistant

> A production-grade Multimodal RAG system for placement intelligence at SVECW — built for RAG-ATHON 24.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama--3.1-orange)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [RAG Pipeline — 6 Stages](#rag-pipeline--6-stages)
- [Chunking Strategy](#chunking-strategy)
- [Tool-Augmented Agent](#tool-augmented-agent)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description

The **Placement Intelligence Assistant** is a full-stack Retrieval-Augmented Generation (RAG) system built on top of the SVECW Placement Dataset. It answers student placement queries — eligibility filtering, package comparisons, interview preparation, trend analysis, and conflict detection — with grounded, hallucination-resistant answers powered by Groq's Llama-3.1 model.

The system handles every content modality in the PDF: structured tables, free-text interview experiences, bar chart images (via Groq vision), temporal trend data, and intentionally conflicting records — each with its own tailored chunking, retrieval, and reasoning strategy.

### Why this project?

Standard RAG systems fail on placement datasets because:

- Tables get merged into garbled text by naive PDF parsers
- Charts are vector graphics — invisible to text extractors
- The same data appears from two sources with different values (conflict)
- Some questions have no answer in the document and need graceful fallback
- Aggregation queries (highest, lowest, best ratio) need multi-chunk reasoning

This project solves all of the above.

---

## Features

- **Hybrid Search** — Dense (FAISS + sentence-transformers) + Sparse (BM25) with Reciprocal Rank Fusion
- **CrossEncoder Reranking** — `ms-marco-MiniLM-L-6-v2` for precision after broad retrieval
- **Vision Chart Support** — Groq `llama-4-scout` reads bar charts and trend graphs rendered as images
- **Conflict Detection** — Flags official vs portal source disagreements; never silently returns one value
- **Fallback Guard** — Out-of-corpus queries get honest "not in documents" responses
- **AIMD Overshadow Limiter** — Binary-search context cap prevents hallucination from context overload
- **Tool-Augmented Agent** — Web search, calculator, date, and opinion guard tools for questions RAG cannot answer
- **Temporal Reasoning** — Year-tagged chunks enable growth calculations across 2021–2024
- **Aggregation Queries** — Multi-chunk scanning for highest/lowest/best comparisons
- **Query Caching** — `@st.cache_data` with 1-hour TTL; response time shows cache effect naturally
- **Persistent Chat History** — ChatGPT-style sidebar history saved to disk across reloads
- **30-Query Evaluator** — Automated scoring across Easy / Medium / Hard / Expert difficulty levels

---

## Architecture

```
User Query
    │
    ▼
Tool Router (Agent)
    │
    ├── In-corpus ──────────────────────────────────────────────┐
    │                                                           │
    │   Stage 1: Rewrite        (query expansion)              │
    │   Stage 2: Retrieve       (hybrid BM25 + FAISS + RRF)    │
    │   Stage 3: Rerank         (CrossEncoder)                  │
    │   Stage 4: Refine         (AIMD overshadow limiter)       │
    │   Stage 5: Insert         (grounded prompt builder)       │
    │   Stage 6: Generate       (Groq Llama-3.1)                │
    │                                                           │
    │   Safety: Conflict Detector + Fallback Guard              │
    │                                                           │
    └── Out-of-corpus ──────────────────────────────────────────┤
                                                                │
        Web Search Tool    (campus dates, real-time info)       │
        Calculator Tool    (ratios, eligibility filtering)      │
        Date Tool          (current date / schedule queries)    │
        Opinion Guard Tool (subjective career questions)        │
                                                                ▼
                                                    Grounded Answer + Tool Trace
```

---

## Project Structure

```
placement-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── core/                        # Abstract interfaces (SOLID)
│   ├── interfaces.py            # ABC classes for every pipeline stage
│   └── pipeline.py              # 6-stage RAG orchestrator + tool router
│
├── ingestion/                   # Stage 0: Parse → Chunk → Embed
│   ├── parser.py                # Docling + pdfplumber + Groq vision
│   ├── chunker.py               # Section-10 chunking strategy
│   ├── deduplicator.py          # TF-IDF cosine deduplication
│   └── embedder.py              # FAISS + BM25 hybrid index
│
├── retrieval/                   # Stages 1–3
│   ├── rewriter.py              # Rule-based query expansion
│   ├── retriever.py             # Hybrid RRF + metadata boosting
│   └── reranker.py              # CrossEncoder reranking
│
├── generation/                  # Stages 4–6
│   ├── refiner.py               # Context pruning wrapper
│   ├── prompt_builder.py        # Grounded prompt + query-type instructions
│   └── generator.py             # Groq Llama-3.1 with self-consistency
│
├── safety/                      # Hallucination prevention
│   ├── conflict_detector.py     # Official vs portal conflict detection
│   ├── fallback_guard.py        # Out-of-corpus detection
│   └── overshadow_limiter.py    # AIMD binary-search context cap
│
├── tools/                       # Tool-augmented agent
│   ├── base_tool.py             # Abstract tool interface
│   ├── tool_router.py           # Query classifier + dispatcher
│   ├── web_search_tool.py       # DuckDuckGo search
│   ├── calculator_tool.py       # Placement math + eligibility
│   ├── date_tool.py             # Current date/time
│   └── opinion_guard_tool.py    # Objective comparison for subjective queries
│
├── evaluation/                  # Eval pipeline
│   ├── evaluator.py             # 30-query automated scorer
│   ├── metrics.py               # Retrieval quality metrics
│   └── queries.py               # All 30 official evaluation queries
│
├── feedback/                    # Background feedback loop
│   └── loop.py                  # AIMD feedback controller
│
├── scripts/
│   ├── ingest.py                # One-time ingestion script
│   └── evaluate.py              # Run full evaluation suite
│
├── data/
│   └── Placement_RAG_Dataset_Enhanced.pdf
│
└── app.py                       # Streamlit UI
```

---

## Technologies Used

| Category | Technology |
|---|---|
| PDF Parsing | Docling 2.x + pdfplumber |
| Vision (Charts) | Groq `llama-4-scout-17b` |
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | FAISS `IndexFlatIP` |
| Sparse Search | BM25Okapi (rank-bm25) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Web Search | DuckDuckGo Search (duckduckgo-search) |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Installation

### Prerequisites

- Python 3.11+
- Git
- A free [Groq API key](https://console.groq.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Harshini6364/placement-rag.git
cd placement-rag

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBED_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index
BM25_PATH=data/bm25_store.pkl
CHUNKS_PATH=data/chunks.pkl
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
```

Get your free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Usage

### Step 1 — Run ingestion (one time only)

```bash
python scripts/ingest.py
```

Expected output:

```
INFO Eligibility chunks: 20
INFO Interview chunks added: 19
INFO Hiring chunks added: 20
INFO Trend chunks added: 47
INFO Conflict chunks added: 10
INFO Vision chart chunks added: 3
INFO Final chunk count: 119 (target: 80-150 ✓)
```

### Step 2 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 3 — Run evaluation (optional)

```bash
python scripts/evaluate.py
```

Results are saved to `data/eval_results.csv`.

---

## RAG Pipeline — 6 Stages

| Stage | Name | What it does |
|---|---|---|
| 1 | **Rewrite** | Expands query into 2–4 variants for better recall |
| 2 | **Retrieve** | Hybrid BM25 + FAISS with RRF fusion + metadata boosting |
| 3 | **Rerank** | CrossEncoder scores every (query, chunk) pair |
| 4 | **Refine** | AIMD overshadow limiter caps context tokens |
| 5 | **Insert** | Grounded prompt with query-type specific instructions |
| 6 | **Generate** | Groq Llama-3.1 with self-consistency check |

---

## Chunking Strategy

Implemented exactly per Section 10 of the dataset PDF:

| Content Type | Strategy | Chunk Size | Key Metadata |
|---|---|---|---|
| Eligibility table | 1 company = 1 chunk | ~80 tokens | `company`, `section=eligibility` |
| Interview experiences | Paragraph split per round | 200–300 tokens | `company`, `round_number` |
| Hiring distribution | 1 company = 1 chunk | ~60 tokens | `chart_type=hiring` |
| Trend data | 1 company per year | ~40 tokens | `company`, `year` |
| Conflict records | Both versions stored | ~50 tokens | `source=official/portal`, `conflict=True` |
| Chart images | Groq vision → text | ~100 tokens | `source=vision` |
| Adversarial queries | **Not embedded** | — | Eval only |

---

## Tool-Augmented Agent

For questions the RAG corpus cannot answer, the system routes to tools:

| Tool | Triggers | Example Query |
|---|---|---|
| `web_search` | campus dates, schedules, real-time info | *"When will TCS visit SVECW?"* |
| `calculator` | ratios, eligibility, ranking | *"I have CGPA 5.0, where can I apply?"* |
| `current_date` | date/time queries | *"What is today's date?"* |
| `opinion_guard` | subjective career questions | *"Should I join Google or Microsoft?"* |

The router classifies every query before RAG runs. Tool answers are supplemented with RAG context where relevant.

---

## Evaluation

The system is evaluated on 30 official queries from Section 9 of the dataset:

| Difficulty | Count | Tests |
|---|---|---|
| Easy | 8 | Direct table lookup, boolean queries |
| Medium | 10 | Multi-row filters, chart comparisons, temporal |
| Hard | 7 | 3-condition filters, conflict resolution, full synthesis |
| Expert | 5 | Out-of-corpus fallback, edge cases |

Run with:

```bash
python scripts/evaluate.py
```

---

## Known Limitations

- Docling layout model causes `std::bad_alloc` on machines with < 8GB RAM — pdfplumber fallback activates automatically
- Vision chart reading depends on Groq's `llama-4-scout` model availability
- Web search requires internet connectivity during query time
- CrossEncoder reranking adds ~1–2s latency on CPU; use `PassthroughReranker` if speed is critical
- Chat history is stored locally in `data/chat_history.json` — not shared across machines

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure your code follows the existing SOLID structure — each new retrieval or generation strategy should implement the corresponding abstract interface in `core/interfaces.py`.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 SVECW — Department of Information Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

> Built with care for RAG-ATHON 24 · SVECW · Department of Information Technology
