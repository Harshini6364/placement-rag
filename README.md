# Placement Intelligence Assistant

> A production-grade Multimodal RAG system for placement intelligence — built for RAG-ATHON 24 at SVECW, Department of Information Technology.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.1-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green?style=flat-square)
![BM25](https://img.shields.io/badge/Sparse-BM25-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Table of Contents

- [Project Description](#project-description)
- [Why This Project](#why-this-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Install and Run](#how-to-install-and-run)
- [How to Use the Project](#how-to-use-the-project)
- [RAG Pipeline — 6 Stages](#rag-pipeline--6-stages)
- [Chunking Strategy](#chunking-strategy)
- [Tool-Augmented Agent](#tool-augmented-agent)
- [Evaluation Results](#evaluation-results)
- [Credits](#credits)
- [License](#license)

---

## Project Description

The **Placement Intelligence Assistant** is a full-stack Retrieval-Augmented Generation (RAG) system built on top of the SVECW Placement Dataset for RAG-ATHON 24.

It answers student placement queries — eligibility filtering, package comparisons, interview preparation, trend analysis, conflict detection, and multi-hop reasoning — with grounded, hallucination-resistant answers powered by Groq's Llama-3.1 model.

The system handles every content modality present in the PDF:

- **Structured tables** — eligibility criteria, hiring distribution, overall statistics
- **Free-text narratives** — interview experiences split per round
- **Bar chart images** — rendered as PNG and described by Groq vision model
- **Temporal trend data** — package growth per company per year (2021–2024)
- **Conflicting records** — intentionally mismatched official vs portal data
- **Out-of-corpus questions** — routed to tools instead of hallucinating

Each content type gets its own chunking, retrieval, and reasoning strategy — exactly as specified in Section 10 of the dataset PDF.

---

## Why This Project

Standard RAG systems fail on placement datasets for several reasons:

- PDF tables get merged into garbled text by naive parsers
- Bar charts are vector graphics invisible to text extractors
- The same company data appears from two sources with different values
- Some questions have no answer in the document and need graceful fallback
- Aggregation queries like "highest paying" need reasoning across many chunks
- Time-based queries like "grew the most since 2021" need year-tagged metadata

This project solves all of the above with a layered, modular architecture built on SOLID principles. Every stage is abstracted behind an interface so components can be swapped without touching the pipeline.

---

## Features

- **Hybrid Search** — Dense FAISS embeddings plus sparse BM25 combined with Reciprocal Rank Fusion
- **CrossEncoder Reranking** — `ms-marco-MiniLM-L-6-v2` for high-precision reranking after broad retrieval
- **Vision Chart Support** — Groq `llama-4-scout` reads bar charts and trend graphs rendered as page images
- **Content-Aware Chunking** — Six different chunking strategies based on content type, following Section 10 of the dataset exactly
- **TF-IDF Deduplication** — Reduces 400+ raw chunks to 80–120 meaningful ones using cosine similarity
- **Conflict Detection** — Flags official vs portal source disagreements and never silently returns one value
- **Fallback Guard** — Out-of-corpus queries receive honest "not in documents" responses
- **AIMD Overshadow Limiter** — Binary-search context cap prevents hallucination from context overload
- **Temporal Reasoning** — Year-tagged trend chunks enable growth calculations across 2021–2024
- **Aggregation Query Boosting** — Injects all relevant section chunks for highest/lowest/best comparisons
- **Multi-Hop Reasoning** — Structured step-by-step filtering across eligibility, hiring, and package data
- **Tool-Augmented Agent** — Web search, calculator, date, and opinion guard tools for unanswerable questions
- **Query Rewriting** — Expands each query into 2–4 variants for better recall
- **Query Caching** — `@st.cache_data` with 1-hour TTL; response time naturally shows cache effect
- **Persistent Chat History** — ChatGPT-style sidebar history saved to disk and restored on reload
- **34-Query Evaluator** — Automated scoring across Easy, Medium, Hard, Expert, and Multi-hop difficulty levels
- **Rate Limit Protection** — Auto-retry with backoff on Groq API 429 errors

---

## Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| PDF Parsing | Docling 2.x + pdfplumber | Layout-aware extraction with table fallback |
| Vision | Groq llama-4-scout-17b | Chart image to text conversion |
| Embedding | all-MiniLM-L6-v2 | Dense vector embeddings |
| Vector Store | FAISS IndexFlatIP | Fast dense similarity search |
| Sparse Search | BM25Okapi | Keyword-based sparse retrieval |
| Fusion | Reciprocal Rank Fusion | Combining dense and sparse results |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision reranking |
| LLM | Groq llama-3.1-8b-instant | Fast answer generation |
| Web Search | ddgs (DuckDuckGo) | Real-time out-of-corpus queries |
| UI | Streamlit | Interactive web interface |
| Language | Python 3.11 | Core implementation |

---

## Project Structure

```
placement-rag/
│
├── README.md                        # This file
├── requirements.txt                 # All Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore                       # Excludes venv, indexes, secrets
├── version.py                       # Project metadata and banner
├── app.py                           # Streamlit UI — main entry point
│
├── core/                            # SOLID abstract interfaces
│   ├── interfaces.py                # ABC classes for every pipeline stage
│   └── pipeline.py                  # 6-stage RAG orchestrator + tool router
│
├── ingestion/                       # Stage 0: Parse → Chunk → Embed
│   ├── parser.py                    # Docling + pdfplumber + Groq vision
│   ├── chunker.py                   # Section-10 chunking strategy
│   ├── deduplicator.py              # TF-IDF cosine deduplication
│   └── embedder.py                  # FAISS + BM25 hybrid index builder
│
├── retrieval/                       # Stages 1–3
│   ├── rewriter.py                  # Query expansion (2–4 variants)
│   ├── retriever.py                 # Hybrid RRF + metadata boosting
│   └── reranker.py                  # CrossEncoder reranking
│
├── generation/                      # Stages 4–6
│   ├── refiner.py                   # Context pruning wrapper
│   ├── prompt_builder.py            # Grounded prompt with query instructions
│   └── generator.py                 # Groq Llama-3.1 with self-consistency
│
├── safety/                          # Hallucination prevention
│   ├── conflict_detector.py         # Official vs portal conflict detection
│   ├── fallback_guard.py            # Out-of-corpus detection
│   └── overshadow_limiter.py        # AIMD binary-search context cap
│
├── tools/                           # Tool-augmented agent
│   ├── base_tool.py                 # Abstract tool interface
│   ├── tool_router.py               # Query classifier and dispatcher
│   ├── web_search_tool.py           # DuckDuckGo search
│   ├── calculator_tool.py           # Eligibility filtering and ratios
│   ├── date_tool.py                 # Current date queries
│   └── opinion_guard_tool.py        # Objective comparison for subjective queries
│
├── evaluation/                      # 34-query evaluation pipeline
│   ├── queries.py                   # 30 official + 4 multi-hop queries
│   ├── metrics.py                   # Retrieval quality metrics
│   └── evaluator.py                 # Automated scorer with visual summary
│
├── feedback/                        # Background AIMD feedback loop
│   └── loop.py                      # Feedback controller (silent)
│
├── scripts/
│   ├── ingest.py                    # One-time ingestion pipeline
│   └── evaluate.py                  # Run full 34-query evaluation
│
└── data/
    ├── README.md                    # Data folder documentation
    └── Placement_RAG_Dataset_Enhanced.pdf   ← place here
```

---

## How to Install and Run

### Prerequisites

- Python 3.11 or higher
- Git
- A free [Groq API key](https://console.groq.com) — no credit card required

### Step 1 — Clone the repository

```bash
git clone https://github.com/Harshini6364/placement-rag.git
cd placement-rag
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables

```bash
copy .env.example .env
```

Open `.env` and add your Groq API key:

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

### Step 5 — Place the PDF

Copy `Placement_RAG_Dataset_Enhanced.pdf` into the `data/` folder.

### Step 6 — Run ingestion (one time only)

```bash
python scripts/ingest.py
```

Expected output:

```
Eligibility chunks: 20
Interview chunks added: 19
Hiring chunks added: 20
Trend chunks added: 47
Conflict chunks added: 10
Vision chart chunks added: 3
Final chunk count: 119 (target: 80-150 ✓)
```

### Step 7 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use the Project

### Asking questions

Type any placement-related question in the input box and click **Ask**. The system retrieves relevant chunks, reranks them, and generates a grounded answer.

### Quick test queries

The sidebar contains pre-loaded queries covering every difficulty level. Click any of them to instantly populate the input box and run the query.

### Reading the answer

| Box colour | Meaning |
|---|---|
| Green | Standard grounded answer from documents |
| Red | Conflicting data detected between official and portal sources |
| Yellow | Out-of-corpus — information not available in the documents |

### Diagnostics

Expand the **Retrieval Diagnostics** section under any answer to see retrieval quality score, context tokens, overshadow risk, and self-consistency score.

### Chat history

All previous questions appear below the current answer, newest first. The sidebar also shows a clickable history list. Reloading the page preserves history — it is saved automatically to `data/chat_history.json`.

### Response time

Each answer shows its response time. Repeated questions show near-zero response time because they are served from the 1-hour cache — no extra API call is made.

### Running the evaluation suite

```bash
python scripts/evaluate.py
```

Runs all 34 queries and saves results to:

```
data/eval_results.csv
data/multihop_results.csv
```

---

## RAG Pipeline — 6 Stages

| Stage | Name | What it does |
|---|---|---|
| 1 | **Rewrite** | Expands query into 2–4 variants for better recall |
| 2 | **Retrieve** | Hybrid BM25 + FAISS with RRF fusion and metadata boosting |
| 3 | **Rerank** | CrossEncoder scores every (query, chunk) pair for precision |
| 4 | **Refine** | AIMD overshadow limiter caps context to 500–1500 token sweet spot |
| 5 | **Insert** | Grounded prompt built with query-type specific instructions |
| 6 | **Generate** | Groq Llama-3.1 generates answer with self-consistency check |

Before Stage 1, a **Tool Router** classifies the query. Out-of-corpus queries are sent directly to tools instead of entering the RAG pipeline.

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
| Adversarial queries | **Not embedded** | — | Evaluation only |

An unoptimised pipeline on this document produces 400+ chunks. After content-aware chunking and TF-IDF deduplication, this system produces 119 meaningful chunks.

---

## Tool-Augmented Agent

For questions the RAG corpus cannot answer, the tool router dispatches to the appropriate tool:

| Tool | Triggers | Example query |
|---|---|---|
| `web_search` | Campus dates, schedules, real-time info | *"When will TCS visit SVECW?"* |
| `calculator` | Ratios, eligibility, below-threshold CGPA | *"I have CGPA 5.0, where can I apply?"* |
| `current_date` | Date and time queries | *"What is today's date?"* |
| `opinion_guard` | Subjective career questions | *"Should I join Google or Microsoft?"* |

Calculator and opinion guard answers are supplemented with RAG context where available.

---

## Evaluation Results

Scores on 34 official queries (30 from Section 9 + 4 multi-hop from Section 4):

| Difficulty | Score | Queries |
|---|---|---|
| Easy | 87.5% | 8 |
| Medium | 78.2% | 10 |
| Hard | 78.6% | 7 |
| Expert | 73.4% | 5 |
| Multi-hop | 87.5% | 4 |
| **Overall** | **77.9%** | **34** |

Skills scoring 100%: Direct table lookup, Temporal reasoning, Full synthesis, Out-of-corpus fallback, 3-condition filter, Column filter, Hiring table aggregation, Chart/table comparison, Filter + sort, Join: tech + hiring, Multi-attribute comparison, Text retrieval + synthesis.

---

## Credits

Built by students of SVECW — Department of Information Technology for RAG-ATHON 24.

Dataset provided by the RAG-ATHON 24 organizing committee.

Libraries and tools that made this possible:

- [Docling](https://github.com/DS4SD/docling) — layout-aware PDF parsing
- [Groq](https://console.groq.com) — fast LLM and vision inference
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search by Meta
- [Sentence Transformers](https://www.sbert.net/) — embedding and cross-encoder models
- [Streamlit](https://streamlit.io) — UI framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF table extraction
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 sparse retrieval

---

## License

```
MIT License

Copyright (c) 2026 SVECW — Department of Information Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.
```

---

> Built with care for RAG-ATHON 24 · SVECW · Department of Information Technology
