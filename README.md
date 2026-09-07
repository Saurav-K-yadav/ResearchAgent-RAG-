# ResearchAgent — Local, Privacy-First RAG Research Assistant

[![Status](https://img.shields.io/badge/status-ready-brightgreen)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Tech](https://img.shields.io/badge/tech-RAG%20%7C%20ChromaDB%20%7C%20Gemini%20%7C%20LangGraph-blue)](#)

A lightweight, local-first research assistant that helps you find, read, and ask questions about academic papers. It’s built to run on your machine, keep your data private, and demonstrate a practical RAG workflow you can extend.

Note: There’s a screenshot placeholder at docs/screenshot.png. If you don’t see an image on the Hub, add a screenshot file (recommended 1280×720) or update the README to point to your own image.

One-line: ResearchAgent — a local RAG assistant for searching arXiv, ingesting PDFs, and asking Gemini-powered questions.

Tags: RAG • ChromaDB • SentenceTransformers • Google Gemini • LangGraph • LangSmith • Gradio • PDF ingestion • Vector search • Python


ResearchAgent is a practical prototype I built to explore how local tools and modern LLMs can help with literature review. It wires together a simple, reusable pipeline: search arXiv, download PDFs, extract text, chunk and embed passages, store them in a local vector store, and answer grounded questions using Google Gemini.

This repository is written for people who want a working demo they can run and customize. It shows real engineering trade-offs and useful patterns, such as:
- A local, persistent vector store using ChromaDB
- Lightweight embeddings with SentenceTransformers (all-MiniLM-L6-v2)
- Google Gemini integration (google.genai) with model-fallback and usage reporting
- A small LangGraph-based orchestrator to pick and run tools
- LangSmith-compatible tracing (direct HTTP) with a local JSON fallback for reliability
- A user-friendly Gradio UI, and conservative RAG fallbacks to avoid hallucination

Key features (what recruiters look for)
- Local-first: no OpenAI keys required; runs using Google Gemini and local embeddings
- End-to-end pipeline: arXiv search → PDF download → PDF parsing → chunking → embedding → ChromaDB ingestion → RAG answers
- Robust orchestration: LangGraph state machine drives tool selection and execution
- Observability: optional LangSmith run logging + safe local JSON fallback logs
- Resilience: automatic Gemini model fallback when facing token limits or deprecated models
- Developer-friendly: clear module boundaries, docstrings for tools, CLI fallback for MCP

Project layout
- [requirements.txt](/home/sky/Desktop/project/requirements.txt)
- [arxiv_fetcher.py](/home/sky/Desktop/project/arxiv_fetcher.py) — arXiv search & PDF download
- [database.py](/home/sky/Desktop/project/database.py) — PDF reading, chunking, embeddings, ChromaDB storage & query
- [llm_provider.py](/home/sky/Desktop/project/llm_provider.py) — Gemini wrapper with model-selection and usage extraction
- [langgraph_agent.py](/home/sky/Desktop/project/langgraph_agent.py) — LangGraph orchestration graph
- [langsmith_integration.py](/home/sky/Desktop/project/langsmith_integration.py) — direct HTTP LangSmith logger + local fallback
- [server.py](/home/sky/Desktop/project/server.py) — MCP tools, CLI fallback, ask_researchagent alias
- [gradio_app.py](/home/sky/Desktop/project/gradio_app.py) — modern Gradio UI for end users
- [.env] (local; not committed) — stores API keys and flags

Quickstart (developer-friendly)
1. Create & activate a Python 3.11+ venv in the project root:
   python3 -m venv venv
   source ./venv/bin/activate
2. Install dependencies:
   ./venv/bin/pip install -r requirements.txt
3. Copy `.env.example` to `.env` and fill in required values (or create `.env`):
   - GOOGLE_API_KEY (required for Google Gemini)
   - LANGSMITH_API_KEY (optional; required for remote tracing)
   - LANGSMITH_ENDPOINT (optional)
   - LANGSMITH_TRACING=true|false
4. Launch the UI:
   ./venv/bin/python gradio_app.py
   Open http://localhost:7860

Usage summary (end-user flow)
1. Find papers: Use the "Find papers" tab to search arXiv and download PDFs.
2. Build your knowledge base: Select a downloaded PDF and ingest it into the local ChromaDB. PDF pages are chunked and embedded with SentenceTransformers.
3. Ask ResearchAgent: Ask focused questions. The system performs retrieval and either returns a grounded, context-aware answer (RAG) or falls back to a concise general LLM answer when local context is missing.
4. Agent helper: Give a single natural-language instruction and the LangGraph orchestrator chooses the right tools (search, ingest, query, or answer).

Developer notes (how it works internally)
- Ingestion: [database.ingest_pdf](/home/sky/Desktop/project/database.py) uses PyMuPDF to extract per-page text, then RecursiveCharacterTextSplitter to create 1000-token chunks with 200-token overlap, then encodes with SentenceTransformers and stores embeddings in a persistent ChromaDB at `./chroma_data`.
- Retrieval: [database.query_database](/home/sky/Desktop/project/database.py) embeds the query and performs a nearest-neighbors search returning documents, metadata and distances.
- LLM: [llm_provider.chat_with_gemini](/home/sky/Desktop/project/llm_provider.py) calls Google Gemini using `google.genai`; it auto-selects healthy models and retries on token-limit or model-deprecation errors. Token/usage metadata is extracted and forwarded to tracing.
- Tracing: [langsmith_integration.log_interaction](/home/sky/Desktop/project/langsmith_integration.py) posts runs to a LangSmith-compatible endpoint (uses `X-Api-Key` header and `run_type` field) and falls back to safe local JSON logs at `./langsmith_logs/` when remote tracing fails.
- Orchestration: [langgraph_agent.py](/home/sky/Desktop/project/langgraph_agent.py) builds a small graph (decide_tool → execute_tool). `server.agent_orchestrate` delegates to this graph; the graph uses Gemini to decide which tool to call and then executes it.

UX & product decisions
- The UI focuses on a linear research workflow (Find → Ingest → Ask → Agent) with progressive disclosure: result panels are hidden until actions produce output.
- Conservative RAG: If retrieved passages are weak (high distance) or empty, the system falls back to a concise LLM answer to avoid hallucination, and displays a short one-line guidance to ingest PDFs for paper-specific summaries.
- Privacy: All data (PDFs, embeddings, ChromaDB, and fallback logs) remain local unless remote tracing is explicitly enabled.

Troubleshooting & common issues
- LangSmith returns 401 or 404: verify `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT` in `.env`. Public LangSmith API expects `X-Api-Key` header and `/api/v1/runs` path with a `run_type` field.
- Gemini model errors: SDK will auto-select available models. If you see `MODEL_NOT_FOUND` or deprecation errors, update `GEMINI_MODEL` in `.env` or let the app auto-pick a supported model.
- PDF extraction issues: Some PDFs have poor text layers. Check `./pdfs/` and ensure PyMuPDF can extract text; fallback is to OCR externally before ingest.

