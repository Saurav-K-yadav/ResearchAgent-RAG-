# ResearchAgent - Local RAG Research Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-FF6B00?style=for-the-badge)](#)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-4B8BBE?style=for-the-badge)](#)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C?style=for-the-badge)](#)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=for-the-badge)](#)

ResearchAgent is a **local-first AI research workflow** for discovering arXiv papers, building a private vector-backed knowledge base, and asking grounded questions over academic PDFs. It combines **Retrieval-Augmented Generation (RAG)**, **semantic search**, **PDF ingestion**, **ChromaDB**, **SentenceTransformers**, **Google Gemini**, **LangGraph agent orchestration**, and optional **LangSmith-style observability** in a clean Gradio interface.

Built as a practical portfolio project for modern AI engineering: not just an LLM wrapper, but an end-to-end pipeline that searches, downloads, parses, chunks, embeds, stores, retrieves, reasons, and responds.

![ResearchAgent local workflow](Image/Screenshot%20from%202026-09-11%2022-35-30.png)

## Why This Project Stands Out

- **End-to-end RAG pipeline**: arXiv search -> PDF download -> text extraction -> chunking -> embeddings -> vector database -> grounded Gemini answers.
- **Agentic workflow orchestration**: LangGraph routes high-level user instructions to the right tool: search, ingest, query, or answer.
- **Local-first knowledge base**: PDFs, embeddings, ChromaDB storage, and fallback logs stay on your machine by default.
- **Production-minded resilience**: Gemini model fallback, conservative retrieval thresholds, structured tool boundaries, and graceful no-context behavior.
- **Recruiter-friendly AI stack**: Python, LLMs, RAG, vector databases, semantic search, ChromaDB, SentenceTransformers, LangGraph, Gradio, LangSmith-style tracing, MCP-ready tools.

## Demo

ResearchAgent provides a simple workflow for literature review and technical research:

1. **Find papers** on arXiv using a keyword, topic, or author query.
2. **Download PDFs** into a local project folder.
3. **Ingest papers** into a persistent ChromaDB vector store.
4. **Ask questions** using retrieved paper passages as context.
5. **Use the agent helper** to automate multi-step research instructions.

![ResearchAgent agent result](Image/Screenshot%20from%202026-09-11%2022-38-55.png)

## Core Features

| Feature | What it does |
| --- | --- |
| arXiv paper discovery | Searches arXiv and downloads relevant PDFs automatically. |
| PDF ingestion | Extracts text from PDFs with PyMuPDF and prepares it for retrieval. |
| Text chunking | Uses LangChain text splitters with overlap for better context preservation. |
| Embeddings | Generates local semantic embeddings with `all-MiniLM-L6-v2`. |
| Vector search | Stores and queries chunks in a persistent ChromaDB collection. |
| RAG answers | Retrieves relevant passages and asks Gemini to answer from context. |
| LLM fallback | Gives a concise general answer when no useful local context exists. |
| LangGraph agent | Chooses the correct research tool from natural-language instructions. |
| Observability | Supports LangSmith-compatible tracing with local JSON fallback logs. |
| Gradio UI | Provides a clean browser interface for the full research workflow. |

## Tech Stack

- **Language**: Python 3.11+
- **LLM**: Google Gemini via `google-genai`
- **Agent Framework**: LangGraph
- **Vector Database**: ChromaDB persistent client
- **Embeddings**: SentenceTransformers `all-MiniLM-L6-v2`
- **PDF Parsing**: PyMuPDF
- **Chunking**: LangChain Text Splitters
- **UI**: Gradio
- **Research Source**: arXiv API
- **Tracing**: LangSmith-compatible HTTP logging with local fallback
- **Tooling**: MCP-style server functions and CLI fallback paths

## Architecture

```text
User / Gradio UI
      |
      v
server.py
      |
      +--> arxiv_fetcher.py   -> search arXiv and download PDFs
      |
      +--> database.py        -> parse PDF, chunk text, embed, store/query ChromaDB
      |
      +--> llm_provider.py    -> call Gemini with model fallback and usage metadata
      |
      +--> langgraph_agent.py -> decide tool, execute tool, return result
      |
      v
Local folders: pdfs/ chroma_data/ langsmith_logs/
```

## Project Structure

```text
ResearchAgent-RAG-/
|-- arxiv_fetcher.py          # arXiv search and PDF download
|-- database.py               # PDF parsing, chunking, embeddings, ChromaDB
|-- gradio_app.py             # Interactive browser UI
|-- langgraph_agent.py        # LangGraph agent orchestration
|-- langsmith_integration.py  # Remote tracing plus local fallback logs
|-- llm_provider.py           # Gemini wrapper and model fallback
|-- server.py                 # MCP tools, RAG answer flow, CLI fallback
|-- requirements.txt          # Python dependencies
|-- examples/                 # Sample research input
`-- Image/                    # README screenshots
```

## Quickstart

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ResearchAgent-RAG-
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

# Optional tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 5. Run the app

```bash
python gradio_app.py
```

Open the local Gradio URL shown in your terminal, usually:

```text
http://localhost:7860
```

## How It Works

### Retrieval-Augmented Generation

ResearchAgent retrieves relevant chunks from the local ChromaDB collection before asking Gemini to answer. If retrieved passages are weak or unavailable, the app clearly falls back to a concise general LLM answer instead of pretending it has paper-specific context.

### PDF Knowledge Base

Each ingested PDF is processed page by page, split into overlapping chunks, embedded with SentenceTransformers, and stored with metadata such as page number and paper title. This makes answers traceable back to source passages.

### LangGraph Orchestration

The agent helper accepts a natural-language instruction, asks Gemini to select one tool, then executes that tool through a small LangGraph state machine:

```text
decide_tool -> execute_tool -> result
```

This keeps the workflow extensible for future tools such as citation export, multi-paper comparison, report generation, and scheduled literature monitoring.

## Example Use Cases

- Build a personal research assistant for academic papers.
- Summarize machine learning, quantum computing, NLP, or AI safety papers.
- Ask grounded questions over a private PDF library.
- Prototype RAG systems with local vector search.
- Demonstrate AI engineering skills in LLM applications, embeddings, agents, and observability.

## Troubleshooting

- **`GOOGLE_API_KEY not set`**: Add your Gemini API key to `.env`.
- **No paper-specific answer**: Ingest a PDF first through the "Build your knowledge base" tab.
- **PDF text looks incomplete**: Some papers have weak text layers and may need OCR before ingestion.
- **Gemini model errors**: Set `GEMINI_MODEL` in `.env` or let the app use its fallback selection.
- **LangSmith tracing fails**: Keep `LANGSMITH_TRACING=false` or verify your endpoint and API key.

## About

ResearchAgent demonstrates how to turn modern AI components into a usable research product: local data ownership, vector retrieval, LLM reasoning, agent orchestration, and a clean UI.
