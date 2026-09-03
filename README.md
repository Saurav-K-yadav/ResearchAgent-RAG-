# ResearchAgent — Local MCP Workflow with Google Gemini and LangSmith

This project builds a local research assistant ("ResearchAgent") that:
- Searches arXiv and downloads PDFs
- Extracts text from PDFs and ingests them into a local, persistent ChromaDB
- Uses local SentenceTransformers embeddings (all-MiniLM-L6-v2)
- Performs Retrieval-Augmented Generation (RAG) by combining retrieved chunks
  with Google Gemini to answer user questions
- Traces interactions to LangSmith via an HTTP API (or local log fallback)
- Exposes tools for MCP (FastMCP when available) and a Gradio UI for interactive use

Contents
- requirements.txt — Python dependencies
- arxiv_fetcher.py — search + download logic
- database.py — PDF reading, chunking (RecursiveCharacterTextSplitter), embeddings and ChromaDB ingestion/query
- server.py — exposes tools (search_and_download_paper, ingest_paper, query_database, ask_gemini, agent_orchestrate) and CLI fallback
- llm_provider.py — Google Gemini (google.genai) wrapper
- langsmith_integration.py — direct LangSmith HTTP logger with safe local fallback
- gradio_app.py — Gradio UI for the core actions
- project_instructions.md — original project instructions (updated)

Important: Security
- All credentials are read from `.env` (project root) via python-dotenv. Do NOT commit `.env` to public repos.
- The application never prints or logs secret values. LangSmith HTTP requests use the API key only in Authorization headers.

Quick setup
1. Create and activate a Python 3.11+ virtualenv in the project root:
   python3 -m venv venv
   source ./venv/bin/activate
2. Install dependencies:
   ./venv/bin/pip install -r requirements.txt
3. Ensure `.env` contains at least:
   - GOOGLE_API_KEY
   - LANGSMITH_API_KEY (optional, used for LangSmith remote logging)
   - LANGSMITH_ENDPOINT (optional)
   - LANGSMITH_TRACING (true/false)
4. Run the Gradio UI:
   ./venv/bin/python gradio_app.py
   Then open http://localhost:7860 in your browser.

How it works (high level)
1. Search & download
   - `arxiv_fetcher.search_and_download` uses the `arxiv` Python library to find results and downloads PDFs to ./pdfs/.
2. Ingest
   - `database.ingest_pdf` reads the PDF with PyMuPDF, splits each page into chunks using `RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)`, embeds chunks with SentenceTransformers (`all-MiniLM-L6-v2`), and stores documents and embeddings in a persistent ChromaDB at `./chroma_data`.
3. Query / Ask
   - `server.ask_gemini` retrieves top-k chunks from ChromaDB, composes a prompt containing those passages and the user's question, and calls Gemini via `llm_provider.chat_with_gemini` (google.genai). The response and context are returned.
4. Tracing
   - `langsmith_integration.log_interaction` attempts to post a run to the configured LangSmith HTTP endpoint. If that fails or is not configured, a safe local JSON log is written to `./langsmith_logs/`.
5. Agent orchestration
   - `server.agent_orchestrate` asks Gemini which local tool to call (returns JSON specifying tool and args). The orchestrator then invokes the selected tool (search/ingest/query/ask) and returns that tool's result.

Files of interest
- [gradio_app.py](/home/sky/Desktop/project/gradio_app.py) — web UI
- [server.py](/home/sky/Desktop/project/server.py) — MCP tools and CLI fallback
- [llm_provider.py](/home/sky/Desktop/project/llm_provider.py) — Google GenAI helper
- [langsmith_integration.py](/home/sky/Desktop/project/langsmith_integration.py) — HTTP LangSmith logging
- [database.py](/home/sky/Desktop/project/database.py) — Chroma ingestion and query
- [arxiv_fetcher.py](/home/sky/Desktop/project/arxiv_fetcher.py) — arXiv download

Notes & troubleshooting
- If using a Google API key, ensure it is valid for Gemini Developer API and set in `.env` as `GOOGLE_API_KEY`.
- The google-genai SDK prefers explicit API key or application credentials; the code auto-selects a model when `GEMINI_MODEL` is not set.
- LangSmith tracing is optional — set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT` in `.env` to enable remote runs.

