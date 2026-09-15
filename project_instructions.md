# Project Overview
I want to build a local Agentic workflow using the Model Context Protocol (MCP). It is an "AI Research Scientist" agent that allows Claude Desktop to autonomously search arXiv, download PDF papers, process them into a local Vector Database, and query them using RAG.

# Tech Stack & Requirements
- **Language:** Python 3.11+
- **MCP Framework:** `mcp` (server exposes FastMCP tools when available; server.py includes a CLI fallback)
- **Agent orchestration:** `langgraph` state machine for deciding which tool to invoke and routing the workflow
- **Model / LLM:** Google Gemini via the `google-genai` SDK (uses API key from .env)
- **LangSmith Tracing:** Direct HTTP integration to LangSmith endpoint (LANGSMITH_ENDPOINT) using the `X-Api-Key` header and required `run_type` payload field, with safe local fallback logs
- **Vector DB:** `chromadb` (PersistentClient, saves to `./chroma_data`)
- **Embeddings:** `sentence-transformers` (Use `all-MiniLM-L6-v2` for local, free embeddings)
- **PDF Parsing:** `pymupdf` (fitz)
- **arXiv API:** `arxiv` python library
- **Text Chunking:** `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- **UI:** `gradio` for a simple browser UI (gradio_app.py)

Note: The project intentionally avoids any OpenAI API keys — it uses Google Gemini locally via API key.

# File Architecture
Please generate the following files:

1. `requirements.txt`: All necessary dependencies.
2. `arxiv_fetcher.py`: Logic to search arXiv, download the top PDF to a local `pdfs/` folder, and return the metadata (title, summary, file path).
3. `database.py`: Logic to initialize persistent ChromaDB. Contains functions to:
   - Chunk text using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200).
   - Read a PDF using PyMuPDF, chunk it, embed it using SentenceTransformers, and store it in ChromaDB. Include metadata like page numbers and paper title.
   - Query ChromaDB with a string, embed the string, and return the top 5 most relevant chunks.
4. `server.py`: The main entry point. Initialize `FastMCP("ResearchAgent")`. Expose 3 tools using the `@mcp.tool()` decorator:
   - `search_and_download_paper(query: str, max_results: int = 1)` -> Uses arxiv_fetcher.
   - `ingest_paper(pdf_path: str)` -> Uses database.py to process the PDF.
   - `query_database(question: str)` -> Uses database.py to search the Vector DB.
   - Note: The server must run using `mcp.run_stdio()` at the end of the file.

# Important Constraints
- NO OpenAI API keys. Everything must run 100% locally.
- Ensure all functions have clear Python docstrings, as FastMCP uses these docstrings to tell Claude what the tool does.
- Handle basic errors (e.g., if a PDF fails to download, or a file isn't found).