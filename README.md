# ResearchAgent

I built a local-first research assistant for academic paper workflows. It helps me search arXiv, download PDFs, ingest them into a local vector database, and ask grounded questions using Gemini.

I have done the following:
- built the end-to-end workflow for paper search and PDF download
- parsed PDF text and split it into chunks for retrieval
- stored embeddings in a local ChromaDB database
- connected Gemini for answer generation and fallback behavior
- added a LangGraph-based agent to decide which tool to use
- added LangSmith tracing with a local fallback when remote tracing is unavailable
- built a polished Gradio interface with tabs for search, ingest, ask, and agent actions
- uploaded the project to Hugging Face with the HF CLI

What I am using in the project now:
- Python for the project logic
- ChromaDB for local vector search
- SentenceTransformers for embeddings
- Google Gemini for answer generation
- LangGraph for orchestration
- LangSmith for tracing when configured
- Gradio for the UI

What I still need to do:
- add OCR support for scanned PDFs that do not contain selectable text
- add automated tests for the retrieval and answer flow
- add a cleaner project screenshot and a stronger demo landing page
- simplify the setup instructions for non-developer users
- add a deployment config for a more production-ready Hugging Face Space setup

How I run it locally:
1. Create a virtual environment
2. Install the dependencies from requirements.txt
3. Add the required environment values in .env
4. Run the Gradio app
5. Use the workflow in the UI: search -> ingest -> ask -> agent

This version is a working version of the project. It is functional, local-first, and ready for further iteration.

