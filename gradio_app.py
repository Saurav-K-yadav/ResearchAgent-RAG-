"""Gradio UI for the ResearchAgent project.

Provides a simple web UI to:
- Search arXiv and download PDFs
- Ingest downloaded PDFs into ChromaDB
- Run retrieval-augmented generation via Gemini (ask)
- Use the agent orchestrator to perform high-level instructions

Run: python gradio_app.py
"""
from __future__ import annotations
import os
import json
import traceback
from typing import List

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

import gradio as gr

# Import the project's functions
import server
import arxiv_fetcher
import database

PDF_DIR = os.path.join(os.path.dirname(__file__), 'pdfs')


def list_pdfs() -> List[str]:
    if not os.path.exists(PDF_DIR):
        return []
    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    return files


def ui_search(query: str, max_results: int = 1):
    try:
        res = server.search_and_download_paper(query, max_results=max_results)
        # present as JSON and list of paths
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"ERROR: {e}\n" + traceback.format_exc()


def ui_ingest(selected_pdf: str, title: str = None):
    try:
        if not selected_pdf:
            return "No PDF selected"
        pdf_path = os.path.join(PDF_DIR, selected_pdf)
        res = server.ingest_paper(pdf_path, paper_title=title)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"ERROR: {e}\n" + traceback.format_exc()


def ui_query(question: str):
    try:
        # prefer the RAG-enabled ask_gemini for concise answers
        res = server.ask_gemini(question)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"ERROR: {e}\n" + traceback.format_exc()


def ui_orchestrate(instruction: str):
    try:
        res = server.agent_orchestrate(instruction)
        return json.dumps(res, indent=2)
    except Exception as e:
        return f"ERROR: {e}\n" + traceback.format_exc()


with gr.Blocks(title="ResearchAgent UI") as demo:
    gr.Markdown("# ResearchAgent — MCP + Gemini + LangSmith UI")

    with gr.Tab('Search & Download'):
        query = gr.Textbox(label='Search query', value='quantum computing')
        maxr = gr.Number(label='Max results', value=1, precision=0)
        out_search = gr.Textbox(label='Search result (JSON)', lines=10)
        btn_search = gr.Button('Search & Download')
        btn_search.click(ui_search, inputs=[query, maxr], outputs=[out_search])

    with gr.Tab('Ingest'):
        pdf_dropdown = gr.Dropdown(label='Select downloaded PDF', choices=list_pdfs(), interactive=True)
        title_in = gr.Textbox(label='Optional title override')
        out_ingest = gr.Textbox(label='Ingest result (JSON)', lines=6)
        btn_refresh = gr.Button('Refresh PDF list')
        btn_ingest = gr.Button('Ingest selected PDF')
        btn_refresh.click(lambda: list_pdfs(), None, pdf_dropdown)
        btn_ingest.click(ui_ingest, inputs=[pdf_dropdown, title_in], outputs=[out_ingest])

    with gr.Tab('Ask (RAG)'):
        question = gr.Textbox(label='Question')
        out_ask = gr.Textbox(label='Answer & Contexts (JSON)', lines=12)
        btn_ask = gr.Button('Ask Gemini (RAG)')
        btn_ask.click(ui_query, inputs=[question], outputs=[out_ask])

    with gr.Tab('Agent Orchestrator'):
        instr = gr.Textbox(label='Instruction for agent (English)')
        out_orch = gr.Textbox(label='Agent response / tool output (JSON)', lines=12)
        btn_orch = gr.Button('Run Orchestrator')
        btn_orch.click(ui_orchestrate, inputs=[instr], outputs=[out_orch])

    gr.Markdown('Logs and LangSmith traces are written to ./langsmith_logs (local fallback) unless remote tracing is configured in .env')

if __name__ == '__main__':
    # Launch the app on port 7860
    demo.launch(server_name='0.0.0.0', server_port=7860)
