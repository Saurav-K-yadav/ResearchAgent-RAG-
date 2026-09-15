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


def format_search_results(payload):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return payload

    if not payload:
        return "No papers were found for your search. Try a broader keyword."

    if isinstance(payload, list):
        lines = []
        for i, item in enumerate(payload, 1):
            if isinstance(item, dict):
                title = item.get('title') or item.get('entry_id') or f'Paper {i}'
                summary = item.get('summary') or 'No summary available.'
                pdf_path = item.get('pdf_path') or item.get('path') or 'Not downloaded'
                lines.append(f"{i}. {title}\n   Summary: {summary[:300]}\n   File: {pdf_path}")
            elif item:
                lines.append(f"{i}. {str(item)}")
        return "\n\n".join(lines)

    return json.dumps(payload, indent=2)


def format_ingest_result(payload):
    if isinstance(payload, dict):
        chunks = payload.get('stored_chunks', 0)
        collection = payload.get('collection', 'papers')
        if payload.get('error'):
            return f"Ingestion failed: {payload['error']}"
        return f"PDF ingested successfully.\nStored chunks: {chunks}\nCollection: {collection}"
    return str(payload)


def format_answer_result(payload):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return payload

    if isinstance(payload, dict):
        answer = payload.get('answer') or 'No answer produced.'
        mode = payload.get('mode') or 'rag'
        contexts = payload.get('contexts') or []

        import re
        # Clean and shorten overly verbose model guidance about uploads
        guidance_triggers = [
            'please provide the paper', 'please provide', 'upload the file', 'paste the text', 'provide a link',
            'paste it', 'attach the file', 'upload', 'paste'
        ]
        ans = str(answer).strip()
        low = ans.lower()
        cut_at = None
        for trig in guidance_triggers:
            i = low.find(trig)
            if i != -1:
                cut_at = i
                break
        if cut_at is not None:
            ans = ans[:cut_at].strip()
        # Collapse excessive blank lines
        ans = re.sub(r"\n{3,}", "\n\n", ans)
        # Trim trailing unfinished conjunctions left by cutting
        ans = re.sub(r"[\s,;:\-]*(?:or|and|but|please|to|then|otherwise)[\s\W]*$", "", ans, flags=re.I)
        # Ensure it ends with a single period before adding guidance
        ans = ans.rstrip('.') + '.' if ans and not ans.endswith('.') else ans
        # Add a concise, user-friendly guidance note if we removed verbose upload instructions
        if cut_at is not None:
            ans = ans + '\n\nNote: No paper-specific context is available. To get a paper-specific summary, ingest the PDF via the "Build your knowledge base" tab or paste the excerpt you want analyzed.'
        # Limit answer length for UI
        if len(ans) > 2000:
            ans = ans[:2000].rstrip() + '\n\n[Answer truncated]'

        lines = [f"Mode: {mode}", '', 'Answer:', ans]
        if contexts:
            lines.append('')
            lines.append('Relevant passages:')
            for idx, context in enumerate(contexts[:3], 1):
                doc = context.get('document', '') if isinstance(context, dict) else str(context)
                meta = context.get('metadata', {}) if isinstance(context, dict) else {}
                title = meta.get('paper_title', 'Unknown paper')
                page = meta.get('page_num', 'n/a')
                excerpt = ' '.join(doc.splitlines())[:250]
                lines.append(f"{idx}. {title} (page {page})\n   {excerpt}...")
        return "\n\n".join(lines)

    return str(payload)


def ui_search(query: str, max_results: int = 1):
    try:
        res = server.search_and_download_paper(query, max_results=max_results)
        txt = format_search_results(res)
        pdf_choices = list_pdfs()
        return txt, gr.update(visible=True), gr.update(choices=pdf_choices, value=None)
    except Exception as e:
        return f"Search failed. {e}", gr.update(visible=True), gr.update(choices=list_pdfs(), value=None)


def ui_ingest(selected_pdf: str, title: str = None):
    try:
        if not selected_pdf:
            return "No PDF selected. Choose one from the list first.", gr.update(visible=True), gr.update(choices=list_pdfs(), value=None)
        pdf_path = os.path.join(PDF_DIR, selected_pdf)
        res = server.ingest_paper(pdf_path, paper_title=title)
        txt = format_ingest_result(res)
        refreshed = list_pdfs()
        return txt, gr.update(visible=True), gr.update(choices=refreshed, value=selected_pdf if selected_pdf in refreshed else None)
    except Exception as e:
        return f"Ingestion failed. {e}", gr.update(visible=True), gr.update(choices=list_pdfs(), value=None)


def ui_query(question: str):
    try:
        res = server.ask_gemini(question)
        txt = format_answer_result(res)
        return txt, gr.update(visible=True)
    except Exception as e:
        return f"Answering failed. {e}", gr.update(visible=True)


def ui_orchestrate(instruction: str):
    try:
        res = server.agent_orchestrate(instruction)
        if isinstance(res, dict) and 'error' in res:
            return f"Agent could not complete the task: {res['error']}", gr.update(visible=True)
        txt = format_answer_result(res)
        return txt, gr.update(visible=True)
    except Exception as e:
        return f"Agent task failed. {e}", gr.update(visible=True)


with gr.Blocks(title="ResearchAgent") as demo:
    gr.Markdown("# ResearchAgent — Local Research Workflow")
    gr.Markdown(
        "ResearchAgent provides a lightweight, local research pipeline: discover open papers on arXiv, add selected PDFs to a local vector-backed knowledge base, and ask focused, grounded questions." 
        "Use the tabs below to progress through the workflow."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Workflow")
            gr.Markdown("1. Find papers on arXiv\n2. Ingest PDFs into your local library\n3. Ask focused questions using retrieved context\n4. Use the agent to automate compound tasks")
        with gr.Column(scale=2):
            gr.Markdown("#### Quick tips")
            gr.Markdown("- Ingest PDFs into the 'Build your knowledge base' tab so answers can be grounded in paper content.\n- If no local context is available the assistant will provide a concise general answer and a one-line note on how to add paper context.")

    with gr.Tab('Find papers'):
        gr.Markdown('Search arXiv and download PDFs. Use specific keywords or author names for better results.')
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(label='Search arXiv', value='quantum computing', placeholder='e.g. graph neural networks, reinforcement learning')
                maxr = gr.Number(label='How many papers?', value=1, precision=0, minimum=1, maximum=10)
            with gr.Column(scale=1):
                btn_search = gr.Button('Search & download', variant='primary')
        out_search = gr.Textbox(label='Search results', lines=12, visible=False)
        # click binding moved below after all components are defined so we can update the PDF dropdown choices

    with gr.Tab('Build your knowledge base'):
        gr.Markdown('Select a downloaded PDF and ingest it into the persistent local vector store (Chroma). This enables grounded answers from your own library.')
        with gr.Row():
            with gr.Column(scale=3):
                pdf_dropdown = gr.Dropdown(label='Choose a downloaded PDF', choices=list_pdfs(), interactive=True, value=None)
                title_in = gr.Textbox(label='Optional paper title override', placeholder='Leave blank to use the file name')
            with gr.Column(scale=1):
                btn_refresh = gr.Button('Refresh list')
                btn_ingest = gr.Button('Ingest PDF', variant='primary')
        out_ingest = gr.Textbox(label='Ingestion summary', lines=6, visible=False)
        btn_refresh.click(lambda: list_pdfs(), None, pdf_dropdown)
        # btn_ingest click binding moved below so the PDF dropdown can be refreshed after ingestion

    with gr.Tab('Ask a question'):
        gr.Markdown('Ask focused questions. The system will use local paper passages when available; otherwise it falls back to a concise general answer.')
        question = gr.Textbox(label='Your question', placeholder='e.g. What methods did the paper use to evaluate performance?')
        out_ask = gr.Textbox(label='Answer and relevant context', lines=14, visible=False)
        btn_ask = gr.Button('Ask ResearchAgent', variant='primary')
        # click binding moved below

    with gr.Tab('Agent helper'):
        gr.Markdown('Provide a high-level instruction and the agent will decide whether to search, ingest, query, or answer.')
        instr = gr.Textbox(label='Tell the agent what to do', placeholder='e.g. find recent papers on transformers for graphs and summarize the top result')
        out_orch = gr.Textbox(label='Agent result', lines=14, visible=False)
        btn_orch = gr.Button('Run agent', variant='primary')
        # click binding moved below

    # Bind events now that all components exist (allows updating the PDF dropdown after downloads)
    btn_search.click(ui_search, inputs=[query, maxr], outputs=[out_search, out_search, pdf_dropdown])
    btn_ingest.click(ui_ingest, inputs=[pdf_dropdown, title_in], outputs=[out_ingest, out_ingest, pdf_dropdown])
    btn_ask.click(ui_query, inputs=[question], outputs=[out_ask, out_ask])
    btn_orch.click(ui_orchestrate, inputs=[instr], outputs=[out_orch, out_orch])

    if __name__ == '__main__':
        # Launch the app on port 7860
        demo.launch(server_name='0.0.0.0', server_port=7860)
