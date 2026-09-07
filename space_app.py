"""A minimal Gradio demo for Hugging Face Spaces that works without API keys.

This small app demonstrates the workflow on a local example file from ./examples/
and provides a simple keyword-based passage retrieval so visitors can try the UI
without configuring external credentials.
"""
from __future__ import annotations
import os
from typing import List
import gradio as gr

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

# Small generation model used for demo. This is lightweight enough for CPU-bound Spaces in many cases.
DEMO_MODEL = "google/flan-t5-small"


def list_examples() -> List[str]:
    if not os.path.exists(EXAMPLES_DIR):
        return []
    return [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith('.txt')]


def read_paragraphs(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split into paragraphs by double newlines
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paras


# Lazy-load the generation pipeline to avoid heavy startup in non-demo contexts
_generator = None


def _load_generator():
    global _generator
    if _generator is not None:
        return _generator
    try:
        from transformers import pipeline
        # use text2text-generation for FLAN-T5 models
        _generator = pipeline("text2text-generation", model=DEMO_MODEL, device=-1)
    except Exception as e:
        # Fail silently — fallback to retrieval-only behavior
        _generator = None
    return _generator


def _generate_answer(passages: str, question: str) -> str:
    gen = _load_generator()
    prompt = f"Use the following excerpts from a paper to answer the question. If the answer is not present, be concise and indicate missing details.\n\nExcerpts:\n{passages}\n\nQuestion: {question}\nAnswer:"
    if gen is None:
        # No generator available; return passages as context
        return "\n\n---\n\n".join([p.strip() for p in passages.split('\n\n---\n\n') if p.strip()])
    try:
        out = gen(prompt, max_length=256, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get('generated_text') or str(out[0])
        return str(out)
    except Exception:
        return "(model generation failed)\n\n" + passages


def simple_retrieve(question: str, filename: str):
    if not filename:
        return "No example selected. Please pick an example file.", None
    path = os.path.join(EXAMPLES_DIR, filename)
    if not os.path.exists(path):
        return f"Example file not found: {filename}", None
    paras = read_paragraphs(path)
    q_terms = set([t.lower() for t in question.split() if t.strip()])
    scored = []
    for p in paras:
        p_terms = set([t.lower().strip('.,()[];:\"\'') for t in p.split() if t.strip()])
        score = len(q_terms & p_terms)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    # pick top passages with non-zero score else return leading paragraph
    if scored and scored[0][0] > 0:
        top = [p for s, p in scored[:3] if s > 0]
        passages = "\n\n---\n\n".join(top)
        # generate answer from small model
        answer = _generate_answer(passages, question)
        return answer, gr.update(visible=True)
    else:
        # fallback: return the first paragraph as a short summary
        fallback = paras[0] if paras else "(empty file)"
        # try to generate a brief answer from fallback as context
        answer = _generate_answer(fallback, question)
        return answer, gr.update(visible=True)


with gr.Blocks(title="ResearchAgent — Demo") as demo:
    gr.Markdown("# ResearchAgent — Demo\n\nTry the demo mode which uses a local example paper and a small on-host model to generate concise answers. This runs without external API keys (the model will be downloaded by the runtime on first run).")
    with gr.Row():
        with gr.Column(scale=3):
            example_dropdown = gr.Dropdown(label='Choose demo paper (local)', choices=list_examples(), value=(list_examples()[0] if list_examples() else None))
            question = gr.Textbox(label='Ask a question about the example', placeholder='e.g. What is the main contribution?')
            btn = gr.Button('Run demo', variant='primary')
        with gr.Column(scale=2):
            out = gr.Textbox(label='Demo answer / matched passages', lines=12, visible=False)
    btn.click(fn=simple_retrieve, inputs=[question, example_dropdown], outputs=[out, out])

# Expose `app` for Spaces runtime
app = demo
