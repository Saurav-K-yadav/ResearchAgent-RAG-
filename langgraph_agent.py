"""LangGraph-based orchestration for the local research agent.

This module replaces the hand-written orchestration logic with a small state graph:
1. decide which tool to call from the user instruction
2. execute that tool
3. return the result

The graph still uses the existing local tool implementations (arXiv search,
ChromaDB ingestion/query, and Gemini RAG answer generation), so the project
keeps the same lifecycle while gaining a graph-based structure that is easier to
extend and trace with LangSmith.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

import arxiv_fetcher
import database
from llm_provider import chat_with_gemini


class ResearchAgentState(TypedDict):
    instruction: str
    tool: str
    tool_args: Dict[str, Any]
    result: Dict[str, Any]


def search_and_download_paper(query: str, max_results: int = 1) -> list[dict]:
    """Search arXiv and download the matching paper(s)."""
    return arxiv_fetcher.search_and_download(query, max_results=max_results)


def ingest_paper(pdf_path: str, paper_title: str | None = None) -> Dict[str, Any]:
    """Ingest a local PDF into the persistent ChromaDB."""
    return database.ingest_pdf(pdf_path, paper_title=paper_title)


def query_database(question: str) -> Dict[str, Any]:
    """Query the vector store for relevant document chunks."""
    return database.query_database(question, top_k=5)


def ask_gemini(question: str) -> Dict[str, Any]:
    """Run retrieval + Gemini answer generation for a user question.

    If local chunks are too weak or irrelevant, fall back to direct Gemini rather
    than forcing the model to answer with a generic "I don't know".
    """
    retrieval = database.query_database(question, top_k=5)
    results = retrieval.get("results", [])
    contexts = [r["document"] for r in results]

    distances = [float(r.get("distance", 0.0)) for r in results if isinstance(r.get("distance"), (int, float))]
    if not contexts or (distances and all(d > 1.5 for d in distances)):
        response = chat_with_gemini(question)
        return {"answer": response.get("text"), "contexts": results, "mode": "llm_fallback"}

    prompt = """Use the following extracted passages from papers to answer the question below. If the answer is not contained, say 'I don't know'.\n\nCONTEXT:\n"""
    for i, context in enumerate(contexts, 1):
        prompt += f"[{i}] {context}\n\n"
    prompt += f"QUESTION: {question}\n\nAnswer concisely with references to the context entries."

    response = chat_with_gemini(prompt)
    return {"answer": response.get("text"), "contexts": results, "mode": "rag"}


def _parse_tool_call(raw_text: str) -> Dict[str, Any]:
    """Parse structured JSON from the model output."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if match:
            return json.loads(match.group(0))
    return {"tool": "ask", "args": {"question": raw_text}}


def decide_tool(state: ResearchAgentState) -> ResearchAgentState:
    """Ask Gemini which local tool best matches the instruction."""
    tool_list = {
        "search": "search_and_download_paper(query: str, max_results: int=1)",
        "ingest": "ingest_paper(pdf_path: str)",
        "query": "query_database(question: str)",
        "ask": "ask_gemini(question: str)",
    }
    prompt = (
        "You are the orchestration layer for a local research agent. "
        "Choose exactly one tool from the list below and return ONLY a JSON object "
        "with keys 'tool' and 'args'. "
        f"Tools: {json.dumps(tool_list)}\n"
        f"User instruction: {state['instruction']}\n"
        "Return only valid JSON."
    )

    response = chat_with_gemini(prompt)
    payload = _parse_tool_call(response.get("text", ""))
    tool = payload.get("tool")
    args = payload.get("args", {})

    if tool not in {"search", "ingest", "query", "ask"}:
        tool = "ask"
        args = {"question": state["instruction"]}

    return {**state, "tool": tool, "tool_args": args}


def execute_tool(state: ResearchAgentState) -> ResearchAgentState:
    """Execute the selected tool and store the returned payload."""
    tool_name = state.get("tool")
    args = state.get("tool_args", {})

    if tool_name == "search":
        result = search_and_download_paper(**args)
    elif tool_name == "ingest":
        result = ingest_paper(**args)
    elif tool_name == "query":
        result = query_database(**args)
    elif tool_name == "ask":
        result = ask_gemini(**args)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return {**state, "result": result}


def build_graph():
    """Create the LangGraph state graph used for orchestration."""
    builder = StateGraph(ResearchAgentState)
    builder.add_node("decide_tool", decide_tool)
    builder.add_node("execute_tool", execute_tool)
    builder.add_edge(START, "decide_tool")
    builder.add_edge("decide_tool", "execute_tool")
    builder.add_edge("execute_tool", END)
    return builder.compile()


RESEARCH_GRAPH = build_graph()


def run_agent(instruction: str) -> Dict[str, Any]:
    """Run the LangGraph agent for a user instruction."""
    state = {"instruction": instruction, "tool": "", "tool_args": {}, "result": {}}
    final_state = RESEARCH_GRAPH.invoke(state)
    return final_state.get("result", {"error": "No result produced"})
