"""Main MCP server exposing tools for searching, ingesting, and querying papers.

This file defines three tools used by the FastMCP agent:
- search_and_download_paper
- ingest_paper
- query_database

Run with: python server.py  (requires a compatible `mcp` runtime)
"""
from typing import Optional, List, Dict
import os
import sys

import arxiv_fetcher
import database


def _direct_gemini_answer(question: str) -> str:
    """Ask Gemini without any local retrieval context as a fallback.

    Provide a system-level instruction so the model understands it has no local
    papers available and should not prompt the user to upload files verbosely.
    Instead it should give a concise general answer and a short single-line
    instruction on how the user can provide the paper for a targeted answer.
    """
    from llm_provider import chat_with_gemini

    system_msg = (
        "You are a concise research assistant. Answer the user's question to the best "
        "of your general knowledge. If the user asks about a specific paper but no "
        "paper context is provided, do not request uploads or paste instructions in "
        "long form. Instead, give a short factual answer if possible, then append a "
        "single short sentence explaining how the user can provide paper context for a "
        "more accurate, paper-specific summary (for example: 'To analyze a specific "
        "paper, please ingest its PDF via the application's 'Build your knowledge base' tab, "
        "or paste the relevant excerpt.'). Keep this guidance to one sentence."
    )

    resp = chat_with_gemini(question, system_message=system_msg)
    return resp.get("text", "")


def _rag_or_llm_fallback(question: str) -> Dict:
    """Answer using retrieved context when it is relevant; otherwise fall back to Gemini."""
    retrieval = database.query_database(question, top_k=5)
    results = retrieval.get("results", [])
    contexts = [r["document"] for r in results]

    if not results:
        answer = _direct_gemini_answer(question)
        return {"answer": answer, "contexts": [], "mode": "llm_fallback"}

    distances = [float(r.get("distance", 0.0)) for r in results if isinstance(r.get("distance"), (int, float))]
    if distances and all(d > 1.5 for d in distances):
        answer = _direct_gemini_answer(question)
        return {"answer": answer, "contexts": results, "mode": "llm_fallback"}

    prompt = """Use the following extracted passages from papers to answer the question below. If the answer is not contained, say 'I don't know'.\n\nCONTEXT:\n"""
    for i, c in enumerate(contexts, 1):
        prompt += f"[{i}] {c}\n\n"
    prompt += f"QUESTION: {question}\n\nAnswer concisely with references to the context entries."

    from llm_provider import chat_with_gemini

    resp = chat_with_gemini(prompt)
    return {"answer": resp.get("text"), "contexts": results, "mode": "rag"}


try:
    from langgraph_agent import run_agent as run_langgraph_agent  # type: ignore
except Exception:
    run_langgraph_agent = None

# Try to use FastMCP if available; otherwise provide a simple CLI fallback
try:
    from mcp import FastMCP  # type: ignore
    _HAS_FAST_MCP = True
except Exception:
    _HAS_FAST_MCP = False

if _HAS_FAST_MCP:
    mcp = FastMCP("ResearchAgent")


    @mcp.tool()
    def search_and_download_paper(query: str, max_results: int = 1) -> List[Dict]:
        """Search arXiv for `query`, download up to `max_results` PDFs, and return metadata.

        This tool downloads PDFs into the local `pdfs/` folder and returns a list of
        metadata dictionaries with keys: title, summary, pdf_path, entry_id. If a paper
        failed to download, the corresponding dict will include an "error" key.
        """
        try:
            results = arxiv_fetcher.search_and_download(query, max_results=max_results)
            return results
        except Exception as exc:
            return [{"error": str(exc)}]


    @mcp.tool()
    def ingest_paper(pdf_path: str, paper_title: Optional[str] = None) -> Dict:
        """Ingest a local PDF into the persistent ChromaDB.

        Arguments:
            pdf_path: Path to a local PDF file (absolute or relative to project root).
            paper_title: Optional human-readable title to attach to chunks.

        Returns a summary dict with the number of stored chunks and the collection name.
        """
        try:
            if not os.path.isabs(pdf_path):
                pdf_path = os.path.join(os.path.dirname(__file__), pdf_path)
            result = database.ingest_pdf(pdf_path, paper_title=paper_title)
            return result
        except FileNotFoundError:
            return {"error": f"File not found: {pdf_path}"}
        except Exception as exc:
            return {"error": str(exc)}


    @mcp.tool()
    def query_database(question: str) -> Dict:
        """Query the local ChromaDB using `question` and return top 5 relevant chunks.

        Returns a dict with 'query' and 'results' where each result contains the chunk
        text and associated metadata (page numbers, paper title) and a distance score.
        """
        try:
            resp = database.query_database(question, top_k=5)
            return resp
        except Exception as exc:
            return {"error": str(exc)}


    # New tool: ask_gemini combines RAG retrieval with Gemini LLM and (optionally) logs to LangSmith
    if _HAS_FAST_MCP:
        @mcp.tool()
        def ask_gemini(question: str) -> Dict:
            """Run RAG when local context exists; otherwise fallback to direct Gemini.

            This keeps retrieval answers grounded when documents are available, but it
            avoids hard-coded ``I don't know`` results when the database is empty.
            """
            try:
                return _rag_or_llm_fallback(question)
            except Exception as exc:
                return {"error": str(exc)}


            @mcp.tool()
            def agent_orchestrate(instruction: str) -> Dict:
                """High-level orchestrator powered by a LangGraph state machine.

                The graph decides which research tool to call based on the instruction,
                executes it, and returns the result. This replaces the earlier ad-hoc
                Python orchestration flow with a graph-based workflow that is easier to
                extend and trace in LangSmith.
                """
                if run_langgraph_agent is not None:
                    return run_langgraph_agent(instruction)

                import json
                from llm_provider import chat_with_gemini
                tool_list = {
                    "search": "search_and_download_paper(query: str, max_results: int=1)",
                    "ingest": "ingest_paper(pdf_path: str)",
                    "query": "query_database(question: str)",
                    "ask": "ask_gemini(question: str)",
                }
                prompt = "You are an agent orchestrator. Decide one tool to call from the following list and return a JSON object with 'tool' and 'args'. Tools: " + json.dumps(tool_list) + "\nUser instruction: " + instruction + "\nReturn only valid JSON."
                try:
                    resp = chat_with_gemini(prompt)
                    text = resp.get("text", "")
                    try:
                        payload = json.loads(text)
                    except Exception:
                        import re
                        m = re.search(r"\{.*\}", text, flags=re.S)
                        if not m:
                            return {"error": "Could not parse JSON from model output", "raw": text}
                        payload = json.loads(m.group(0))

                    tool = payload.get("tool")
                    args = payload.get("args", {})
                    if tool == "search":
                        return search_and_download_paper(**args)
                    if tool == "ingest":
                        return ingest_paper(**args)
                    if tool == "query":
                        return query_database(**args)
                    if tool == "ask":
                        return ask_gemini(**args)
                    return {"error": f"Unknown tool: {tool}", "raw_payload": payload}
                except Exception as exc:
                    return {"error": str(exc)}


    # When FastMCP is not available the CLI fallback exposes the ask_gemini command
    else:
        def ask_gemini(question: str) -> Dict:
            try:
                return _rag_or_llm_fallback(question)
            except Exception as exc:
                return {"error": str(exc)}

        def ask_gemini(question: str) -> Dict:
            try:
                return _rag_or_llm_fallback(question)
            except Exception as exc:
                return {"error": str(exc)}


    if __name__ == "__main__":
        # Run the MCP agent over stdio as requested in the instructions
        if _HAS_FAST_MCP:
            mcp.run_stdio()
        else:
            # CLI fallback (keeps previous behavior)
            if len(sys.argv) < 2:
                _usage()
                sys.exit(0)

            cmd = sys.argv[1]
            if cmd == "search":
                if len(sys.argv) < 3:
                    print("search requires a query string")
                    sys.exit(2)
                q = sys.argv[2]
                maxr = int(sys.argv[3]) if len(sys.argv) > 3 else 1
                res = search_and_download_paper(q, max_results=maxr)
                print(res)
            elif cmd == "ingest":
                if len(sys.argv) < 3:
                    print("ingest requires a pdf_path")
                    sys.exit(2)
                path = sys.argv[2]
                title = sys.argv[3] if len(sys.argv) > 3 else None
                res = ingest_paper(path, paper_title=title)
                print(res)
            elif cmd == "query":
                if len(sys.argv) < 3:
                    print("query requires a question string")
                    sys.exit(2)
                q = sys.argv[2]
                res = query_database(q)
                print(res)
            elif cmd == "ask":
                if len(sys.argv) < 3:
                    print("ask requires a question string")
                    sys.exit(2)
                q = sys.argv[2]
                res = ask_gemini(q)
                print(res)
            else:
                _usage()
                sys.exit(2)

else:
    # CLI fallback for local testing and demonstration when the installed `mcp`
    # package does not provide FastMCP. This keeps the project runnable and
    # allows interactive search/ingest/query operations.
    def search_and_download_paper(query: str, max_results: int = 1) -> List[Dict]:
        """CLI wrapper: call arXiv search and download."""
        return arxiv_fetcher.search_and_download(query, max_results=max_results)

    def ingest_paper(pdf_path: str, paper_title: Optional[str] = None) -> Dict:
        """CLI wrapper: ingest PDF into local ChromaDB."""
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(os.path.dirname(__file__), pdf_path)
        return database.ingest_pdf(pdf_path, paper_title=paper_title)

    def query_database(question: str) -> Dict:
        """CLI wrapper: query the local vector DB."""
        return database.query_database(question, top_k=5)

    def _usage():
        print("Usage: python server.py [command] [args]")
        print("Commands:")
        print("  search <query> [max_results]      - search arXiv and download PDFs")
        print("  ingest <pdf_path> [title]         - ingest a local PDF into ChromaDB")
        print("  query <question>                  - query the local vector DB (top 5)")
        print("  ask <question>                    - run retrieval + Gemini answer")

    if __name__ == "__main__":
        if len(sys.argv) < 2:
            _usage()
            sys.exit(0)

        cmd = sys.argv[1]
        if cmd == "search":
            if len(sys.argv) < 3:
                print("search requires a query string")
                sys.exit(2)
            q = sys.argv[2]
            maxr = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            res = search_and_download_paper(q, max_results=maxr)
            print(res)
        elif cmd == "ingest":
            if len(sys.argv) < 3:
                print("ingest requires a pdf_path")
                sys.exit(2)
            path = sys.argv[2]
            title = sys.argv[3] if len(sys.argv) > 3 else None
            res = ingest_paper(path, paper_title=title)
            print(res)
        elif cmd == "query":
            if len(sys.argv) < 3:
                print("query requires a question string")
                sys.exit(2)
            q = sys.argv[2]
            res = query_database(q)
            print(res)
        elif cmd == "ask":
            if len(sys.argv) < 3:
                print("ask requires a question string")
                sys.exit(2)
            q = sys.argv[2]
            # Define ask_gemini here if it wasn't defined earlier
            if 'ask_gemini' not in globals():
                def ask_gemini(question: str) -> Dict:
                    try:
                        return _rag_or_llm_fallback(question)
                    except Exception as exc:
                        return {"error": str(exc)}
                globals()['ask_gemini'] = ask_gemini
            res = ask_gemini(q)
            print(res)
        elif cmd == "orchestrate":
            if len(sys.argv) < 3:
                print("orchestrate requires an instruction string")
                sys.exit(2)
            instr = sys.argv[2]
            # define orchestrator if missing
            if 'agent_orchestrate' not in globals():
                def agent_orchestrate(instruction: str) -> Dict:
                    if run_langgraph_agent is not None:
                        return run_langgraph_agent(instruction)
                    from llm_provider import chat_with_gemini
                    import json, re
                    tool_list = {
                        "search": "search_and_download_paper(query: str, max_results: int=1)",
                        "ingest": "ingest_paper(pdf_path: str)",
                        "query": "query_database(question: str)",
                        "ask": "ask_gemini(question: str)",
                    }
                    prompt = "You are an agent orchestrator. Decide one tool to call from the following list and return a JSON object with 'tool' and 'args'. Tools: " + json.dumps(tool_list) + "\nUser instruction: " + instruction + "\nReturn only valid JSON."
                    resp = chat_with_gemini(prompt)
                    text = resp.get('text', '')
                    try:
                        payload = json.loads(text)
                    except Exception:
                        m = re.search(r"\{.*\}", text, flags=re.S)
                        if not m:
                            return {"error": "Could not parse JSON from model output", "raw": text}
                        payload = json.loads(m.group(0))
                    tool = payload.get('tool')
                    args = payload.get('args', {})
                    if tool == 'search':
                        return search_and_download_paper(**args)
                    if tool == 'ingest':
                        return ingest_paper(**args)
                    if tool == 'query':
                        return query_database(**args)
                    if tool == 'ask':
                        return ask_gemini(**args)
                    return {"error": f"Unknown tool: {tool}", "raw_payload": payload}
                globals()['agent_orchestrate'] = agent_orchestrate
            res = agent_orchestrate(instr)
            print(res)
        else:
            _usage()
            sys.exit(2)

# Ensure ask_gemini is available at import time for programmatic use (CLI also wired above)
if 'ask_gemini' not in globals():
    def ask_gemini(question: str) -> Dict:
        try:
            return _rag_or_llm_fallback(question)
        except Exception as exc:
            return {"error": str(exc)}
    globals()['ask_gemini'] = ask_gemini

# Provide an alias for a nicer public API name
def ask_researchagent(question: str) -> Dict:
    """Compatibility alias: ask_researchagent calls the RAG-enabled ask_gemini flow."""
    return ask_gemini(question)
globals()['ask_researchagent'] = ask_researchagent

# Ensure agent_orchestrate is available programmatically
if 'agent_orchestrate' not in globals():
    def agent_orchestrate(instruction: str) -> Dict:
        if run_langgraph_agent is not None:
            return run_langgraph_agent(instruction)
        import json, re
        from llm_provider import chat_with_gemini
        tool_list = {
            "search": "search_and_download_paper(query: str, max_results: int=1)",
            "ingest": "ingest_paper(pdf_path: str)",
            "query": "query_database(question: str)",
            "ask": "ask_gemini(question: str)",
        }
        prompt = "You are an agent orchestrator. Decide one tool to call from the following list and return a JSON object with 'tool' and 'args'. Tools: " + json.dumps(tool_list) + "\nUser instruction: " + instruction + "\nReturn only valid JSON."
        resp = chat_with_gemini(prompt)
        text = resp.get('text', '')
        try:
            payload = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                return {"error": "Could not parse JSON from model output", "raw": text}
            payload = json.loads(m.group(0))
        tool = payload.get('tool')
        args = payload.get('args', {})
        if tool == 'search':
            return search_and_download_paper(**args)
        if tool == 'ingest':
            return ingest_paper(**args)
        if tool == 'query':
            return query_database(**args)
        if tool == 'ask':
            return ask_gemini(**args)
        return {"error": f"Unknown tool: {tool}", "raw_payload": payload}
    globals()['agent_orchestrate'] = agent_orchestrate
