"""arXiv fetching utilities

Provides functions to search arXiv, download the top PDF(s) to ./pdfs,
and return basic metadata.
"""
import os
import requests
from typing import Dict, List, Optional

import arxiv

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")


def ensure_pdf_dir() -> None:
    """Ensure the local PDFs directory exists."""
    os.makedirs(PDF_DIR, exist_ok=True)


def search_and_download(query: str, max_results: int = 1) -> List[Dict]:
    """Search arXiv for `query` and download up to `max_results` PDFs.

    Returns a list of metadata dicts with keys: title, summary, pdf_path, entry_id

    Raises:
        RuntimeError: if no matching results found or download fails for all.
    """
    ensure_pdf_dir()
    results = []
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    any_found = False
    # Use arXiv Client.results to fetch Result iterator (handles pagination).
    try:
        client = arxiv.Client()
        iterator = client.results(search)
    except Exception:
        # Fallback: some older versions exposed Search.results()
        try:
            iterator = search.results()
        except Exception:
            # As a final fallback, raise a helpful error
            raise RuntimeError("arxiv library does not expose a compatible results() API")

    for i, paper in enumerate(iterator):
        any_found = True
        try:
            # derive a safe filename
            safe_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in paper.title]).strip()
            filename = f"{safe_title[:120]}_{paper.get_short_id()}.pdf"
            file_path = os.path.join(PDF_DIR, filename)

            # If the file already exists, skip download
            if not os.path.exists(file_path):
                # Use the PDF URL via requests so we have deterministic behavior
                pdf_url = paper.pdf_url
                resp = requests.get(pdf_url, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(f"Failed to download PDF: {pdf_url} (status {resp.status_code})")
                with open(file_path, "wb") as f:
                    f.write(resp.content)

            results.append({
                "title": paper.title,
                "summary": paper.summary,
                "pdf_path": file_path,
                "entry_id": paper.entry_id,
            })
        except Exception as exc:
            # continue to attempt other results but record error
            results.append({
                "title": getattr(paper, "title", None),
                "summary": getattr(paper, "summary", None),
                "pdf_path": None,
                "entry_id": getattr(paper, "entry_id", None),
                "error": str(exc),
            })

    if not any_found:
        raise RuntimeError(f"No arXiv results for query: {query}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--max", type=int, default=1)
    args = parser.parse_args()

    print(search_and_download(args.query, args.max))
