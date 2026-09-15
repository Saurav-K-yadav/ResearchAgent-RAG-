"""Chroma DB ingestion and query utilities.

This module sets up a persistent chromadb client at ./chroma_data and
exposes functions to ingest a PDF (chunk, embed, store) and to query
for the top-k relevant chunks.
"""
from typing import List, Dict, Optional
import os
import uuid

import numpy as np
import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder for chroma persistent storage
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
COLLECTION_NAME = "papers"

# Initialize a SentenceTransformer model for embeddings
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedding_model


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts into lists of floats suitable for chromadb."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    # Ensure list of lists (python floats)
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    return embeddings


def _init_client_and_collection():
    """Initialize a persistent chromadb client and return collection.

    Uses chromadb.PersistentClient with the folder specified by CHROMA_DIR.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    # Use PersistentClient if available
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception:
        # Fallback to regular Client with persisted folder via Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))

    # collection may already exist
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)

    return client, collection


def _read_pdf_text(pdf_path: str) -> List[Dict]:
    """Read a PDF and return a list of page dicts: [{page_num: int, text: str}]."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def chunk_texts(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Chunk a list of pages (dicts with text) into chunks with metadata.

    Returns list of items: {id, text, metadata}
    metadata includes: page_num, page_range (start-end)
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    items = []
    for p in pages:
        text = p.get("text", "")
        if not text or text.strip() == "":
            continue
        chunks = splitter.split_text(text)
        for idx, ch in enumerate(chunks):
            items.append({
                "text": ch,
                "metadata": {"page_num": p.get("page_num")},
                "id": str(uuid.uuid4()),
            })
    return items


def ingest_pdf(pdf_path: str, paper_title: Optional[str] = None) -> Dict:
    """Read a PDF, chunk it, embed chunks, and store them in the persistent Chroma DB.

    Returns a summary dict with counts and the collection name.
    """
    pages = _read_pdf_text(pdf_path)
    if len(pages) == 0:
        raise RuntimeError("PDF contained no extractable text")

    items = chunk_texts(pages, chunk_size=1000, overlap=200)
    if not items:
        raise RuntimeError("No chunks produced from PDF")

    texts = [it["text"] for it in items]
    ids = [it["id"] for it in items]
    metadatas = [
        {**it["metadata"], "paper_title": paper_title or os.path.basename(pdf_path)} for it in items
    ]

    embeddings = _embed_texts(texts)

    client, collection = _init_client_and_collection()

    # Add to collection
    try:
        collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    except TypeError:
        # Some chroma versions expect positional args or different names; try a tolerant add
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        # try upserting embeddings via separate API if available

    return {"stored_chunks": len(texts), "collection": COLLECTION_NAME}


def query_database(question: str, top_k: int = 5) -> Dict:
    """Query the Chroma DB with `question` and return the top_k results.

    Returns a dict with keys: query, results where results is a list of dicts containing
    document text, metadata and distance.
    """
    client, collection = _init_client_and_collection()
    # If collection is empty, return empty results
    try:
        count = collection.count()
    except Exception:
        # Some chroma clients don't expose count()
        count = None

    if count == 0:
        return {"query": question, "results": []}

    q_emb = _embed_texts([question])[0]

    # Query collection
    try:
        resp = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    except TypeError:
        resp = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])

    results = []
    # chroma returns nested lists (one query -> list of lists)
    docs = resp.get("documents", [[]])[0]
    metas = resp.get("metadatas", [[]])[0]
    dists = resp.get("distances", [[]])
    dists = dists[0] if dists else [None] * len(docs)

    for d, m, dist in zip(docs, metas, dists):
        results.append({"document": d, "metadata": m, "distance": dist})

    return {"query": question, "results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--title")
    args = parser.parse_args()

    print(ingest_pdf(args.pdf_path, paper_title=args.title))
