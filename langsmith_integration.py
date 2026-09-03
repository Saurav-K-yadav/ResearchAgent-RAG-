"""Direct LangSmith HTTP integration with safe local fallback.

This module provides a `log_interaction(prompt, response_text, model)` helper
that attempts to record the interaction to the LangSmith HTTP API using the
LANGSMITH_ENDPOINT and LANGSMITH_API_KEY from .env. If the HTTP call fails,
falls back to writing a safe local JSON log under ./langsmith_logs/.

To keep this implementation resilient, the function will never include API
keys in logs and will truncate prompt/response fields.
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

LANGSMITH_ENABLED = os.environ.get("LANGSMITH_TRACING", "false").lower() in ("1", "true", "yes")
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "default")

_log_dir = os.path.join(os.path.dirname(__file__), "langsmith_logs")
os.makedirs(_log_dir, exist_ok=True)


def _http_log(prompt: str, response_text: str, model: str) -> bool:
    """Post a run to the LangSmith HTTP API. Returns True on success."""
    if not LANGSMITH_ENDPOINT or not LANGSMITH_API_KEY:
        return False
    # Compose a lightweight payload. LangSmith's API expects a particular shape;
    # this implementation uses a best-effort generic payload that includes project
    # and run data. If the API shape differs, the call will likely return an error
    # and we'll fall back to file logging.
    url = LANGSMITH_ENDPOINT.rstrip("/") + "/v1/runs"
    headers = {"Authorization": f"Bearer {LANGSMITH_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "project": LANGSMITH_PROJECT,
        "name": "agent-interaction",
        "inputs": {
            "prompt": prompt[:500],
            "response": response_text[:2000],
            "model": model,
        },
        "metadata": {"source": "local-agent"},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            return True
        return False
    except Exception:
        return False


def log_interaction(prompt: str, response_text: str, model: str) -> None:
    """Record an interaction to LangSmith via HTTP if configured, else write local JSON.

    This function never logs secrets or API keys. Local logs store only timestamp,
    model name, prompt (truncated), and response (truncated).
    """
    if not LANGSMITH_ENABLED:
        return

    try:
        ok = _http_log(prompt, response_text, model)
        if ok:
            return
    except Exception:
        pass

    # File fallback
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "project": LANGSMITH_PROJECT,
        "model": model,
        "prompt": prompt[:500],
        "response": response_text[:2000],
    }
    try:
        fname = os.path.join(_log_dir, datetime.utcnow().strftime("%Y%m%dT%H%M%S.%f") + ".json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
