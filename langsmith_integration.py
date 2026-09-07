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


def _http_log(prompt: str, response_text: str, model: str, usage: dict | None = None) -> bool:
    """Post a run to the LangSmith HTTP API. Returns True on success.

    LangSmith requires the API key in the X-Api-Key header and a run_type field
    in the payload. Using Authorization: Bearer ... does not authenticate for the
    public API, which is why tracing previously appeared to fail even when the
    key was valid.
    """
    if not LANGSMITH_ENDPOINT or not LANGSMITH_API_KEY:
        return False

    url = LANGSMITH_ENDPOINT.rstrip("/") + "/api/v1/runs"
    headers = {"X-Api-Key": LANGSMITH_API_KEY, "Content-Type": "application/json"}

    usage_payload = {}
    if usage:
        for k in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
            if k in usage:
                usage_payload[k] = usage[k]

    payload = {
        "project": LANGSMITH_PROJECT,
        "name": "agent-interaction",
        "run_type": "chain",
        "inputs": {
            "prompt": prompt[:500],
            "model": model,
        },
        "outputs": {
            "response": response_text[:2000],
        },
        "metadata": {
            "source": "local-agent",
            "usage": usage_payload,
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201, 202):
            return True
        return False
    except Exception:
        return False


def log_interaction(prompt: str, response_text: str, model: str, usage: dict | None = None) -> None:
    """Record an interaction to LangSmith via HTTP if configured, else write local JSON.

    This function never logs secrets or API keys. Local logs store only timestamp,
    model name, prompt (truncated), response (truncated), and token usage if available.
    """
    if not LANGSMITH_ENABLED:
        return

    try:
        ok = _http_log(prompt, response_text, model, usage)
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
        "usage": usage or {},
    }
    try:
        fname = os.path.join(_log_dir, datetime.utcnow().strftime("%Y%m%dT%H%M%S.%f") + ".json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
