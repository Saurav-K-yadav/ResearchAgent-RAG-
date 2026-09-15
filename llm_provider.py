"""Google Gemini integration helper.

Reads GOOGLE_API_KEY (and optional GEMINI_MODEL) from .env and exposes a
chat_with_gemini(prompt, system_message=None) function that returns the
model's text response. If LangSmith tracing is available it will be used by
langsmith_integration.log_interaction to record prompts/responses.

The implementation is defensive: if the google.generativeai package is not
installed or configured, helpful errors are raised.
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load .env into the environment but do NOT print or expose secrets
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")  # prefer explicit env; if not set we'll pick an available model at runtime

# Prefer the newer google.genai SDK when available, otherwise fall back to the older google.generativeai
try:
    import google.genai as genai  # type: ignore
    _HAS_GENAI = True
    _GENAI_NEW = True
except Exception:
    try:
        import google.generativeai as genai  # type: ignore
        _HAS_GENAI = True
        _GENAI_NEW = False
    except Exception:
        genai = None
        _HAS_GENAI = False

# LangSmith logging helper is optional and imported at runtime to avoid hard
# dependency issues. The langsmith_integration module exposes a log_interaction
# function if available.
try:
    from langsmith_integration import log_interaction  # type: ignore
    _HAS_LANGSMITH = True
except Exception:
    log_interaction = None
    _HAS_LANGSMITH = False


def _configure_genai() -> None:
    if not _HAS_GENAI:
        raise RuntimeError("No Google GenAI SDK installed. Install 'google-genai' or 'google-generativeai'.")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file")
    # Configure the appropriate library
    if _GENAI_NEW:
        # google.genai uses Client instances rather than a global configure
        return
    else:
        # older google.generativeai
        genai.configure(api_key=GOOGLE_API_KEY)


def _extract_text_from_response(response) -> str:
    """Robust extraction of text from diverse SDK response shapes."""
    # google.genai GenerateContentResponse-like
    try:
        # new SDK: response.result.candidates[0].content.parts -> list of dicts with 'text'
        res = response
        if hasattr(res, 'result') and getattr(res.result, 'candidates', None):
            cand = res.result.candidates[0]
            content = getattr(cand, 'content', None)
            # content may have .parts
            parts = getattr(content, 'parts', None)
            if parts:
                texts = []
                for p in parts:
                    # p may be object with text attribute or dict
                    t = getattr(p, 'text', None) or (p.get('text') if isinstance(p, dict) else None)
                    if t:
                        texts.append(t)
                if texts:
                    return ''.join(texts)
        # older SDK: response.candidates or response.output
        if hasattr(res, 'candidates') and res.candidates:
            cand = res.candidates[0]
            cont = getattr(cand, 'content', None)
            if isinstance(cont, dict) and 'parts' in cont:
                texts = [part.get('text', '') for part in cont['parts'] if isinstance(part, dict)]
                if texts:
                    return ''.join(texts)
        # direct .text attribute
        if hasattr(res, 'text') and res.text:
            return res.text
    except Exception:
        pass
    # Fallback
    return str(response)


def _extract_usage_metadata(response) -> Dict[str, Any]:
    """Extract token usage from Gemini responses, if available."""
    usage = {}
    try:
        # google.genai uses response.usage_metadata on the top-level response
        if hasattr(response, 'usage_metadata'):
            usage_obj = response.usage_metadata
            if usage_obj is not None:
                for key in ["prompt_token_count", "candidates_token_count", "total_token_count", "cache_tokens_details"]:
                    val = getattr(usage_obj, key, None)
                    if val is not None:
                        usage[key] = val
        if hasattr(response, 'result') and hasattr(response.result, 'usage_metadata'):
            usage_obj = response.result.usage_metadata
            if usage_obj is not None:
                for key in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
                    val = getattr(usage_obj, key, None)
                    if val is not None:
                        usage[key] = val
        # dict-like payloads
        if isinstance(response, dict):
            for key in ["usage_metadata", "usage"]:
                val = response.get(key)
                if isinstance(val, dict):
                    usage.update(val)
        # older google.generativeai response may have .usage_metadata or .usage
        if not usage and hasattr(response, 'usage'):
            usage = getattr(response, 'usage')
    except Exception:
        pass

    if not usage:
        return {}

    normalized = {}
    for key in ["prompt_token_count", "candidates_token_count", "total_token_count", "cache_tokens_details"]:
        if key in usage:
            normalized[key] = usage[key]
    return normalized


def _choose_model_name(client, preferred_model: Optional[str] = None) -> str:
    """Resolve a working Gemini model name.

    Keep the user's explicit preference if it still exists, otherwise use a known
    stable model from the account's available model list. This avoids failures when
    the configured model name has been deprecated or renamed by Google.
    """
    preferred = preferred_model or GEMINI_MODEL
    available = []
    try:
        iterator = client.models.list()
        for item in iterator:
            name = getattr(item, "name", None) or str(item)
            if name:
                available.append(name.replace("models/", ""))
    except Exception:
        available = []

    preference_order = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-flash-latest",
    ]

    if preferred:
        if preferred.startswith("models/"):
            preferred = preferred.removeprefix("models/")
        if preferred in available:
            return preferred

    for candidate in preference_order:
        if candidate in available:
            return candidate

    if preferred and preferred not in available:
        return preferred

    if available:
        return available[0]
    return "gemini-2.5-flash"


def _model_candidates(client, preferred_model: Optional[str] = None) -> list[str]:
    """Return ordered model names, trying the preferred one first and falling back if needed."""
    preferred = preferred_model or GEMINI_MODEL
    if preferred and preferred.startswith("models/"):
        preferred = preferred.removeprefix("models/")

    available = []
    try:
        iterator = client.models.list()
        for item in iterator:
            name = getattr(item, "name", None) or str(item)
            if name:
                available.append(name.replace("models/", ""))
    except Exception:
        available = []

    ordered = []
    seen = set()

    def add(name):
        if not name:
            return
        name = name.replace("models/", "")
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    if preferred:
        add(preferred)
    for model_name in [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-flash-latest",
    ]:
        add(model_name)
    for model_name in available:
        add(model_name)
    if not ordered:
        ordered = ["gemini-2.5-flash"]
    return ordered


def _is_token_limit_error(exc: Exception) -> bool:
    """Detect retryable Gemini model errors such as token-limit or model-availability failures."""
    msg = str(exc).lower()
    tokens = [
        "token limit",
        "maximum context length",
        "prompt too long",
        "too many tokens",
        "maximum number of tokens",
        "resource exhausted",
        "429",
        "quota exceeded",
        "context length",
        "not found",
        "not available",
        "no longer available",
        "model .* is no longer available",
    ]
    return any(token in msg for token in tokens)


def chat_with_gemini(prompt: str, system_message: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Send a chat-style prompt to Gemini and return the response dict.

    The returned dict has keys: text (str), raw (original response object).

    Raises RuntimeError when configuration or the SDK is missing.
    """
    selected_model = model or GEMINI_MODEL

    _configure_genai()

    # Build messages (system + user) as a single prompt for SDKs that accept text
    full_prompt = (system_message + "\n\n" if system_message else "") + prompt

    last_error = None
    if _GENAI_NEW:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        model_names = _model_candidates(client, selected_model)
        for model_name in model_names:
            try:
                chat = client.chats.create(model=model_name)
                response = chat.send_message(full_prompt)
                text = _extract_text_from_response(response)
                raw = response
                usage = _extract_usage_metadata(raw)
                try:
                    if _HAS_LANGSMITH and log_interaction:
                        try:
                            log_interaction(prompt=full_prompt, response_text=text, model=model_name, usage=usage)
                        except Exception:
                            pass
                except Exception:
                    pass
                return {"text": text, "raw": raw, "usage": usage}
            except Exception as exc:
                last_error = exc
                if not _is_token_limit_error(exc):
                    break
        raise RuntimeError(f"Gemini (google.genai) request failed after trying models {model_names}: {last_error}")

    # Older SDK fallback path remains supported but without advanced retry logic.
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model_names = [selected_model] if selected_model else ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
        for model_name in model_names:
            try:
                model_obj = genai.GenerativeModel(model_name=model_name)
                response = model_obj.generate_content(full_prompt)
                text = _extract_text_from_response(response)
                raw = response
                usage = _extract_usage_metadata(raw)
                try:
                    if _HAS_LANGSMITH and log_interaction:
                        try:
                            log_interaction(prompt=full_prompt, response_text=text, model=model_name, usage=usage)
                        except Exception:
                            pass
                except Exception:
                    pass
                return {"text": text, "raw": raw, "usage": usage}
            except Exception as exc:
                last_error = exc
                if not _is_token_limit_error(exc):
                    break
        raise RuntimeError(f"Gemini request failed after trying models {model_names}: {last_error}")
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}")
