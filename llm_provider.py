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


def chat_with_gemini(prompt: str, system_message: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Send a chat-style prompt to Gemini and return the response dict.

    The returned dict has keys: text (str), raw (original response object).

    Raises RuntimeError when configuration or the SDK is missing.
    """
    if model is None:
        model = GEMINI_MODEL

    _configure_genai()

    # Build messages (system + user) as a single prompt for SDKs that accept text
    full_prompt = (system_message + "\n\n" if system_message else "") + prompt

    if _GENAI_NEW:
        # google.genai usage
        client = genai.Client(api_key=GOOGLE_API_KEY)
        try:
            # Determine model name: prefer explicit model, otherwise try listing models
            model_name = model
            try:
                if not model_name:
                    models_iter = client.models.list()
                    # models_iter may be a pager/generator; take first
                    first = None
                    for m in models_iter:
                        first = getattr(m, 'name', None) or str(m)
                        break
                    if first:
                        model_name = first
            except Exception:
                # ignore listing failures
                pass

            # Create a chat and send the prompt
            chat = client.chats.create(model=model_name)
            response = chat.send_message(full_prompt)
            text = _extract_text_from_response(response)
            raw = response
        except Exception as e:
            raise RuntimeError(f"Gemini (google.genai) request failed: {e}")
    else:
        # older google.generativeai fallback
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Try the GenerativeModel API if available
            try:
                model_obj = genai.GenerativeModel(model_name=model)
                response = model_obj.generate_content(full_prompt)
                text = _extract_text_from_response(response)
                raw = response
            except Exception:
                # chat or generate_text fallback
                try:
                    response = genai.chat.create(model=model, messages=[{"role": "user", "content": full_prompt}])
                    text = _extract_text_from_response(response)
                    raw = response
                except Exception as ex:
                    response = genai.generate_text(model=model, input=full_prompt)
                    text = _extract_text_from_response(response)
                    raw = response
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}")

    # Log to LangSmith if available (do not expose secrets in logs)
    try:
        if _HAS_LANGSMITH and log_interaction:
            try:
                log_interaction(prompt=full_prompt, response_text=text, model=model)
            except Exception:
                # Non-fatal: tracing failures should not block main flow
                pass
    except Exception:
        pass

    return {"text": text, "raw": raw}

    # Log to LangSmith if available (do not expose secrets in logs)
    try:
        if _HAS_LANGSMITH and log_interaction:
            try:
                log_interaction(prompt=prompt, response_text=text, model=model)
            except Exception:
                # Non-fatal: tracing failures should not block main flow
                pass
    except Exception:
        pass

    return {"text": text, "raw": response}
