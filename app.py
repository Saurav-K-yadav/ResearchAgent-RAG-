"""App wrapper for Hugging Face Spaces.

This wrapper prefers a lightweight demo (space_app.py) when present so the
Space can run without external API keys. If space_app isn't available, it
falls back to the full gradio_app demo.

Do not call `.launch()` here; the Spaces runtime will host the app and
expect a variable named `app` that is a Gradio Blocks or Interface.
"""
import os

# Prefer a small, self-contained demo app if present (good for Spaces without keys)
app = None
try:
    import space_app
    app = getattr(space_app, "app", None) or getattr(space_app, "demo", None)
except Exception:
    app = None

if app is None:
    try:
        import gradio_app
        app = getattr(gradio_app, "demo", None) or getattr(gradio_app, "app", None)
    except Exception as e:
        raise RuntimeError("Failed to import any Gradio app (space_app or gradio_app). Ensure one of them exposes 'app' or 'demo'.") from e

__all__ = ["app"]
