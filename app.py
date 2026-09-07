"""App wrapper for Hugging Face Spaces.

Spaces expect an entrypoint (app.py) that exposes a Gradio `Blocks` or `Interface` instance.
This file imports the demo from gradio_app and exposes it as the `app` variable.

Do not call `.launch()` here; the Spaces runtime will host the app.
"""
import os

try:
    import gradio_app
except Exception as e:
    raise RuntimeError("Failed to import gradio_app. Ensure dependencies are installed and gradio_app.py is present.") from e

# gradio_app defines the Blocks as `demo` in the module scope
app = getattr(gradio_app, "demo", None)
if app is None:
    # Try alternative names
    app = getattr(gradio_app, "app", None)

if app is None:
    raise RuntimeError("Could not find a Gradio Blocks/Interface instance named 'demo' or 'app' in gradio_app.py")

# Expose `app` for HF Spaces runtime
__all__ = ["app"]
