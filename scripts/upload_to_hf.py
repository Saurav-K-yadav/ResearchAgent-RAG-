#!/usr/bin/env python3
"""
Upload this project to the Hugging Face Hub.

Usage:
  - Set HF_TOKEN in your environment (export HF_TOKEN=hf_... or add to .env)
  - Run:
      python scripts/upload_to_hf.py --repo-id your-username/ResearchAgent-RAG --public

Options:
  --repo-id      The HF repo id (namespace/repo-name). If omitted, will use "<whoami>/ResearchAgent-RAG".
  --space        Create a Space repo (repo_type="space") instead of a normal Hub repo.
  --public       Make the repo public (default false -> private). Use this flag to make it public.
  --local-path   Path to project root (default: project parent directory where this script lives).

Notes:
  - Large runtime artifacts are excluded by default (venv, pdfs, chroma_data, langsmith_logs).
  - You must provide HF_TOKEN with proper permissions to create/push repos.
  - This script uses huggingface_hub HfApi.upload_folder which will upload files recursively.

"""
import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, HfHubHTTPError

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IGNORE_PATTERNS = [
    "venv",
    "./venv",
    ".git",
    "*.pyc",
    "__pycache__",
    "pdfs",
    "./pdfs",
    "chroma_data",
    "./chroma_data",
    "langsmith_logs",
    "./langsmith_logs",
    "node_modules",
]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=None, help="HF repo id (username/repo)")
    p.add_argument("--space", action="store_true", help="Create a Space (Gradio) repo")
    p.add_argument("--public", action="store_true", help="Make the repo public (default: private)")
    p.add_argument("--local-path", default=PROJECT_DIR, help="Local project path to upload")
    return p.parse_args()


def main():
    args = _args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not found. Set HF_TOKEN and re-run.")
        print("You can run: export HF_TOKEN=hf_...\nOr add it to your .env file and `source .env` before running this script.")
        sys.exit(1)

    api = HfApi()
    try:
        whoami = api.whoami(token=hf_token)
        hf_user = whoami.get("name") or whoami.get("user", {}).get("name")
    except Exception as e:
        print(f"Failed to fetch whoami: {e}")
        hf_user = None

    default_repo = f"{hf_user}/ResearchAgent-RAG" if hf_user else "ResearchAgent-RAG"
    repo_id = args.repo_id or default_repo
    repo_type = "space" if args.space else None

    print(f"Preparing to upload from: {args.local_path}")
    print(f"Target repo_id: {repo_id} (type={repo_type or 'repo'})")
    print("This will exclude paths:")
    for p in IGNORE_PATTERNS:
        print(f"  - {p}")

    # Create repo if does not exist
    try:
        create_repo(repo_id=repo_id, token=hf_token, private=not args.public, repo_type=repo_type)
        print(f"Created repo {repo_id}")
    except HfHubHTTPError as e:
        # 409 means already exists
        if e.status_code == 409:
            print(f"Repo {repo_id} already exists -- will upload files to it")
        else:
            print(f"Error creating repo: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error creating repo: {e}")
        sys.exit(1)

    # Upload folder recursively
    try:
        print("Uploading files (this may take a while for many files)...")
        api.upload_folder(
            folder_path=args.local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo="",
            token=hf_token,
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"Upload complete. Repository available at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

    if args.space:
        print("Note: For a Space, ensure there is an app entrypoint (app.py) or proper runtime file at the repo root.")
        print("If your Gradio app is in gradio_app.py, consider adding a small app.py that imports and launches it:")
        print("  from gradio_app import build_interface\n  iface = build_interface()\n  iface.launch(server_name='0.0.0.0')\")

    print("Done.")


if __name__ == '__main__':
    main()
