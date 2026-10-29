"""Download the MuScriptor-large weights for local use.

Tries the official gated repo first (needs HF_TOKEN + accepted license),
then falls back to the public mirror that carries the identical weights.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path(__file__).resolve().parent.parent / "models" / "muscriptor-large"
ALLOW = ["model.safetensors", "config.json", "README.md"]

REPOS = ["MuScriptor/muscriptor-large", "sonic-coder/muscriptor-large"]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    last_err = None
    for repo in REPOS:
        try:
            print(f"[download] trying {repo}", file=sys.stderr, flush=True)
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=ALLOW,
                local_dir=str(DEST),
                token=os.environ.get("HF_TOKEN"),
                max_workers=4,
            )
            print(f"[download] OK {repo} -> {path}", file=sys.stderr, flush=True)
            return 0
        except Exception as e:  # noqa: BLE001 - report and try the next repo
            last_err = e
            print(f"[download] {repo} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"[download] all repos failed: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
