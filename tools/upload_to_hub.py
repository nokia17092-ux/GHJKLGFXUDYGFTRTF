"""Publish this package (and optionally enhanced weights) to Hugging Face.

Two things can be uploaded:

1. the ``muscriptor-plus`` package itself, as a normal model repo with the
   code and a model card, so others can install the enhancement layer
   (``--code``, on by default); and
2. any extra files you point it at - a fine-tuned checkpoint, a velocity head,
   evaluation outputs - with ``--weights path[:path-in-repo]``.

Nothing is uploaded unless ``--repo-id`` is given and ``HF_TOKEN`` (or a
logged-in session) is available, so an accidental run does no harm.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def _parse_weights(specs: list[str]) -> list[tuple[Path, str]]:
    out = []
    for spec in specs:
        if ":" in spec and not spec[1:3] == ":\\":  # keep Windows drive letters
            local, remote = spec.split(":", 1)
        else:
            local, remote = spec, Path(spec).name
        path = Path(local)
        if not path.exists():
            raise FileNotFoundError(f"weights path not found: {path}")
        out.append((path, remote))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. yourname/muscriptor-plus")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Skip uploading the package source and model card",
    )
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        metavar="LOCAL[:REMOTE]",
        help="Extra file/dir to upload (repeatable)",
    )
    parser.add_argument(
        "--model-card",
        default="HUGGINGFACE_MODEL_CARD.md",
        help="Model card to publish as README.md",
    )
    parser.add_argument("--commit-message", default="Upload muscriptor-plus")
    args = parser.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        print(
            "HF_TOKEN is not set. Create a write token at "
            "https://huggingface.co/settings/tokens and export it first.",
            file=sys.stderr,
        )
        return 2

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"[upload] repo ready: {args.repo_id}", file=sys.stderr)

    root = Path(__file__).resolve().parent.parent

    if not args.no_code:
        card = root / args.model_card
        if card.exists():
            api.upload_file(
                path_or_fileobj=str(card),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                commit_message=args.commit_message,
            )
            print("[upload] model card -> README.md", file=sys.stderr)
        api.upload_folder(
            folder_path=str(root / "src" / "muscriptor_plus"),
            path_in_repo="muscriptor_plus",
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            ignore_patterns=["__pycache__", "*.pyc"],
        )
        print("[upload] package source -> muscriptor_plus/", file=sys.stderr)
        for name in ("pyproject.toml", "README.md"):
            f = root / name
            if f.exists():
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=name,
                    repo_id=args.repo_id,
                    commit_message=args.commit_message,
                )

    for local, remote in _parse_weights(args.weights):
        if local.is_dir():
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=remote,
                repo_id=args.repo_id,
                commit_message=args.commit_message,
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=args.repo_id,
                commit_message=args.commit_message,
            )
        print(f"[upload] {local} -> {remote}", file=sys.stderr)

    print(f"[upload] done: https://huggingface.co/{args.repo_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
