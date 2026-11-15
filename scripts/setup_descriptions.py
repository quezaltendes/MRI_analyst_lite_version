#!/usr/bin/env python3
"""Install pinned local description weights after checking disk space. Never uploads images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "mlx-community/medgemma-1.5-4b-it-4bit"
REVISION = "762746b7a72b0b68dd8fa3f9aea5dc3980674d23"
RESERVE_BYTES = 512 * 1024**2
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "README.md", "LICENSE*"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4*1024**2): digest.update(chunk)
    return digest.hexdigest()


def main():
    # Download straight into the project; avoid a second multi-GB Xet cache.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "description")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = args.model_dir.expanduser().resolve()
    parent = target
    while not parent.exists(): parent = parent.parent
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    records = [x for x in info.siblings if x.rfilename != ".gitattributes" and x.size is not None]
    total = sum(x.size for x in records)
    existing = sum(x.size for x in records if (target/x.rfilename).is_file() and (target/x.rfilename).stat().st_size == x.size)
    needed = total-existing+RESERVE_BYTES
    free = shutil.disk_usage(parent).free
    print(json.dumps({"repo": REPO, "revision": REVISION, "download_bytes": total-existing,
                      "free_bytes": free, "required_free_bytes": needed, "enough_space": free >= needed,
                      "model_directory": str(target)}, indent=2), flush=True)
    if args.check_only:
        return
    if free < needed:
        raise SystemExit("Not enough free space. Free at least 5 GB or choose --model-dir on another disk. No weights were downloaded.")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, revision=REVISION, local_dir=target, allow_patterns=ALLOW_PATTERNS, max_workers=3, token=False)
    config = json.loads((target/"config.json").read_text())
    if config.get("model_type") != "gemma3":
        raise SystemExit("Unexpected model architecture. Setup was not activated.")
    files = []
    for record in records:
        path = target/record.rfilename
        if not path.is_file() or path.stat().st_size != record.size:
            raise SystemExit("Incomplete description model download. Setup was not activated.")
        checksum = sha256(path)
        expected = getattr(record.lfs, "sha256", None) if record.lfs else None
        if isinstance(record.lfs, dict): expected = record.lfs.get("sha256")
        if expected and checksum != expected:
            raise SystemExit("Description model checksum mismatch. Setup was not activated.")
        files.append({"path": record.rfilename, "size": record.size, "sha256": checksum})
    (target/"description-manifest.json").write_text(json.dumps({"repo_id": REPO, "revision": REVISION,
                                                              "license": "health-ai-developer-foundations", "files": files}, indent=2)+"\n")
    (ROOT/".description-local.json").write_text(json.dumps({"model_path": str(target)}, indent=2)+"\n")
    print("Local description weights are verified. Install the descriptions extra and restart the app.")


if __name__ == "__main__":
    main()
