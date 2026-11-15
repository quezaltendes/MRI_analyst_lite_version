#!/usr/bin/env python3
"""Download and verify the small local chat model; no user text is sent anywhere."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_doctor.chat import CHAT_REPO, CHAT_REVISION


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "chat-lite")
    args = parser.parse_args()
    directory = args.model_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(CHAT_REPO, revision=CHAT_REVISION, files_metadata=True, token=False)
    files = [f for f in info.siblings if f.rfilename != ".gitattributes" and f.size is not None]
    missing = sum(f.size for f in files if not (directory/f.rfilename).is_file() or (directory/f.rfilename).stat().st_size != f.size)
    if shutil.disk_usage(directory).free < missing + 256*1024**2:
        raise SystemExit("Free at least 650 MB for chat setup.")
    snapshot_download(CHAT_REPO, revision=CHAT_REVISION, local_dir=directory,
                      allow_patterns=[f.rfilename for f in files], max_workers=3, token=False)
    records = []
    for f in files:
        path = directory / f.rfilename
        if not path.is_file() or path.stat().st_size != f.size:
            raise SystemExit("Incomplete chat download.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4*1024**2): digest.update(block)
        sha = digest.hexdigest()
        expected = f.lfs.get("sha256") if isinstance(f.lfs, dict) else getattr(f.lfs, "sha256", None)
        if expected and sha != expected:
            raise SystemExit("Chat checksum mismatch.")
        records.append({"path": f.rfilename, "size": f.size, "sha256": sha})
    (directory/"chat-manifest.json").write_text(json.dumps({"repo_id": CHAT_REPO, "revision": CHAT_REVISION,
                                                          "license": "apache-2.0", "files": records}, indent=2)+"\n")
    print("Lite chat is installed. Restart the local server.")


if __name__ == "__main__":
    main()
