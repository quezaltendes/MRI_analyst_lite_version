#!/usr/bin/env python3
"""Reproducible, resumable downloads; no credentials in source or output."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HF_REV = "c8582059ea82d48a07796a1de29af12200594e3e"
ASSETS = {
    "knee": {
        "file": "knee_v76.pt", "size": 105587131,
        "sha256": "1630d6ce3ecd79947b68bb9a2554be63ff778ca505a4219577c75255d77476ff",
        "url": "https://github.com/huanghoujing/kaggle_rsna2026_knee/releases/download/v76-data/"
               "v76_r50rad_288_spatial_os10_kd56_joint3d_orient_ft8_pl5m_bksyn_gold2_ft8_kd70f_ft3_nojoint.pt",
        "source": "https://github.com/huanghoujing/kaggle_rsna2026_knee", "license": "CC-BY-NC-SA-4.0",
    },
    **{task: {
        "file": task + ".safetensors", "size": 353379268,
        "sha256": {"mci": "c57e48a51a4a7d638108953d74ed8661846c508dfb8fd5b8084f8ab3a45b2fda",
                   "idh": "7a040faaba33bb7c940f4996f2a644045f0aff5ae276e04995f5cea70a9b5995"}[task],
        "url": f"https://huggingface.co/eugenehp/brainiac/resolve/{HF_REV}/{task}.safetensors",
        "source": "https://github.com/AIM-KannLab/BrainIAC",
        "mirror": f"https://huggingface.co/eugenehp/brainiac/tree/{HF_REV}",
        "license": "Mass General Brigham non-commercial academic research",
    } for task in ("mci", "idh")},
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def download(key, workers=8, destination=None):
    asset = ASSETS[key]
    directory = Path(destination or ROOT / "models")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / asset["file"]
    if target.exists() and target.stat().st_size == asset["size"]:
        if asset.get("sha256") and digest(target) != asset["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {target.name}")
        return target
    chunks = directory / (asset["file"] + ".chunks")
    chunks.mkdir(exist_ok=True)
    chunk_size = 2 * 1024**2
    ranges = [(i, start, min(start + chunk_size, asset["size"]) - 1)
              for i, start in enumerate(range(0, asset["size"], chunk_size))]

    def fetch(item):
        i, start, end = item
        dest = chunks / str(i)
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                request = urllib.request.Request(asset["url"], headers={
                    "Range": f"bytes={start}-{end}", "User-Agent": "MRI-Doctor-Research/0.1"})
                with urllib.request.urlopen(request, timeout=180) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{asset['size']}":
                        raise RuntimeError("Download server did not honor byte range")
                    data = r.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated model chunk")
                dest.write_bytes(data)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, x) for x in ranges]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 20 == 0 or n == len(ranges):
                print(f"{key}: {n}/{len(ranges)} chunks", flush=True)
    temp = target.with_suffix(target.suffix + ".download")
    with temp.open("wb") as f:
        for i, _, _ in ranges:
            f.write((chunks / str(i)).read_bytes())
    checksum = digest(temp)
    if asset.get("sha256") and checksum != asset["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {target.name}")
    temp.replace(target)
    for path in chunks.iterdir():
        path.unlink()
    chunks.rmdir()
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(ASSETS), default=list(ASSETS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest_path = ROOT / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        pending = {pool.submit(download, k, args.workers): k for k in args.models}
        for future in as_completed(pending):
            key, path = pending[future], future.result()
            manifest[key] = {**ASSETS[key], "sha256": digest(path)}
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("Models downloaded and checksummed.")


if __name__ == "__main__":
    main()
