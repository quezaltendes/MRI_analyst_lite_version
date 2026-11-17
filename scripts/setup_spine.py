#!/usr/bin/env python3
"""Download only the official compact T2 spinal-cord model files."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/spinalcordtoolbox/deepseg_sc_models/releases/download/r20180610/20220325_deepseg_sc_models_onnx.zip"
SHA256 = "55eac72344702e4ebc72d31fc8468bcc8d1c2dcbeeef126342f1ab8f2dcffd35"


def main():
    target = ROOT/"models"/"spine"; target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary)/"models.zip"
        with urllib.request.urlopen(URL,timeout=120) as response, archive.open("wb") as stream:
            while block := response.read(1024**2): stream.write(block)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise SystemExit("Spinal model archive checksum mismatch.")
        with zipfile.ZipFile(archive) as package:
            for name in ("t2_ctr.onnx","t2_sc.onnx","README.txt"):
                (target/name).write_bytes(package.read(name))
    print("T2 spinal cord models installed (about 8 MB). Restart the application.")


if __name__ == "__main__": main()
