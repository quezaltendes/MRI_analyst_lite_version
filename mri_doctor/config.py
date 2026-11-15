from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MRI_MODEL_DIR", ROOT / "models"))
VENDOR = ROOT / "vendor" / "knee_v76"
sys.path.insert(0, str(VENDOR))
MAX_UPLOAD = int(os.environ.get("MRI_MAX_UPLOAD_MB", "512")) * 1024**2
MAX_UNPACKED = 2 * 1024**3
MAX_FILES = 10000
NOTICE = ("Research model estimate, not a medical diagnosis. "
          "Clinical calibration for your population has not been verified. "
          "A radiologist must review the result.")


def device_name():
    import torch
    requested = os.environ.get("MRI_DEVICE", "auto")
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("MRI_DEVICE must be auto, cpu, cuda or mps")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # CPU is the verified portable default; MPS can be explicitly selected.
    return "cpu"
