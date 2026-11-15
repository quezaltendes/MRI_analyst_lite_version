"""Local image-grounded descriptions. Classifier scores are never captioning inputs."""
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import ROOT

REPORT_MODEL = "mlx-community/medgemma-1.5-4b-it-4bit"
REPORT_KEYS = {"technique", "observations", "impression", "limitations"}
MAX_PANELS = 6


class DescriptionError(RuntimeError):
    pass


class EvidencePanels(list):
    error = None


def collect_evidence(sink, collector, *args):
    """Optional description preparation must not invalidate classifier predictions."""
    if getattr(sink, "error", None):
        return
    try:
        collector(*args, sink)
    except Exception:
        for panel in sink:
            panel.image.close()
        sink.clear()
        if isinstance(sink, EvidencePanels):
            sink.error = "Selected slices could not be prepared for a written description. Probability estimates remain available."
        else:
            raise DescriptionError("Could not prepare description evidence.")


@dataclass
class ReportPanel:
    image: Image.Image
    label: str
    sampling: list
    source_frame_count: int


def model_directory():
    configured = os.environ.get("MRI_REPORT_MODEL_PATH")
    if not configured:
        try:
            configured = json.loads((ROOT / ".description-local.json").read_text()).get("model_path")
        except (OSError, ValueError, AttributeError):
            configured = None
    return Path(configured or ROOT / "models" / "description").expanduser().resolve()


@lru_cache(maxsize=1)
def description_runtime():
    """Probe native library support once per server process, without loading weights."""
    if importlib.util.find_spec("mlx_vlm") is None:
        return "runtime_missing"
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx; mx.eval(mx.array([1]) + 1)"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
        return "ready" if probe.returncode == 0 else "runtime_unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "runtime_unavailable"


def description_status():
    """No downloads, model loading, credentials, or outbound calls during a status check."""
    directory = model_directory()
    config = directory / "config.json"
    manifest = directory / "description-manifest.json"
    installed = False
    if config.is_file() and manifest.is_file():
        try:
            metadata = json.loads(manifest.read_text())
            listed = metadata.get("files", [])
            installed = bool(listed) and any(x["path"].endswith(".safetensors") for x in listed)
            for record in listed:
                path = directory / record["path"]
                if not path.resolve().is_relative_to(directory) or not path.is_file() or path.stat().st_size != record["size"]:
                    installed = False
                    break
        except (OSError, ValueError, KeyError, TypeError):
            installed = False
    reason = description_runtime() if installed else "weights_missing"
    return {"available": reason == "ready", "reason": reason, "local_only": True,
            "input_scope": "sampled_slices", "message": "Written descriptions are ready." if reason == "ready"
            else "The local description runtime cannot start. Probability analysis is still available. See REPORTING.md."
            if reason == "runtime_unavailable"
            else "Written descriptions need a one-time local setup. Probability analysis is still available."}


def _tile_image(array, spacing=(1., 1.)):
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise DescriptionError("Invalid description image input.")
    values = data[data != 0]
    if not len(values):
        gray = np.zeros(data.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(values, (1, 99))
        gray = np.uint8(np.clip((data-lo) / max(float(hi-lo), 1e-6), 0, 1)*255)
        gray[data == 0] = 0
    image = Image.fromarray(gray).convert("RGB")
    # Account for voxel aspect ratio before fitting into the contact sheet.
    width = max(1, int(round(image.width * float(spacing[1]) / max(float(spacing[0]), 1e-6))))
    width = min(width, 4096)
    if width != image.width:
        image = image.resize((width, image.height), Image.Resampling.BILINEAR)
    return image


def contact_sheet(frames, title):
    if not 1 <= len(frames) <= 9:
        raise DescriptionError("A description panel must contain between one and nine slices.")
    sheet = Image.new("RGB", (896, 896), "black")
    draw = ImageDraw.Draw(sheet)
    # Only generated, controlled labels are drawn: no DICOM free text or identifiers.
    draw.text((12, 8), title, fill="white")
    for index, frame in enumerate(frames):
        tile = ImageOps.contain(frame, (290, 268), Image.Resampling.BILINEAR)
        x, y = (index % 3)*298 + 4, (index // 3)*286 + 30
        sheet.paste(tile, (x+(290-tile.width)//2, y+(268-tile.height)//2))
        draw.text((x+5, y+266), str(index+1), fill="white")
    return sheet


def append_knee_panel(windows, series_type, relative_positions, source_depth, sink):
    if len(sink) >= MAX_PANELS:
        return
    # The classifier's canonical, geometrically ordered windows; no probability outputs.
    depth = windows.shape[0]
    indices = np.unique(np.linspace(0, depth-1, min(depth, 9)).round().astype(int))
    plane = ("Sagittal", "Coronal", "Axial")[series_type // 2]
    fat = "fat-suppressed" if series_type % 2 else "without fat suppression"
    label = f"Series {len(sink)+1}: {plane}, {fat}"
    frames = [_tile_image(windows[int(i), 1].detach().cpu().numpy()) for i in indices]
    sink.append(ReportPanel(contact_sheet(frames, label), label,
                            [round(float(relative_positions[i]), 4) for i in indices], int(source_depth)))
    for image in frames:
        image.close()


def append_brain_panels(volumes, task, sink):
    import nibabel as nib
    sequences = ["T1 provided by user"] if task == "mci" else ["FLAIR provided by user", "T1CE provided by user"]
    for (source, _), sequence in zip(volumes, sequences):
        canonical = nib.as_closest_canonical(source)
        data = canonical.get_fdata(dtype=np.float32)
        zooms = canonical.header.get_zooms()[:3]
        for axis, plane in [(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]:
            if len(sink) >= MAX_PANELS:
                return
            occupied = np.flatnonzero(np.any(data != 0, axis=tuple(a for a in range(3) if a != axis)))
            if not len(occupied):
                raise DescriptionError("No nonzero brain slices are available for description.")
            start, end = int(occupied[0]), int(occupied[-1])
            indices = np.unique(np.linspace(start+.1*(end-start), end-.1*(end-start), 9).round().astype(int))
            remaining = [a for a in range(3) if a != axis]
            frames = [_tile_image(np.rot90(np.take(data, int(i), axis=axis)),
                                  spacing=(zooms[remaining[1]], zooms[remaining[0]])) for i in indices]
            label = f"{sequence}: {plane}"
            sink.append(ReportPanel(contact_sheet(frames, label), label, [int(i) for i in indices], data.shape[axis]))
            for image in frames:
                image.close()


def build_prompt(mode, task, panels):
    if mode not in {"knee", "brain"} or not panels or len(panels) > MAX_PANELS:
        raise DescriptionError("Description evidence is missing or invalid.")
    target = ("Describe the visible knee anatomy, ligaments, menisci, cartilage, marrow, and joint fluid only where assessable. "
              "Do not grade a tear or osteoarthritis without sufficient visible evidence." if mode == "knee" else
              "Describe visible brain anatomy, symmetry, ventricles, and signal differences only where assessable. "
              "Do not infer cognitive impairment, IDH mutation, molecular subtype, or patient symptoms from these images.")
    evidence = "\n".join(f"Image {i+1}: {p.label}; {len(p.sampling)} ordered sampled slices; source frame count {p.source_frame_count}."
                         for i, p in enumerate(panels))
    return ("You are drafting an experimental, image-grounded MRI description for radiologist review. "
            "The attached images are contact sheets of selected slices, NOT complete 3D studies. "
            "Tiles are ordered left-to-right, top-to-bottom; each tile number is local to its sheet. "
            "Use only what is visible. Do not invent history, age, sex, laterality, measurements, sequence weighting, "
            "contrast administration, or findings on unseen slices. Sequence names are user-supplied or metadata-derived, not independently verified. "
            "Do not treat inability to see a finding as proof of its absence. Express uncertainty and sampling limitations. "
            "Do not give treatment advice or a definitive diagnosis. Do not refer to classifier scores; none are provided. "
            + target + "\n" + evidence + "\n"
            "Return only an English JSON object with exactly these fields: "
            '{"technique":"short description of the supplied images",'
            '"observations":["visible observation with image/tile reference, or explicitly not assessable"],'
            '"impression":["cautious synthesis restricted to those observations"],'
            '"limitations":["limitations including sampled slices, missing sequences, and any non-assessable structures"]}. '
            "Do not include markdown, additional keys, or an assertion that the entire study is normal.")


def parse_description(text):
    if not isinstance(text, str) or len(text) > 20000:
        raise DescriptionError("The description response is invalid.")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except ValueError as e:
        raise DescriptionError("The image analysis did not return a structured description.") from e
    if not isinstance(value, dict) or set(value) != REPORT_KEYS:
        raise DescriptionError("The description schema is invalid.")
    if not isinstance(value["technique"], str) or not 1 <= len(value["technique"].strip()) <= 1500:
        raise DescriptionError("The description technique is invalid.")
    for key in ("observations", "impression", "limitations"):
        lines = value[key]
        if not isinstance(lines, list) or not 1 <= len(lines) <= 10 or any(
                not isinstance(x, str) or not 1 <= len(x.strip()) <= 1500 for x in lines):
            raise DescriptionError("The description sections are invalid.")
        value[key] = [x.strip() for x in lines]
    value["technique"] = value["technique"].strip()
    # Always disclose system-known limits, even if the generator omitted them.
    value["limitations"].append("Selected slices were reviewed, not every slice of the full study. This generated draft requires radiologist review.")
    return value


def generate_description(mode, task, panels, workspace):
    if getattr(panels, "error", None):
        return {"status": "error", "message": panels.error}
    capability = description_status()
    if not capability["available"]:
        return {"status": "unavailable", "reason": capability["reason"], "message": capability["message"]}
    try:
        prompt = build_prompt(mode, task, panels)
        evidence_dir = Path(workspace) / "description"
        evidence_dir.mkdir()
        paths = []
        for index, panel in enumerate(panels):
            path = evidence_dir / f"panel-{index}.png"
            panel.image.save(path)
            paths.append(str(path))
        input_path, output_path = evidence_dir / "input.json", evidence_dir / "output.json"
        input_path.write_text(json.dumps({"prompt": prompt, "images": paths}))
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "MPLBACKEND": "Agg",
               "MPLCONFIGDIR": str(ROOT / "models" / ".runtime-cache" / "matplotlib")}
        completed = subprocess.run([sys.executable, "-m", "mri_doctor.description_worker", "--model-path",
                                    str(model_directory()), "--input", str(input_path), "--output", str(output_path)],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                   timeout=int(os.environ.get("MRI_REPORT_TIMEOUT", "600")), check=False)
        if completed.returncode != 0 or not output_path.is_file():
            raise DescriptionError("Local image description failed. Check the local description setup.")
        response = json.loads(output_path.read_text())
        report = parse_description(response["text"])
        return {"status": "complete", "kind": "sampled_image_description", "language": "en", "local_only": True,
                "full_volume_review": False, "sections": report,
                "coverage": [{"label": p.label, "sampled_frames": len(p.sampling), "source_frame_count": p.source_frame_count}
                             for p in panels], "panel_count": len(panels)}
    except (DescriptionError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as e:
        message = str(e) if isinstance(e, DescriptionError) else "Local image description could not be completed. Probability estimates remain available."
        return {"status": "error", "message": message}
