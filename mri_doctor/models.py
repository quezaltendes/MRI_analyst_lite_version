"""Adapters preserve released architectures and preprocessing; no random fallback."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import MODEL_DIR, ROOT, device_name
from .errors import InputError, ModelUnavailable
from .inputs import preview_image, read_brain

FILES = {"knee": ("knee_v76.pt", 105587131), "mci": ("mci.safetensors", 353379268),
         "idh": ("idh.safetensors", 353379268)}
KNEE_NAMES = [
    "Anterior cruciate ligament injury", "Medial collateral ligament injury",
    "Medial meniscus injury", "Lateral meniscus injury",
    "Medial osteoarthritis", "Lateral osteoarthritis", "Patellofemoral osteoarthritis",
    "Effusion", "Synovitis", "Baker cyst", "Bone contusion", "Fracture",
]


def availability():
    return {k: {"installed": (MODEL_DIR / name).exists()
                and (MODEL_DIR / name).stat().st_size == size, "file": name,
                "megabytes": round(size / 1e6)} for k, (name, size) in FILES.items()}


@lru_cache(maxsize=3)
def verify_file(key, mtime_ns):
    name, size = FILES[key]
    path = MODEL_DIR / name
    manifest = MODEL_DIR / "manifest.json"
    if not path.exists() or path.stat().st_size != size or not manifest.exists():
        raise ModelUnavailable("Model weights are not installed. Run python scripts/download_models.py.")
    expected = json.loads(manifest.read_text()).get(key, {}).get("sha256")
    if not expected:
        raise ModelUnavailable("manifest.json has no SHA-256 for this model. Download the weights again.")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024**2):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ModelUnavailable("The model weight checksum does not match. Download the weights again.")
    return path


def checkpoint_path(key):
    path = MODEL_DIR / FILES[key][0]
    if not path.exists():
        raise ModelUnavailable("Model weights are not installed. Run python scripts/download_models.py.")
    return verify_file(key, path.stat().st_mtime_ns)


class BrainClassifier(nn.Module):
    """Released MONAI ViT with a binary task head; first patch token, not token mean."""
    def __init__(self):
        super().__init__()
        from monai.networks.nets import ViT
        self.backbone = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
                            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                            save_attn=False)
        # MONAI 1.5 allocates dormant cross-attention parameters even for a plain ViT.
        # The released BrainIAC encoder predates these modules and never calls them.
        # Remove only that inactive branch so ALL active parameters still load strictly.
        for block in self.backbone.blocks:
            if getattr(block, "with_cross_attention", False):
                raise ModelUnavailable("BrainIAC requires a self-attention-only encoder.")
            if hasattr(block, "cross_attn"):
                block.cross_attn = nn.Identity()
                block.norm_cross_attn = nn.Identity()
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        # IDH: independently encode both modalities, then mean features, then classify.
        features = [self.backbone(scan)[0][:, 0] for scan in x]
        return self.classifier(torch.stack(features).mean(0)).squeeze(-1)


@lru_cache(maxsize=1)
def load_model(key):
    path = checkpoint_path(key)
    device = device_name()
    if key == "knee":
        from knee.model.knee_net import build_model
        # Released archive is a state-dict + primitive config, never uploaded by a visitor.
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = state["cfg"]
        if cfg["model"].get("joint_layers", 0) or cfg["input"]["aug"].get("channels") != "center":
            raise ModelUnavailable("Unsupported knee model configuration.")
        model = build_model(cfg["model"], pretrained=False, pretrained_path=None)
        model.load_state_dict(state["model"], strict=True)

        def bounded_features(self, x):
            # Window embeddings are independent in eval mode; keep attention over ALL windows.
            flat = x.flatten(0, 1)
            return torch.cat([self.encoder.forward_features(part.contiguous(memory_format=torch.channels_last))
                              for part in flat.split(8)], dim=0)
        model.features = MethodType(bounded_features, model)
    else:
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
        # The published safetensors mirror stores the fine-tuned encoder at root and head as head.fc.*.
        translated = {("classifier." + k.removeprefix("head.fc.") if k.startswith("head.fc.")
                       else "backbone." + k): v for k, v in state.items()}
        model, cfg = BrainClassifier(), None
        model.load_state_dict(translated, strict=True)
    return model.to(device).eval(), cfg, device


def probabilities(logits):
    if not torch.isfinite(logits).all():
        raise ModelUnavailable("The model returned a non-finite result; probabilities were not generated.")
    values = torch.sigmoid(logits.float()).detach().cpu().flatten().tolist()
    return [float(x) for x in values]


def brain_transform(path):
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, NormalizeIntensityd, ToTensord
    transform = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), ToTensord(keys=["image"]),
    ])
    return transform({"image": str(path)})["image"].as_tensor().unsqueeze(0)


def predict_brain(task, paths, evidence_sink=None):
    volumes = [read_brain(p) for p in paths]
    if task == "idh":
        a, b = [v[0] for v in volumes]
        if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-3):
            raise InputError("FLAIR and T1CE must be registered in the same space (matching shape and affine).")
    if evidence_sink is not None:
        from .reporting import append_brain_panels, collect_evidence
        collect_evidence(evidence_sink, append_brain_panels, volumes, task)
    model, _, device = load_model(task)
    tensors = [brain_transform(p).to(device) for p in paths]
    with torch.inference_mode():
        p = probabilities(model(tensors))[0]
    rows = ([{"key": "MCI", "label": "Mild cognitive impairment (MCI)", "probability": p},
             {"key": "control", "label": "Control group in the binary task", "probability": 1 - p}]
            if task == "mci" else
            [{"key": "IDH_mutant", "label": "IDH-mutant", "probability": p},
             {"key": "IDH_wild_type", "label": "IDH wild-type", "probability": 1 - p}])
    data = volumes[0][1]
    return {"model": "BrainIAC ViT-B/16 — " + task.upper(), "probabilities": rows,
            "device": device, "preview": preview_image(np.rot90(data[:, :, data.shape[2] // 2])),
            "warnings": ["Preprocessed volumes accepted; preprocessing was confirmed by the user."]
            + (["IDH estimates mutation status in glioma; it is not the probability of a tumor being present."] if task == "idh" else []),
            "calibrated": False, "probability_type": "binary", "input_volumes": len(paths),
            "source": "https://github.com/AIM-KannLab/BrainIAC"}


def eval_knee_crop(images, aug, encoder_cfg):
    """Exactly GpuAugment(train=False, mirror_all=False), with a caller-selected device."""
    n, k, _, h, w = images.shape
    x = images[:, :, 1:2].expand(-1, -1, 3, -1, -1).float().div(255)
    base = float(aug["crop_mm"]) / float(aug.get("fov_mm", 176.0))
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = base
    px = int(aug["out_px"])
    grid = F.affine_grid(theta.repeat_interleave(k, 0), (n * k, 3, px, px), align_corners=False)
    x = F.grid_sample(x.reshape(n * k, 3, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mean = torch.tensor(encoder_cfg.get("mean", (.485, .456, .406)), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(encoder_cfg.get("std", (.229, .224, .225)), device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).view(n, k, 3, px, px)


def predict_knee(series, warnings, evidence_sink=None):
    from knee.constants import LABELS
    from knee.dataloading.build_cache import CFG
    from knee.dataloading.dataset import StudyDataset
    from knee.inference.partial_reader import cpu_select, gpu_windows
    model, cfg, device = load_model("knee")
    settings = cfg["input"]
    cache_name = Path(cfg["data"]["cache"]).name
    cache_cfg = CFG[cache_name]
    selector = StudyDataset(["uploaded-study"], series, k_slices=settings["k_slices"],
                            max_series=settings["max_series"], coverage=settings["coverage"])
    selected = selector._select_series(selector.by_study["uploaded-study"], np.random.default_rng(0))
    windows, types, positions = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in selected:
            item = cpu_select(row.series_dir, cache_cfg, settings["k_slices"], settings["coverage"], pool)
            if item is None:
                raise InputError("Could not decode a selected DICOM series. Check the codecs and geometry.")
            window = gpu_windows(item, cache_cfg, device)
            windows.append(window)
            if evidence_sink is not None:
                from .reporting import append_knee_panel, collect_evidence
                collect_evidence(evidence_sink, append_knee_panel, window, row.type, item["rel"], item["io"]["depth"])
            types.append(row.type)
            positions.append(item["rel"])
    if not windows:
        raise InputError("No readable series; no result was generated.")
    images = torch.stack(windows)
    preview = preview_image(images[0, images.shape[1] // 2, 1].cpu().numpy())
    x = eval_knee_crop(images, settings["aug"], model.encoder.pretrained_cfg)
    t = torch.tensor(types, dtype=torch.long, device=device)
    pos = torch.tensor(np.asarray(positions), dtype=torch.float32, device=device)
    study_idx = torch.zeros(len(types), dtype=torch.long, device=device)
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                values = probabilities(model(x, t, pos, study_idx, 1))
        else:
            values = probabilities(model(x, t, pos, study_idx, 1))
    if len(values) != 12:
        raise ModelUnavailable("The number of model outputs does not match the 12 RSNA categories.")
    return {"model": "RSNA Knee v76 · Efficiency Track #3", "probabilities": [
                {"key": key, "label": label, "probability": p} for key, label, p in zip(LABELS, KNEE_NAMES, values)],
            "device": device, "preview": preview, "warnings": warnings,
            "calibrated": False, "probability_type": "multilabel", "input_series": len(types),
            "slices_per_series": settings["k_slices"],
            "source": "https://github.com/huanghoujing/kaggle_rsna2026_knee"}
