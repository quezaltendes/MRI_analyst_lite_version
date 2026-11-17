"""Local anatomy routing, separate from disease classification. Ambiguity abstains."""
from collections import defaultdict
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

import nibabel as nib
import numpy as np
import pydicom

from .config import ROOT, MAX_FILES, MAX_UNPACKED
from .errors import InputError
from .reporting import _tile_image, contact_sheet, description_status, model_directory

REGIONS = {"brain", "knee", "spine", "unknown"}
TERMS = {
    "brain": r"\b(BRAIN|HEAD|CEREBRAL|CEREBRUM|CRANIAL|CRANIUM)\b",
    "knee": r"\b(KNEE|KNEES|GENOU)\b",
    "spine": r"\b(SPINE|SPINAL|CSPINE|TSPINE|LSPINE|CERVICAL|THORACIC|LUMBAR)\b",
}


def _matches(text):
    normalized = re.sub(r"[_\-/]", " ", text.upper())
    return {region for region, pattern in TERMS.items() if re.search(pattern, normalized)}


def _decision(region="unknown", method="none", message=None):
    return {"region":region, "method":method, "local_only":True, "manual_review":True,
            "message":message or ("Region detected. You can change the selection." if region != "unknown"
                                   else "Region is uncertain. Please choose it manually.")}


def _zip_evidence(path):
    """Read headers in place; decode at most nine representative single-frame images."""
    frames, groups, studies, regions = [], defaultdict(list), set(), set()
    try:
        with zipfile.ZipFile(path) as package:
            entries = package.infolist()
            if len(entries) > MAX_FILES or sum(e.file_size for e in entries) > MAX_UNPACKED:
                raise InputError("The archive exceeds the study size limits.")
            for entry in entries:
                name = PurePosixPath(entry.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                    raise InputError("Unsafe or encrypted ZIP entry.")
                if entry.is_dir() or "__MACOSX" in name.parts or name.name.startswith(".") or name.suffix == ".csv": continue
                if name.name.lower().endswith((".nii", ".nii.gz")):
                    return set(), [], "Extract the ZIP first, then select the .nii or .nii.gz volume."
                try:
                    with package.open(entry) as stream:
                        header = pydicom.dcmread(stream,stop_before_pixels=True)
                except Exception as exc:
                    raise InputError("The ZIP must contain DICOM series or NIfTI files.") from exc
                if header.get("Modality") != "MR":
                    raise InputError("Automatic routing accepts MRI studies only.")
                studies.add(str(header.get("StudyInstanceUID", "")))
                regions |= _matches(str(header.get("BodyPartExamined", "")))
                regions |= _matches(str(header.get("SeriesDescription", "")))
                groups[str(header.get("SeriesInstanceUID", ""))].append(entry)
            if len(studies) != 1 or "" in studies:
                raise InputError("Upload exactly one MRI study.")
            if regions:
                return regions, [], None
            for group in sorted(groups.values(),key=len,reverse=True)[:3]:
                for index in np.unique(np.linspace(0,len(group)-1,min(3,len(group))).round().astype(int)):
                    entry = group[int(index)]
                    if entry.file_size > 64*1024**2:
                        raise InputError("A DICOM slice is too large for automatic routing.")
                    with package.open(entry) as stream: image = pydicom.dcmread(stream)
                    if int(image.get("NumberOfFrames",1)) != 1: continue
                    data = image.pixel_array.astype(np.float32)*float(image.get("RescaleSlope",1))+float(image.get("RescaleIntercept",0))
                    if image.get("PhotometricInterpretation") == "MONOCHROME1": data = -data
                    spacing = tuple(float(x) for x in image.get("PixelSpacing",[1,1]))
                    frames.append(_tile_image(data,spacing))
        return regions, frames, None
    except zipfile.BadZipFile as exc:
        raise InputError("Could not read the ZIP archive.") from exc


def _nifti_evidence(path):
    try:
        source = nib.load(path)
        if len(source.shape) != 3 or min(source.shape) < 3 or np.prod(source.shape) > 64_000_000:
            raise InputError("Automatic routing requires one bounded 3D MRI volume.")
        # Only explicit anatomical words count; file extensions and names are never anatomy evidence.
        regions = _matches(bytes(source.header['descrip']).decode('utf-8',errors='ignore'))
        if regions: return regions, []
        image = nib.as_closest_canonical(source)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or np.count_nonzero(data) < 1000:
            raise InputError("The volume is empty or contains invalid values.")
        zooms = image.header.get_zooms()
        frames = []
        for axis in (2,1,0):
            others = [i for i in range(3) if i != axis]
            for position in (.35,.5,.65):
                frames.append(_tile_image(np.rot90(np.take(data,int((data.shape[axis]-1)*position),axis=axis)),
                                          (zooms[others[1]],zooms[others[0]])))
        return set(), frames
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Could not read the NIfTI volume.") from exc


def detect_region(path, workspace):
    is_zip = str(path).lower().endswith('.zip')
    frames = []
    try:
        if is_zip:
            regions, frames, note = _zip_evidence(path)
            if note: return _decision(message=note)
        else:
            regions, frames = _nifti_evidence(path)
        if len(regions) > 1:
            return _decision(method="conflicting_metadata",message="Conflicting anatomical metadata. Choose the region manually and verify that this is one study.")
        if len(regions) == 1:
            return _decision(regions.pop(),"dicom_metadata" if is_zip else "nifti_metadata")
        if not frames or not description_status()["available"]:
            return _decision(message="No explicit anatomy tags and the local vision classifier is unavailable. Choose the region manually.")
        directory = Path(workspace)/"anatomy"; directory.mkdir()
        sheet = contact_sheet(frames,"MRI views: identify anatomical region only")
        image = directory/"views.png"; sheet.save(image); sheet.close()
        prompt = (
            "Identify the anatomical region in these MRI views. The sheet shows sampled slices of ONE study. "
            "Choose brain for intracranial brain/head MRI, knee for the knee joint, spine for spinal column/spinal cord MRI. "
            "Choose unknown for another organ, unclear/empty images, mixed regions, or insufficient evidence. "
            "Do not diagnose disease, infer sequence type, or use file format as evidence. "
            'Answer with exactly one word: brain, knee, spine, or unknown.'
        )
        request, output = directory/"input.json", directory/"output.json"
        request.write_text(json.dumps({"images":[str(image)],"prompt":prompt,"max_tokens":16}))
        env = {**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","DO_NOT_TRACK":"1",
               "TOKENIZERS_PARALLELISM":"false","MPLBACKEND":"Agg","MPLCONFIGDIR":str(ROOT/"models/.runtime-cache/matplotlib")}
        completed = subprocess.run([sys.executable,"-m","mri_doctor.description_worker","--model-path",str(model_directory()),
                                    "--input",str(request),"--output",str(output)],cwd=ROOT,env=env,
                                   stdin=subprocess.DEVNULL,capture_output=True,timeout=240,check=False)
        if completed.returncode or not output.is_file(): return _decision(method="vision_failed")
        text = json.loads(output.read_text())["text"].strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$","",text).strip()
        if text.lower() in REGIONS:
            return _decision(text.lower(),"local_vision")
        decision = json.loads(text)
        if not isinstance(decision, dict) or set(decision) != {"region"} or not isinstance(decision["region"],str) or decision["region"] not in REGIONS:
            return _decision(method="vision_uncertain")
        return _decision(decision["region"],"local_vision")
    except InputError:
        raise
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError):
        return _decision(method="vision_failed")
    finally:
        for frame in frames: frame.close()
