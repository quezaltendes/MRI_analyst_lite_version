"""Bounded input handling. Uploaded names never become filesystem paths."""
import base64
from collections import defaultdict
import io
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from .config import MAX_FILES, MAX_UNPACKED
from .errors import InputError


def unpack_zip(archive: Path, destination: Path):
    try:
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            if len(entries) > MAX_FILES or sum(x.file_size for x in entries) > MAX_UNPACKED:
                raise InputError("The archive exceeds the limit: 10,000 files / 2 GB uncompressed.")
            for item in entries:
                name = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if (name.is_absolute() or ".." in name.parts or "\\" in item.filename
                        or stat.S_ISLNK(mode) or item.flag_bits & 1):
                    raise InputError("The ZIP contains an invalid path, symbolic link, or encrypted entry.")
                if item.is_dir() or "__MACOSX" in name.parts or name.name.startswith("."):
                    continue
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                # zipfile checks CRC; stream to bound peak memory.
                with z.open(item) as src, target.open("wb") as dst:
                    while chunk := src.read(1024**2):
                        dst.write(chunk)
    except (zipfile.BadZipFile, RuntimeError, OSError) as e:
        raise InputError("Could not read the ZIP archive.") from e


def preview_image(array):
    a = np.asarray(array, dtype=np.float32)
    a = np.nan_to_num(a)
    lo, hi = np.percentile(a, [1, 99])
    a = np.uint8(np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255)
    image = Image.fromarray(a)
    image.thumbnail((384, 384))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def infer_fat_suppression(ds):
    desc = str(ds.get("SeriesDescription", "")).upper()
    suppression = str(ds.get("SpectrallySelectedSuppression", "")).upper()
    options = {str(x).upper() for x in ds.get("ScanOptions", [])}
    if suppression in {"FAT", "WATER_AND_FAT"} or "FS" in options:
        return 1
    if re.search(r"\b(STIR|SPIR|SPAIR|TIRM|FS|FATSAT|FAT[ _-]?SAT|DIXON[ _-]?W)\b", desc):
        return 1
    if suppression == "NONE" or re.search(r"\b(NFS|NOFS|NON[ _-]?FS|NO[ _-]?FS)\b", desc):
        return 0
    return None


def prepare_knee(folder: Path, destination: Path):
    groups = defaultdict(list)
    studies, sides = set(), set()
    csvs = list(folder.rglob("*series.csv"))
    metadata = None
    if len(csvs) > 1:
        raise InputError("Include only one series.csv table in the ZIP.")
    if csvs:
        try:
            metadata = pd.read_csv(csvs[0], dtype={"SeriesInstanceUID": str, "StudyInstanceUID": str})
            required = {"SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"}
            if not required.issubset(metadata.columns) or metadata.SeriesInstanceUID.duplicated().any():
                raise ValueError()
            metadata = metadata.set_index("SeriesInstanceUID")
        except Exception as e:
            raise InputError("series.csv requires unique SeriesInstanceUID values, Anatomical_Plane, and Fat_Suppression.") from e
    candidates = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() != ".csv"]
    for path in candidates:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as e:
            raise InputError("The ZIP contains a file that is neither DICOM nor series.csv.") from e
        if ds.get("Modality") != "MR":
            raise InputError("Knee mode accepts MRI only (DICOM Modality=MR).")
        required = ["StudyInstanceUID", "SeriesInstanceUID", "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"]
        if any(k not in ds for k in required):
            raise InputError("The DICOM is missing a study identifier or spatial geometry.")
        if int(ds.get("NumberOfFrames", 1)) != 1:
            raise InputError("Enhanced/multiframe DICOM is not supported yet; export individual slices.")
        if int(ds.get("Rows", 0)) > 2048 or int(ds.get("Columns", 0)) > 2048:
            raise InputError("The DICOM resolution is too large.")
        body = str(ds.get("BodyPartExamined", "")).upper()
        if body and body not in {"KNEE", "LOWEREXTREMITY", "EXTREMITY", "LEG"}:
            raise InputError("BodyPartExamined indicates a different anatomical region.")
        studies.add(str(ds.StudyInstanceUID))
        side = str(ds.get("ImageLaterality", ds.get("Laterality", ""))).upper()
        if side in {"L", "R"}:
            sides.add(side)
        # Use controlled numeric names; UIDs from uploads never become paths.
        groups[str(ds.SeriesInstanceUID)].append((path, ds))
    if len(studies) != 1 or len(sides) > 1:
        raise InputError("Upload one study of one knee; mixed studies or sides are not allowed.")
    rows, mapping, skipped = [], {}, 0
    for index, (uid, slices) in enumerate(sorted(groups.items())):
        ds = slices[0][1]
        if len(slices) < 3 or re.search(r"LOCALIZER|SCOUT|SURVEY", str(ds.get("SeriesDescription", "")).upper()):
            skipped += 1
            continue
        iop = np.asarray(ds.ImageOrientationPatient, float)
        if iop.shape != (6,) or not np.isfinite(iop).all():
            raise InputError("Invalid DICOM geometry.")
        normal = np.cross(iop[:3], iop[3:])
        plane = ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(normal)))]
        fs = infer_fat_suppression(ds)
        if metadata is not None and uid in metadata.index:
            record = metadata.loc[uid]
            supplied_plane = str(record.Anatomical_Plane)
            if supplied_plane not in {"Sagittal", "Coronal", "Axial"} or supplied_plane != plane:
                raise InputError("Anatomical_Plane in series.csv does not match the DICOM geometry.")
            try:
                fs_value = float(record.Fat_Suppression)
                if fs_value not in {0.0, 1.0}:
                    raise ValueError()
                fs = int(fs_value)
            except (TypeError, ValueError) as e:
                raise InputError("Fat_Suppression must be 0 or 1.") from e
        if fs is None:
            raise InputError("Could not determine fat suppression. Add series.csv with Fat_Suppression=0/1 for each series.")
        target = destination / str(index)
        target.mkdir(parents=True)
        for n, (source, _) in enumerate(slices):
            # Moving within the temporary folder does not duplicate patient data.
            source.rename(target / f"{n:05d}.dcm")
        rows.append({"StudyInstanceUID": "uploaded-study", "SeriesInstanceUID": str(index),
                     "type": ["Sagittal", "Coronal", "Axial"].index(plane) * 2 + fs,
                     "series_dir": str(target)})
        mapping[str(index)] = len(slices)
    if not rows:
        raise InputError("No diagnostic series with at least three slices were found.")
    warnings = [f"Skipped localizer/short series: {skipped}."] if skipped else []
    if len({r['type'] // 2 for r in rows}) < 3:
        warnings.append("Some imaging planes are missing; reliability may be lower than on competition data.")
    return pd.DataFrame(rows), warnings


def read_brain(path: Path):
    import nibabel as nib
    try:
        image = nib.load(path)
        if len(image.shape) != 3 or min(image.shape) < 16 or np.prod(image.shape) > 512**3:
            raise InputError("Provide one 3D NIfTI volume (each dimension ≥16, total ≤512³ voxels).")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            raise InputError("Invalid NIfTI affine matrix.")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or np.count_nonzero(data) < 1000:
            raise InputError("The NIfTI is empty or contains NaN/Inf.")
        values = data[data != 0]
        if float(values.std()) < 1e-6:
            raise InputError("The NIfTI has no intensity variation.")
        return image, data
    except InputError:
        raise
    except Exception as e:
        raise InputError("Could not read the NIfTI (.nii or .nii.gz).") from e
