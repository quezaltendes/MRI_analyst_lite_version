#!/usr/bin/env python3
"""Exercise real model weights over HTTP using SYNTHETIC inputs, never clinical validation."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import httpx
import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def synthetic_knee(folder):
    folder.mkdir()
    study = generate_uid()
    for plane, iop in [("SAG", [0, 1, 0, 0, 0, 1]), ("COR", [1, 0, 0, 0, 0, 1]), ("AX", [1, 0, 0, 0, 1, 0])]:
        series = generate_uid()
        normal = np.cross(iop[:3], iop[3:])
        for i in range(16):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(str(folder / f"{plane}-{i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.Modality, ds.BodyPartExamined, ds.SeriesDescription = "MR", "KNEE", plane + " PD FS"
            ds.ImageOrientationPatient = iop
            ds.ImagePositionPatient = (normal * i * 3).tolist()
            ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 3
            ds.Rows = ds.Columns = 96
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 15, 0, 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            yy, xx = np.mgrid[:96, :96]
            image = np.exp(-((xx-48)**2+(yy-48)**2) / 500) * (1000 + i*20) + xx + yy
            ds.PixelData = image.astype(np.uint16).tobytes()
            ds.save_as(ds.filename, enforce_file_format=True)
    archive = folder.parent / "synthetic-knee.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir(): z.write(path, path.name)
    return archive


def run(base_url, output, include_description=False, selected=None):
    reports = []
    with tempfile.TemporaryDirectory(prefix="mri-smoke-") as tmp, httpx.Client(base_url=base_url, timeout=60) as client:
        root = Path(tmp)
        knee = synthetic_knee(root / "dicom")
        z, y, x = np.mgrid[-1:1:64j, -1:1:64j, -1:1:64j]
        mask = x*x + y*y + z*z < 0.7
        paths = []
        for name, data in [("t1", (500 + x*100 + y*150 + z*80) * mask),
                           ("flair", (700 - x*100 + y*120 - z*60) * mask)]:
            path = root / (name + ".nii.gz")
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
            paths.append(path)
        for mode, task in [("knee", "mci"), ("brain", "mci"), ("brain", "idh")]:
            if selected and ("knee" if mode == "knee" else task) != selected:
                continue
            inputs = ({"file": knee} if mode == "knee" else {"file": paths[0]} if task == "mci"
                      else {"flair": paths[1], "t1ce": paths[0]})
            files = {key: (p.name, p.read_bytes()) for key, p in inputs.items()}
            response = client.post("/api/jobs", data={"mode": mode, "task": task, "preprocessed": "true",
                                   "include_description": str(include_description).lower()}, files=files)
            response.raise_for_status()
            job_id = response.json()["id"]
            print(mode, task, "submitted", flush=True)
            deadline = time.monotonic() + 900
            while True:
                response = client.get(f"/api/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                if job["status"] == "error": raise RuntimeError(job["error"])
                if job["status"] == "complete": break
                if time.monotonic() > deadline: raise TimeoutError("Inference smoke test timed out")
                time.sleep(1)
            result = job["result"]
            expected = 12 if mode == "knee" else 2
            assert len(result["probabilities"]) == expected
            assert all(0 <= p["probability"] <= 1 for p in result["probabilities"])
            assert result["calibrated"] is False
            if mode == "brain": assert abs(sum(p["probability"] for p in result["probabilities"]) - 1) < 1e-6
            if include_description:
                assert result["description"]["status"] == "complete", result["description"]
                assert result["description"]["local_only"] is True
                assert result["description"]["full_volume_review"] is False
            reports.append({"mode": mode, "task": result["task"], "model": result["model"],
                            "seconds": result["elapsed_seconds"], "outputs": expected, "status": "passed",
                            "description": result["description"]})
            client.delete(f"/api/jobs/{job_id}").raise_for_status()
            assert client.get(f"/api/jobs/{job_id}").status_code == 404
            print(reports[-1], flush=True)
    report = {"input_type": "synthetic, not patient MRI", "clinical_validation": False, "checks": reports}
    if output: Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output")
    parser.add_argument("--include-description", action="store_true")
    parser.add_argument("--selected", choices=["knee", "mci", "idh"])
    args = parser.parse_args()
    run(args.base_url, args.output, args.include_description, args.selected)
