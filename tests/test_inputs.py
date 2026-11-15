import io
from pathlib import Path
import stat
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import pytest

from mri_doctor.errors import InputError
from mri_doctor.inputs import prepare_knee, read_brain, unpack_zip


def make_series(folder, study=None, series=None, modality="MR", fs=True):
    study, series = study or generate_uid(), series or generate_uid()
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.SOPInstanceUID, ds.SOPClassUID = meta.MediaStorageSOPInstanceUID, MRImageStorage
        ds.Modality, ds.BodyPartExamined, ds.Laterality = modality, "KNEE", "L"
        ds.SeriesDescription = "SAG PD FS" if fs else "SAG PD"
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        ds.ImagePositionPatient = [i * 3.0, 0, 0]
        ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 3.0
        ds.Rows, ds.Columns = 64, 64
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        yy, xx = np.mgrid[:64, :64]
        ds.PixelData = np.uint16((xx + yy + i * 2) * 10).tobytes()
        ds.save_as(ds.filename, enforce_file_format=True)
    return study, series


def test_zip_traversal_and_symlinks(tmp_path):
    for n, info in enumerate(["../escape.txt", "/escape.txt", "a\\b.txt", zipfile.ZipInfo("link")]):
        if isinstance(info, zipfile.ZipInfo):
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        path = tmp_path / f"{n}.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(info, "malicious")
        with pytest.raises(InputError):
            unpack_zip(path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_zip_expansion_limit(tmp_path, monkeypatch):
    import mri_doctor.inputs as inputs
    monkeypatch.setattr(inputs, "MAX_UNPACKED", 10)
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data", "a" * 1000)
    with pytest.raises(InputError, match="limit"):
        unpack_zip(path, tmp_path / "out")


def test_knee_rejects_other_modalities_and_mixed_studies(tmp_path):
    make_series(tmp_path / "ct", modality="CT")
    with pytest.raises(InputError, match="Modality"):
        prepare_knee(tmp_path / "ct", tmp_path / "out")
    make_series(tmp_path / "mixed" / "a")
    make_series(tmp_path / "mixed" / "b")
    with pytest.raises(InputError, match="one study"):
        prepare_knee(tmp_path / "mixed", tmp_path / "out")


def test_unknown_fat_suppression_requires_explicit_metadata(tmp_path):
    source = tmp_path / "source"
    study, uid = make_series(source, fs=False)
    with pytest.raises(InputError, match="fat suppression"):
        prepare_knee(source, tmp_path / "out")
    pd.DataFrame([{"SeriesInstanceUID": uid, "Anatomical_Plane": "Sagittal", "Fat_Suppression": 0}]).to_csv(source / "series.csv", index=False)
    series, warnings = prepare_knee(source, tmp_path / "out")
    assert series.iloc[0].type == 0
    assert len(list((tmp_path / "out").rglob("*.dcm"))) == 8
    assert warnings  # incomplete planes are disclosed


def test_partial_dicom_read_does_not_need_posix_shared_memory(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from knee.dataloading.build_cache import CFG
    from knee.inference.partial_reader import cpu_select, gpu_windows
    folder = tmp_path / "series"
    make_series(folder)
    def forbidden(self):
        raise AssertionError("The local reader must not allocate POSIX shared memory")
    monkeypatch.setattr(torch.Tensor, "share_memory_", forbidden)
    with ThreadPoolExecutor(2) as pool:
        item = cpu_select(str(folder), CFG["v1"], 4, (0.05, 0.95), pool)
    assert item is not None
    windows = gpu_windows(item, CFG["v1"], "cpu")
    assert windows.shape == (4, 3, 352, 352)
    assert windows.dtype == torch.uint8


@pytest.mark.parametrize("kind", ["empty", "nan", "4d", "constant"])
def test_invalid_nifti_rejected(tmp_path, kind):
    data = np.ones((24, 24, 24), dtype=np.float32)
    if kind == "empty": data[:] = 0
    if kind == "nan": data[0, 0, 0] = np.nan
    if kind == "4d": data = data[..., None]
    path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    with pytest.raises(InputError): read_brain(path)
