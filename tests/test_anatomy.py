import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient

import mri_doctor.anatomy as anatomy
import mri_doctor.app as api
from mri_doctor.errors import InputError
from test_inputs import make_series


def package_series(tmp_path, regions):
    source = tmp_path / 'dicom'
    study = None
    for n, region in enumerate(regions):
        folder = source / str(n)
        study, _ = make_series(folder, study=study)
        for path in folder.iterdir():
            ds = pydicom.dcmread(path)
            ds.BodyPartExamined = region
            ds.save_as(path, enforce_file_format=True)
    path = tmp_path / 'unrelated-name.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for file in source.rglob('*.dcm'):
            archive.write(file, file.relative_to(source))
    return path


def test_knee_and_brain_from_dicom_tags_without_vision(tmp_path, monkeypatch):
    monkeypatch.setattr(anatomy, 'description_status', lambda: pytest.fail('Metadata must not load vision'))
    for region in ['KNEE', 'BRAIN']:
        path = package_series(tmp_path / region, [region])
        result = anatomy.detect_region(path, tmp_path)
        assert result['region'] == region.lower()
        assert result['method'] == 'dicom_metadata'


def test_mixed_anatomy_abstains(tmp_path):
    path = package_series(tmp_path, ['KNEE', 'BRAIN'])
    result = anatomy.detect_region(path, tmp_path)
    assert result['region'] == 'unknown'
    assert result['method'] == 'conflicting_metadata'


def volume(tmp_path):
    path = tmp_path / 'brain.nii.gz'
    # A misleading filename and valid file format must never determine anatomy.
    nib.save(nib.Nifti1Image(np.arange(4096, dtype=np.float32).reshape(16,16,16), np.eye(4)), path)
    return path


def test_nifti_is_not_assumed_to_be_brain(tmp_path, monkeypatch):
    monkeypatch.setattr(anatomy, 'description_status', lambda: {'available': False})
    assert anatomy.detect_region(volume(tmp_path), tmp_path)['region'] == 'unknown'


@pytest.mark.parametrize('answer,expected', [
    ('brain\n', 'brain'), ('knee', 'knee'),
    ('{"region":"brain"}', 'brain'), ('```json\n{"region":"knee"}\n```', 'knee'),
    ('{"region":"unknown"}', 'unknown'), ('{"region":["brain"]}', 'unknown'),
    ('["brain"]', 'unknown'), ('{"region":"brain","confidence":1}', 'unknown'),
    ('This is a brain.', 'unknown'),
])
def test_local_vision_strict_output_and_offline(tmp_path, monkeypatch, answer, expected):
    monkeypatch.setattr(anatomy, 'description_status', lambda: {'available': True})
    def worker(args, **kwargs):
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        request = json.loads(Path(args[args.index('--input')+1]).read_text())
        assert request['max_tokens'] == 16
        assert len(request['images']) == 1
        Path(args[args.index('--output')+1]).write_text(json.dumps({'text':answer}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(anatomy.subprocess, 'run', worker)
    assert anatomy.detect_region(volume(tmp_path), tmp_path)['region'] == expected


def test_detect_rejects_unsafe_archive(tmp_path):
    path = tmp_path / 'study.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('../escape.dcm', 'invalid')
    with pytest.raises(InputError, match='Unsafe'):
        anatomy.detect_region(path, tmp_path)


def test_detect_endpoint_cleans_uploaded_file(monkeypatch):
    paths = []
    def detector(path, workspace):
        paths.append(Path(workspace))
        assert path.read_bytes() == b'study'
        return {'region':'knee','method':'dicom_metadata'}
    monkeypatch.setattr(api, 'detect_region', detector)
    client = TestClient(api.app)
    response = client.post('/api/detect', files={'file':('study.zip', b'study')})
    assert response.status_code == 200
    assert response.json()['region'] == 'knee'
    assert all(not path.exists() for path in paths)
    assert client.post('/api/detect', files={'file':('empty.zip', b'')}).status_code == 422
    # The gate is released after both successful and rejected requests.
    assert client.post('/api/detect', files={'file':('study.zip', b'study')}).status_code == 200
