from fastapi.testclient import TestClient
import pytest

import mri_doctor.app as api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_status_and_home(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert set(r.json()["models"]) == {"knee", "mci", "idh"}
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_brain_requires_preprocessing_confirmation(client):
    r = client.post("/api/jobs", data={"mode": "brain", "task": "mci"}, files={"file": ("image.nii.gz", b"invalid")})
    assert r.status_code == 422
    assert "N4" in r.json()["detail"]


def test_idh_requires_both_modalities(client):
    r = client.post("/api/jobs", data={"mode": "brain", "task": "idh", "preprocessed": "true"}, files={"flair": ("flair.nii.gz", b"invalid")})
    assert r.status_code == 422
    assert "both are required" in r.json()["detail"]


def test_no_model_no_prediction(client, monkeypatch):
    monkeypatch.setattr(api, "availability", lambda: {"knee": {"installed": False}})
    r = client.post("/api/jobs", data={"mode": "knee"}, files={"file": ("study.zip", b"invalid")})
    assert r.status_code == 503
    assert "weights are not installed" in r.json()["detail"]


def test_cross_origin_upload_rejected(client):
    r = client.post("/api/jobs", data={"mode": "knee"}, headers={"origin": "https://unrelated.example"})
    assert r.status_code == 403


def test_bad_enum_and_unknown_job(client):
    assert client.post("/api/jobs", data={"mode": "xray"}).status_code == 422
    assert client.get("/api/jobs/not-a-job").status_code == 404


def test_job_error_deletes_temporary_images(tmp_path, monkeypatch):
    import tempfile
    from pathlib import Path
    workspace = tempfile.TemporaryDirectory(dir=tmp_path)
    uploaded = Path(workspace.name) / "input.zip"
    uploaded.write_bytes(b"not a zip")
    api.jobs["cleanup-test"] = {"status": "queued"}
    api.process("cleanup-test", workspace, "knee", "mci", [uploaded])
    assert api.jobs["cleanup-test"]["status"] == "error"
    assert not Path(workspace.name).exists()
    api.jobs.pop("cleanup-test")


def test_spine_requires_confirmation(client):
    response = client.post('/api/jobs', data={'mode':'spine'}, files={'file':('image.nii.gz',b'invalid')})
    assert response.status_code == 422
    assert 'T2' in response.json()['detail']


def test_mask_is_downloaded_separately_and_deleted(client):
    import time
    api.jobs['mask-test'] = {'status':'complete','updated_at':time.time(),
                            'result':{'probabilities':[],'_mask_nifti_gz':b'private-binary'}}
    try:
        assert '_mask_nifti_gz' not in client.get('/api/jobs/mask-test').json()['result']
        assert client.get('/api/jobs/mask-test/mask').content == b'private-binary'
        assert client.delete('/api/jobs/mask-test').status_code == 200
        assert client.get('/api/jobs/mask-test/mask').status_code == 404
    finally:
        api.jobs.pop('mask-test',None)
