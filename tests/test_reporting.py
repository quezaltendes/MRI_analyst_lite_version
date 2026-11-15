import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image
import pytest
import torch

from mri_doctor import reporting


def panel():
    return reporting.ReportPanel(Image.new("RGB", (896, 896)), "Series 1: Sagittal, fat-suppressed", [.1, .5, .9], 30)


def valid_text():
    return json.dumps({"technique": "Selected sagittal MRI slice views.",
                       "observations": ["Assessment is limited on the supplied views."],
                       "impression": ["No reliable diagnostic conclusion is possible from this test input."],
                       "limitations": ["Only selected slices are provided."]})


def test_description_parser_requires_structured_sections():
    result = reporting.parse_description("```json\n"+valid_text()+"\n```")
    assert result["observations"]
    assert "not every slice" in result["limitations"][-1]
    for text in ["Normal MRI", "{}", '[]', valid_text().replace('"observations": [', '"observations": null, "extra": [')]:
        with pytest.raises(reporting.DescriptionError): reporting.parse_description(text)


def test_description_prompt_is_image_grounded_and_not_probability_based():
    p = panel()
    brain = reporting.build_prompt("brain", "idh", [p])
    knee = reporting.build_prompt("knee", "mci", [p])
    assert "none are provided" in brain
    assert "Do not infer cognitive impairment, IDH mutation" in brain
    assert "NOT complete 3D studies" in brain
    assert "menisci" in knee
    assert "source frame count 30" in knee
    assert "PatientName" not in brain
    p.image.close()


def test_brain_evidence_covers_three_planes_per_volume():
    grid = np.indices((24, 26, 28)).sum(0).astype(np.float32)+1
    image = nib.Nifti1Image(grid, np.diag([1., 2., 3., 1.]))
    sink = reporting.EvidencePanels()
    reporting.append_brain_panels([(image, grid), (image, grid)], "idh", sink)
    assert len(sink) == 6
    assert all(p.image.size == (896, 896) and len(p.sampling) == 9 for p in sink)
    assert {p.label.split(": ")[-1] for p in sink} == {"Axial", "Coronal", "Sagittal"}
    assert {p.source_frame_count for p in sink} == {24, 26, 28}
    for p in sink: p.image.close()


def test_knee_evidence_uses_actual_canonical_window_samples():
    windows = torch.arange(16*3*16*16).reshape(16,3,16,16).to(torch.uint8)
    sink = reporting.EvidencePanels()
    reporting.append_knee_panel(windows, 1, np.linspace(.05,.95,16), 40, sink)
    assert len(sink[0].sampling) == 9
    assert sink[0].sampling[0] == .05 and sink[0].sampling[-1] == .95
    assert sink[0].source_frame_count == 40
    assert "Sagittal" in sink[0].label and "fat-suppressed" in sink[0].label
    sink[0].image.close()


def test_missing_description_weights_return_unavailable_without_fabricated_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_REPORT_MODEL_PATH", str(tmp_path))
    status = reporting.description_status()
    assert status["available"] is False
    result = reporting.generate_description("knee", "mci", [], tmp_path)
    assert result["status"] == "unavailable"
    assert "sections" not in result


def test_native_runtime_failure_is_detected_without_exposing_stderr(monkeypatch):
    reporting.description_runtime.cache_clear()
    monkeypatch.setattr(reporting.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(reporting.subprocess, "run", lambda *a, **k: type("Completed", (), {
        "returncode": 1, "stderr": b"private native library path"})())
    try:
        assert reporting.description_runtime() == "runtime_unavailable"
    finally:
        reporting.description_runtime.cache_clear()


def test_evidence_preparation_failure_is_separate_from_classifier():
    sink = reporting.EvidencePanels([panel()])
    def bad(*args): raise RuntimeError("synthetic failure")
    reporting.collect_evidence(sink, bad)
    assert sink.error and len(sink) == 0
    assert reporting.generate_description("knee", "mci", sink, ".")["status"] == "error"


def test_local_worker_contract_is_offline_and_success_is_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "description_status", lambda: {"available": True})
    seen = {}
    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        output = Path(argv[argv.index("--output")+1])
        source = Path(argv[argv.index("--input")+1])
        request = json.loads(source.read_text())
        assert len(request["images"]) == 1 and Path(request["images"][0]).is_file()
        output.write_text(json.dumps({"text": valid_text()}))
        return type("Completed", (), {"returncode":0})()
    monkeypatch.setattr(reporting.subprocess, "run", fake_run)
    p = panel()
    result = reporting.generate_description("knee", "mci", [p], tmp_path)
    assert result["status"] == "complete"
    assert result["local_only"] is True and result["full_volume_review"] is False
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and seen["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert seen["stdin"] == reporting.subprocess.DEVNULL
    p.image.close()


def test_description_failure_keeps_successful_probability_job(tmp_path, monkeypatch):
    import mri_doctor.app as api
    monkeypatch.setattr(api, "description_status", lambda: {"available":False})
    monkeypatch.setattr(api, "predict_brain", lambda *args, **kwargs: {"probabilities":[{"probability": .2}]})
    monkeypatch.setattr(api, "generate_description", lambda *args: {"status":"error", "message":"synthetic caption error"})
    workspace = tempfile.TemporaryDirectory(dir=tmp_path)
    api.jobs["description-test"] = {"status":"queued"}
    api.process("description-test", workspace, "brain", "mci", [], True)
    assert api.jobs["description-test"]["status"] == "complete"
    assert api.jobs["description-test"]["result"]["probabilities"][0]["probability"] == .2
    assert api.jobs["description-test"]["result"]["description"]["status"] == "error"
    assert not Path(workspace.name).exists()
    api.jobs.pop("description-test")
