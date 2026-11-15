import numpy as np
import pytest
import torch

from mri_doctor.errors import ModelUnavailable
from mri_doctor.models import eval_knee_crop, probabilities


def test_independent_sigmoids_not_softmax():
    p = probabilities(torch.tensor([[2.0, 2.0, -2.0]]))
    assert p[0] > 0.8 and p[1] > 0.8
    assert sum(p) > 1
    with pytest.raises(ModelUnavailable):
        probabilities(torch.tensor([float("nan")]))


def test_portable_knee_crop_matches_upstream_eval(monkeypatch):
    from knee.dataloading.gpu_augment import GpuAugment
    # Test upstream CUDA entry calls on CPU without changing its math.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    torch.manual_seed(12)
    images = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    batch = {"images": images, "pos": torch.rand(2, 3), "types": torch.tensor([0, 3]),
             "study_idx": torch.zeros(2, dtype=torch.long), "flip": torch.tensor([False])}
    settings = {"out_px": 48, "crop_mm": 160., "fov_mm": 176., "channels": "center"}
    expected, _ = GpuAugment(**settings)(batch, train=False)
    actual = eval_knee_crop(images, settings, {})
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
