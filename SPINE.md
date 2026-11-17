# Optional spinal cord segmentation

This compact CPU mode segments the spinal cord in one T2-weighted 3D NIfTI volume. It produces an overlay, foreground volume, and downloadable binary NIfTI mask in the input grid and affine. It does not classify spinal diseases or produce disease probabilities. Review the mask before use.

Install with:

```bash
.venv/bin/python -m pip install -e '.[spine]'
.venv/bin/python scripts/setup_spine.py
```

The pinned legacy [SCT DeepSeg SC](https://github.com/spinalcordtoolbox/deepseg_sc_models) ONNX centerline and segmentation weights occupy approximately 8 MB. The downloader verifies the official archive checksum; inference verifies individual model checksums. This older model was chosen for its small size and CPU speed. SCT now provides newer segmentation workflows; see its [documentation](https://spinalcordtoolbox.com/stable/user_section/command-line/sct_deepseg_sc.html).

The local implementation adapts SCT's centerline tracking and two-stage inference, rather than reproducing the entire SCT command. It uses nibabel LAS orientation (SCT's RPI convention), 0.5 mm in-plane resampling, centerline-guided patches, original intensity normalization, and a 0.7 segmentation threshold. Binary masks are resampled back to the original grid. The adaptation is not clinically validated. A single upstream example was used to check geometry and agreement with its reference mask; that is not a clinical accuracy estimate.

The source and LGPL-3.0 license are retained in `vendor/spinalcord_reference/`. Upstream source commit: `c343d2c0f66fe912fa1c1b3e664f0ee3c2bb5f6a`. The local adaptation is `mri_doctor/spine.py`.

Results and masks remain in memory until deletion, restart, or the existing one-hour expiration. `GET /api/jobs/{id}/mask` downloads the mask. Input must be T2 and cover the cord; the interface requires user confirmation. Optional text descriptions are currently limited to knee and brain.
