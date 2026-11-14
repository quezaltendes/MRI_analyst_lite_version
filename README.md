# MRI Doctor

A local research application with two modes, trained model weights, an English browser interface, and an HTTP API. Scans are not sent to external services.

Optional English written descriptions are generated locally from sampled views of the uploaded images, alongside the probabilities. See [REPORTING.md](REPORTING.md) for the model comparison, installation, privacy details, and sampling limits. This generator receives images rather than classifier scores.

| Mode | Model | Input | Output |
|---|---|---|---|
| Knee | RSNA Knee v76, 3rd place in the Efficiency Track | ZIP containing one DICOM study | 12 independent model probabilities |
| Brain / MCI | BrainIAC with MCI-specific weights | Non-contrast T1 NIfTI | MCI / control group |
| Brain / IDH | BrainIAC with IDH-specific weights | FLAIR + T1CE NIfTI | IDH-mutant / wild-type status in glioma |

IDH does not detect whether a tumor is present; the MCI control class does not rule out other conditions. Outputs are sigmoid scores from trained classifiers. Clinical calibration for your population and suitability for medical decisions have not been established. Competition AUC does not establish clinical calibration.

## Run locally

After completing the installation below, run from the project directory:

```bash
cd mri-doctor
bash scripts/run.sh
```

Open [MRI Doctor](http://127.0.0.1:8000). API documentation: [Swagger](http://127.0.0.1:8000/docs).

The server listens on `127.0.0.1` only. The first analysis includes reading and verifying weights and importing libraries. Models load on demand; the most recently used model remains cached. Jobs run sequentially, with room for one waiting job.

## Install from scratch

Requires Python 3.10+. Tested on Python 3.12, macOS arm64, and PyTorch 2.7.1 / CPU. Run from the project root during development. Keep `vendor/` next to the application.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python scripts/download_models.py
bash scripts/run.sh
```

`requirements-tested.txt` records tested direct dependency versions, rather than a complete transitive lock file. For CUDA, install a compatible PyTorch/torchvision pair following the PyTorch instructions, then install the remaining dependencies. CUDA is selected automatically when available; otherwise the application uses CPU. Set `MRI_DEVICE=cpu` to force CPU. `MRI_DEVICE=mps` is experimental and has not been verified here.

Weights total approximately 812 MB. The downloader resumes partial downloads and verifies pinned SHA-256 checksums. Sources, mirror revision, and licenses are recorded in `models/manifest.json`. The full competition dataset and training are not required. Knee weights come from the author's public release. MCI and IDH weights come from the identified BrainIAC safetensors mirror rather than the official Dropbox. Equivalence of that conversion to the original Dropbox files has not been independently verified; all active parameter structures are checked strictly.

These public weight downloads do not require a Kaggle token. Keep any personal credentials outside the repository.

## Prepare input data

### Knee

Upload one ZIP containing individual DICOM slices from one study of one knee. Subdirectories are allowed; file order does not matter. Preserve `StudyInstanceUID`, `SeriesInstanceUID`, `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, pixel data, and MR tags. Enhanced/multiframe DICOM is not supported yet.

Fat suppression is identified from explicit tags and FS/STIR/SPIR/SPAIR/TIRM labels. Processing stops if this information is ambiguous. For series without an explicit indicator, include `series.csv` in the ZIP:

```csv
SeriesInstanceUID,Anatomical_Plane,Fat_Suppression
1.2.840.10008.123.1,Sagittal,1
1.2.840.10008.123.2,Coronal,0
1.2.840.10008.123.3,Axial,1
```

`Fat_Suppression`: 1 means present; 0 means absent. `Anatomical_Plane` must be `Sagittal`, `Coronal`, or `Axial` and is checked against DICOM geometry. Competition metadata tables with extra columns are accepted, but series identifiers must be unique.

Localizers and series with fewer than three slices are skipped with a warning. A warning is also shown if any of the three imaging planes is missing. If a selected diagnostic series cannot be decoded, no probabilities are returned. Install JPEG/JPEG2000 codecs if needed:

```bash
.venv/bin/python -m pip install -e '.[dicom-codecs]'
```

The 12 outputs in official order are ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, and Fracture. These findings are independent: their probabilities **do not add up to 1**.

Competition preprocessing is preserved: patient-space slice ordering, canonical orientation, physical field of view, intensity scaling, 16 positions per series, up to 8 series, and a 160 mm crop resized to 288 px. No test-time augmentation or ensemble is used. Slices are encoded in microbatches of 8, then attention covers all selected slices. The CPU image transformation was checked for exact equality with the original `GpuAugment(train=False)`. The only change to the copied upstream reader replaces POSIX shared memory with a regular buffer: this application uses threads within one process rather than separate DataLoader processes. Pixel transformations are unchanged.

### Brain

Provide **preprocessed** 3D NIfTI volumes (`.nii` / `.nii.gz`) following the BrainIAC protocol:

1. N4 bias-field correction.
2. Registration to a template matching the original model protocol.
3. Skull stripping (brain extraction / HD-BET in the upstream project).

The application **does not automatically perform these three operations**. You must confirm completion in the interface or API. It applies the original MONAI validation transform: trilinear resizing to 96×96×96 and normalization of nonzero voxels. No additional reorientation or cropping is introduced. The checkbox records user confirmation; it does not verify preprocessing quality.

MCI requires a non-contrast T1 volume. IDH requires FLAIR and contrast-enhanced T1 from the same person, co-registered in the same space. The application checks matching shapes and affine matrices. It cannot independently verify patient identity or sequence type from a NIfTI file alone.

The model is ViT-B/16 with 88,341,505 parameters including the task head. Following the original BrainIAC, it uses the **first patch token**, rather than the average of all tokens. For IDH, features from both volumes are averaged before the classifier head. Only the inactive cross-attention branch absent from the released weights is removed in MONAI 1.5; all active parameters load with `strict=True`.

## API

```bash
# Model availability and device
curl http://127.0.0.1:8000/api/status

# Knee
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F mode=knee -F file=@/path/to/knee.zip

# MCI
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F mode=brain -F task=mci -F preprocessed=true \
  -F file=@/path/to/preprocessed_t1.nii.gz

# IDH
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F mode=brain -F task=idh -F preprocessed=true \
  -F flair=@/path/to/preprocessed_flair.nii.gz \
  -F t1ce=@/path/to/preprocessed_t1ce.nii.gz
```

POST returns `202` and an `id`. Poll `GET /api/jobs/{id}` until `status=complete` or `error`. The `result` contains probabilities, the model, device, elapsed time, warnings, `calibrated=false`, one PNG preview, and `description`. Written descriptions are requested by default; pass `-F include_description=false` to skip them. `/api/status` includes local description readiness. A description failure leaves successful probabilities available. Delete a finished result with `DELETE /api/jobs/{id}`.

Missing weights return `503`, invalid parameters return `422`, and a full queue returns `429`. There are no demonstration, random, or constant fallback probabilities.

## Validation

```bash
.venv/bin/python -m pytest -q
# With the server running; uses actual weights and synthetic images:
.venv/bin/python scripts/smoke_test.py --output validation.json
```

Unit checks cover ZIP traversal, symlinks, expansion limits, incorrect modalities, mixed studies, required fat-suppression information, invalid NIfTI inputs, MCI/IDH parameters, missing weights, temporary-file cleanup, independent sigmoid outputs, and agreement with the original image transformation.

The smoke test checks actual weights and the complete HTTP → preparation → inference → probabilities → result deletion path for all three models. **Synthetic data verifies software operation, not medical accuracy.** This application has not been validated on a real clinical cohort. CUDA/MPS and compressed DICOM codecs have not been tested in this environment.

## Data handling and operating limits

Original uploads are stored in a temporary directory and deleted after processing succeeds or fails. Results, including one preview, remain in server memory until deletion, restart, or expiration after one hour; expired results are cleared every 30 seconds. Results already displayed in the browser remain until deletion or closing the tab. Users manage downloaded JSON files themselves.

Default limits: 512 MB combined upload size, 2 GB uncompressed ZIP size, and 10,000 archive entries. Each NIfTI must be 3D, have at least 16 voxels per dimension, and contain no more than 512³ voxels. The server does not log upload filenames or DICOM tags. This is a local environment without authentication, not a production multi-user medical service.

Settings: `MRI_PORT` (8000), `MRI_DEVICE` (auto/cpu/cuda/mps), `MRI_MAX_UPLOAD_MB` (512), `MRI_MODEL_DIR` (directory containing weights and manifest), `MRI_PYTHON` (Python executable for run.sh), `MRI_REPORT_MODEL_PATH` (local description model directory), and `MRI_REPORT_TIMEOUT` (600 seconds).

## Sources and licenses

- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview): definitions of the 12 findings.
- [Huang Houjing, Efficiency Track 3rd Place](https://github.com/huanghoujing/kaggle_rsna2026_knee): included source and v76 model. The author reports private AUC 0.950 and 285 seconds for approximately 1,300 studies on two T4 GPUs. These figures **have not been reproduced on this computer**. Code: CC BY-NC 4.0. Weights: CC BY-NC-SA 4.0. The original license, README, and commit `55f6193cfd5f056bfc8c38290f93f962bc14a939` are retained in `vendor/knee_v76/`.
- [Official BrainIAC](https://github.com/AIM-KannLab/BrainIAC), [Nature Neuroscience paper](https://www.nature.com/articles/s41593-026-02202-6), [MCI documentation](https://github.com/AIM-KannLab/BrainIAC/blob/main/docs/downstream_tasks/mild_cognitive_impairment_classification.md), and [IDH documentation](https://github.com/AIM-KannLab/BrainIAC/blob/main/docs/downstream_tasks/idh_mutation_classification.md): architecture, task definitions, and preprocessing. Reference source files and the full license are included in `vendor/brainiac_reference/`.
- [eugenehp/brainiac safetensors mirror](https://huggingface.co/eugenehp/brainiac): full fine-tuned MCI/IDH models at pinned revision `c8582059ea82d48a07796a1de29af12200594e3e`.

Both models have non-commercial usage restrictions. BrainIAC permits non-commercial academic research and education only and specifically restricts clinical workflows and decision-support use. Clinical or commercial deployment requires appropriate permission from the rights holders and separate validation.

## Repository contents

This repository contains the application, browser assets, tests, weight download scripts, and third-party source/license notices. Model weights, uploaded studies, virtual environments, credentials, machine-specific paths, and local preview screenshots are excluded. Each computer downloads its own weights using the setup scripts above.

GitHub stores the source code. This FastAPI application still runs on your computer; GitHub Pages cannot run its Python inference backend. See [GITHUB.md](GITHUB.md) for upload instructions and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for included sources and assets.
