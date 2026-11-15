# Local written MRI descriptions

The application can return English image descriptions alongside the existing probabilities for knee, brain/MCI, and brain/IDH jobs. The selected generator is **MedGemma 1.5 4B IT**, using the MLX community 4-bit conversion. This is an experimental draft from sampled image views; no clinical accuracy claim is made for either brain or knee reports.

## Model choice

| Candidate | Relevant evidence | Decision |
|---|---|---|
| MedGemma 1.5 4B | Google's multimodal medical model supports high-dimensional CT/MRI inputs. The MLX conversion has approximately 3.44 GB of files. | One local runtime for both regions, with controlled sampling and explicit uncertainty. Requires separate evaluation on the intended knee and brain population. |
| NeuroVFM-LLaVA | Brain-specific vision-language research using Qwen3-14B. | Much heavier than the chosen 4B model and does not cover knee MRI. |
| 3DReasonKnee | Knee MRI reporting dataset, fine-tuning and evaluation research. | Useful for later domain-specific training/evaluation; not treated as a ready, validated local checkpoint. |

Sources: [official MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card), [official model](https://huggingface.co/google/medgemma-1.5-4b-it), [MLX conversion](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-4bit), [NeuroVFM paper](https://doi.org/10.1038/s41591-026-04497-1), [3DReasonKnee project](https://github.com/rajpurkarlab/3DReasonKnee).

## Setup on Apple Silicon

Use the existing project environment. Core probability inference does not require these optional dependencies.

```bash
.venv/bin/python -m pip install -e '.[descriptions]'
.venv/bin/python scripts/setup_descriptions.py --check-only
.venv/bin/python scripts/setup_descriptions.py
bash scripts/run.sh
```

Allow at least 5 GB free before setup, plus working memory for inference. The pinned files total 3,439,896,078 bytes. The downloader uses the public conversion at revision `762746b7a72b0b68dd8fa3f9aea5dc3980674d23`, verifies available upstream LFS checksums, then records SHA-256 hashes for every local file. Each inference verifies those hashes. Original HADF model terms and the conversion's notices continue to apply; inspect the downloaded model README/license. The installer does not accept gated repository terms on your behalf.

Default model directory: `models/description`. To use an external disk:

```bash
.venv/bin/python scripts/setup_descriptions.py --model-dir /Volumes/YourDisk/mri-description
```

The local machine-specific path is saved in `.description-local.json`; `MRI_REPORT_MODEL_PATH` overrides it. Restart the server after installing or changing runtime dependencies. `/api/status` checks file sizes and probes native runtime support once per server process; it does not load the model or download anything.

### Restricted macOS sessions

A Metal-enabled MLX wheel may fail during import with `No Metal device available` in a sandbox or headless session. This is a runtime restriction, not missing weights. The application reports descriptions as unavailable and keeps probability inference working. An ordinary native macOS process with Metal access can use the standard wheel.

MLX also supports a CPU-only source build, using its documented CMake options. This avoids requiring a GPU; it does not change the model or its quantization. Build inside the project virtual environment, with a compatible Xcode command-line toolchain:

```bash
.venv/bin/python -m pip install cmake nanobind packaging
# Clone the official source into a scratch directory, then from that directory:
# CMAKE_ARGS='-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF' \
# CMAKE_BUILD_PARALLEL_LEVEL=4 /absolute/path/to/.venv/bin/python \
#   -m pip install --no-deps --no-build-isolation .
```

Reference: [official MLX build instructions](https://ml-explore.github.io/mlx/build/html/install.html), [maintainer explanation of CPU-only builds for restricted sessions](https://github.com/ml-explore/mlx/issues/3148). CPU generation can be substantially slower. This project's `MRI_DEVICE` setting controls the probability models, independently of MLX's device selection.

## Image preparation and scope

- **Knee:** up to six canonical series windows already prepared from the actual uploaded DICOM data. Nine ordered positions are sampled from each 16-position window and arranged in a 3×3 contact sheet. Plane and fat-suppression labels come from validated study metadata. The generator sees the central channel of each window, without probability values.
- **Brain/MCI:** the uploaded preprocessed T1 volume is visualized in canonical orientation in three planes, nine sampled views per plane.
- **Brain/IDH:** the uploaded, geometrically matched FLAIR and T1CE volumes each supply three planes: up to six panels / 54 views in total.

Visualization reorientation affects only description evidence. Existing classifier transforms and scores remain unchanged. Contact sheets are 896×896 RGB, use robust grayscale scaling, and preserve slice aspect ratio. Only controlled labels are drawn; filenames and patient DICOM text are excluded. Source pixel data can still contain burned-in annotations, so no claim of complete image anonymization is made.

This is a sampled image description, not full-volume review. Small tears, focal lesions, subtle enhancement, and other findings may be missed. Brain images do not establish MCI or molecular IDH status: the prompt explicitly prohibits inferring those labels from appearance. The separate probability outputs retain their original task scope. The generator is not provided classifier scores.

## API and user interface

The upload form includes **Include a written description**, enabled by default. `POST /api/jobs` accepts `include_description=true` (default); use `false` for probability-only processing and faster results.

`result.description` has one of these states:

- `complete`: English `sections` containing `technique`, `observations`, `impression`, and `limitations`; sampled-view `coverage`; `local_only=true`; `full_volume_review=false`.
- `unavailable`: runtime or weights missing/unusable. No fabricated description is returned.
- `error`: preparation, inference, timeout, or schema validation failed. The successful probability result remains available.
- `not_requested`: the option was disabled.

Successful text is displayed in its own result section and can be copied. JSON export includes the description. Invalid free text is rejected rather than shown as a structured report. The default description subprocess timeout is 600 seconds; override with `MRI_REPORT_TIMEOUT`.

## Privacy and failure isolation

During inference, local weights and evidence PNGs are passed to a child process. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, telemetry opt-out, and `trust_remote_code=False` prevent hub lookups and remote model code. There is no external inference API, and no scan upload to a third party. These flags are library controls, not an operating-system network firewall.

Temporary contact sheets and generated intermediary files are deleted with the job workspace, on success or failure. Descriptions remain with results in server memory under the existing one-hour expiration/deletion policy. Worker errors do not expose stderr, patient metadata, or local filenames through the browser. A failed optional description does not discard valid classifier probabilities.

## Verification

`tests/test_reporting.py` covers the structured response contract, evidence sampling, score-free prompts, offline worker invocation, failure isolation and workspace cleanup. These tests do not establish medical validity. Any real model smoke results are recorded separately with their input provenance and runtime limitations.
