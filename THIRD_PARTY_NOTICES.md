# Third-party sources and assets

This project includes components governed by their own licenses. No blanket
license for the whole repository overrides those terms.

| Component | Source and notice |
|---|---|
| Knee inference/reference code | [huanghoujing/kaggle_rsna2026_knee](https://github.com/huanghoujing/kaggle_rsna2026_knee). Upstream commit in `vendor/knee_v76/UPSTREAM_COMMIT`; license in `vendor/knee_v76/LICENSE`. The local reader removes the unnecessary shared-memory allocation. |
| BrainIAC reference code and task documentation | [AIM-KannLab/BrainIAC](https://github.com/AIM-KannLab/BrainIAC). License retained in `vendor/brainiac_reference/LICENSE`; restrictions described in README.md. |
| Optional MLX CPU compatibility patch | [ml-explore/mlx](https://github.com/ml-explore/mlx), v0.32.4. Patch provenance and MIT license in `vendor/mlx_cpu_compat/`. |
| Inter font | License retained at `mri_doctor/static/assets/Inter-LICENSE.txt`. |
| MRI reference images | Supplied by the project owner for the interface. Original acquisition/source rights have not been independently verified. These are interface assets, not inference results. |
| Unused brain sculpture image | Generated illustration; provenance in DESIGN.md. |

Downloaded model files are excluded from Git. Their sources, checksums and terms
are described in `models/manifest.json` and REPORTING.md. Downloading them does not
remove their original usage restrictions.
