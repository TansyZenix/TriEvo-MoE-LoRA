# Third-party notices

| Component | Source | License / status | Included material |
| --- | --- | --- | --- |
| SAM 2 / SAM 2.1 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2/tree/05d9e57fb3945b10c861046c1e6749e2bfc258e3) | Apache-2.0 | Python modules and YAML configs in `sam2/`; original copyright headers retained |
| ACO-MoE-LoRA | [lky0234/ACO-MoE-LoRA](https://github.com/lky0234/ACO-MoE-LoRA) | No LICENSE found on 2026-10-29; redistribution authorization pending | Local adaptation ancestry of the evolutionary LoRA architecture, training, data and inference code |
| SAM2-UNet | [WZH0120/SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) | Upstream root LICENSE identifies Apache-2.0 | Architectural reference for the Hiera encoder and U-Net decoding path; no fresh code copied from this repository |
| PySODMetrics | [lartpang/PySODMetrics](https://github.com/lartpang/PySODMetrics) | Installed dependency; retain its distributed license | Segmentation metrics, imported as `py_sod_metrics` |

SAM 2 is Copyright (c) Meta Platforms, Inc. and affiliates. The exact vendored files and their SHA-256 checksums are recorded in `licenses/sam2_manifest.json`. Unneeded notebooks, demos, video data and compiled CUDA extensions are not bundled. The segmentation path here does not require SAM 2's optional connected-component CUDA extension.

Dataset access and pretrained checkpoints are governed by their respective providers. Neither datasets nor pretrained or trained weights are redistributed in this repository.
