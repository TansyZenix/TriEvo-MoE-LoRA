# Local validation record

Validation date: 2026-10-29. These checks validate this staged implementation, not the manuscript's numerical results.

## Environment

- Windows, Python 3.11.17, PyTorch 2.5.1+cu121, CUDA available.
- Main executable tests ran in the user-managed `pytorchEnv` Conda environment, independently of the Codex application runtime.
- Installed missing Hydra, iopath and PySODMetrics dependencies for verification. The complete release installation is documented in README; a clean Linux Conda install and the remote GitHub Actions job were not executed locally.

## Executed checks

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 23 tests passed, 38.874 seconds |
| `python train.py --help` | Exit 0 |
| `python test.py --help` | Exit 0 |
| `python eval.py --help` | Exit 0 |
| `python run_reviewer_diagnostics.py --help` | Exit 0 |
| `python diagnostics/summarize_multiseed.py --help` | Exit 0 |
| Real SAM2.1 Hiera-Large model construction | Passed without pretrained weights |
| Real model CPU forward with input `[1,3,352,352]` | Output `[1,1,352,352]`, all finite |
| Real model trainable parameter count | 8,241,029 |
| Real model total retained parameter count | 220,390,357 |
| Encoder LoRA linear wrappers | 99 |

The trainable count rounds to the manuscript's 8.24M. The ratio to the retained graph above is approximately 3.74%; the paper's 3.63% uses a separate full-fine-tuning comparison denominator and is not asserted for this graph.

The suite includes CPU component backward passes, CUDA BF16 tiny-model training, discrete rank masks, Fisher EMA, routing consistency, saved topology restoration, seed/RNG recovery, threaded metrics, threshold handling, CLI evaluation, synthetic prediction exports, and a training-IoU checkpoint-selection regression. The full-model forward uses random initialization; it does not certify successful loading of a downloaded pretrained checkpoint. The synthetic checkpoint-selection test replaces the backbone and epoch measurements to isolate the selection rule.

## Fixes verified during staging

- Changed main defaults to the supplied manuscript's search space, batch size and auxiliary weights.
- Selected the checkpoint by training IoU and saved it as `best.pth`.
- Used a pinned official SAM2.1 source/configuration instead of silently pairing a SAM 2.0 config with SAM2.1 weights.
- Initialized the backbone on CPU before the caller selects its execution device.
- Corrected weighted BCE to request `reduction='none'` explicitly.
- Replaced path-dependent OpenCV file I/O with encoded buffers and NumPy file I/O. Prediction export and evaluation passed under a Chinese Windows project path.

## Unexecuted experiments and release status

No datasets or trained checkpoints were copied. No full 50-epoch training, three-seed benchmark or official pretrained checkpoint download/load was performed. No numerical performance table is certified by these checks. The complete component-ablation configuration and historical main metric/TTA details remain as documented in `REPRODUCIBILITY.md`.

The GitHub workflow is included but has not run on GitHub. The project has not been uploaded or pushed. Redistribution authorization for the ACO-MoE-LoRA-derived code remains pending; see `LICENSE.md`.

## Final artifact checks and cleanup

All 37 Python files parsed successfully. All 36 vendored SAM2 source/config checksums matched the pinned upstream manifest, and all relative Markdown links resolved. The final staged tree contains 68 files, with no weights, bytecode, server credentials, deployment scripts or archives. Two generated NVRTC kernel cache files totaling 26.05 KB were moved to the Windows Recycle Bin. Exact filenames are recorded in `cleanup_report.json`.
