# TriEvo-MoE-LoRA

**Parameter-Efficient Evolutionary Adaptation of SAM2 Across Specialized Domains**

TriEvo-MoE-LoRA is a parameter-efficient framework for adapting SAM2 to supervised binary segmentation. It combines a frozen Hiera encoder, low-rank adaptation and a U-Net-style decoder for polyp segmentation, camouflaged object detection, salient object detection and marine animal segmentation.

- **UCGE-Gate** uses EMA-smoothed gradient sensitivity and prediction entropy for layer credit and search exploration.
- **SARM** predicts active sample ranks from normalized spatial-frequency entropy, bounded by searched layer ceilings.
- **SAOED** orders paired low-rank channels, partitions them into disjoint expert groups and regularizes cross-group correlations.

## Documentation

- [Installation and quick start](#installation)
- [Dataset preparation and splits](docs/DATASETS.md)
- [Model configuration and experimental protocols](docs/REPRODUCIBILITY.md)
- [Third-party provenance](THIRD_PARTY_NOTICES.md)
- [Contributing](CONTRIBUTING.md)

## Installation

Use an independent Conda environment. The commands below target a CUDA 12.1 GPU runtime; Linux is recommended for the full experiments. Windows can use the same Python entry points. Run commands from the repository root.

```bash
conda env create -f environment.yml
conda activate trievo
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use Python 3.11 and PyTorch 2.5.1. SAM2.1 is included at a fixed official revision and requires PyTorch >=2.3.1; no separate SAM 2 installation is needed.

Download [SAM2.1 Hiera Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) from the [official SAM 2 project](https://github.com/facebookresearch/sam2). Place it at `checkpoints/sam2.1_hiera_large.pt`. Use `configs/sam2.1/sam2.1_hiera_l.yaml` with this checkpoint; do not pair it with a SAM 2.0 config.

## Dataset layout

Follow the official training/test splits described in [DATASETS.md](docs/DATASETS.md). Images and masks must have matching filename stems.

```text
dataset/
  COD10K/
    train/Imgs/   train/GT/
    test/Imgs/    test/GT/
  CAMO/
    train/Imgs/   train/GT/
    test/Imgs/    test/GT/
```

## Training

Example: train one COD model using the two official training sources. Quote paths containing spaces. Inputs are resized to 352 x 352. Training uses batch size 12, 50 epochs, AdamW and cosine learning rate annealing by default.

```bash
python train.py --hiera_path checkpoints/sam2.1_hiera_large.pt --train_image_paths dataset/COD10K/train/Imgs dataset/CAMO/train/Imgs --train_mask_paths dataset/COD10K/train/GT dataset/CAMO/train/GT --save_path outputs/COD/seed42 --seed 42
```

Repeat independently with seeds **42, 3407 and 114147**, changing both `--seed` and `--save_path`. Set `PYTHONHASHSEED` to the same seed before starting each process when exact hash randomization is required. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by the entry points. cuDNN deterministic mode is enabled by default; BF16 and TF32 are enabled. On GPUs without BF16 support, use `--amp off`.

The checkpoint with the highest **training IoU** is saved to `outputs/COD/seed42/best.pth`. Training uses the full official training split. See [experimental protocols](docs/REPRODUCIBILITY.md) for configuration details.

Training writes per-epoch metrics, layer configuration history and compute statistics. Use `--diagnostics` to save rank, Fisher and expert diagnostics. Resume from an interrupted run's saved best epoch with `--resume --resume_path outputs/COD/seed42/best.pth`; the last epoch is not separately saved. Keep the same protocol settings.

The separate Polyp design study uses `--batch_size 16 --dice_loss_weight 1 --boundary_loss_weight 1`. Its controls are exposed by `--credit_mode uniform_credit`, `--sarm_mode random_rank`, `--sarm_mode fixed_mean_rank --fixed_ranks 6,11,10,7`, and `--grouping_mode random_grouping`. Train each control independently with otherwise matching settings.

## Prediction

`test.py` restores the checkpoint's saved topology and rank warmup state, saves 8-bit probability PNGs, and reports metrics against the specified ground truth. Provide both the test images and their ground-truth masks. TTA and postprocessing are disabled by default.

```bash
python test.py --dataset_name COD10K --checkpoint outputs/COD/seed42/best.pth --test_image_path dataset/COD10K/test/Imgs --test_gt_path dataset/COD10K/test/GT --save_path outputs/predictions/COD10K/seed42 --use_tta 0 --use_postprocess 0
```

Optional multi-scale/flip evaluation is available through `--use_tta 1 --tta_scales 0.75,1.0,1.25`. Record the TTA settings when comparing evaluation results.

## Evaluation

```bash
python eval.py --dataset_name COD10K --pred_path outputs/predictions/COD10K/seed42 --gt_path dataset/COD10K/test/GT --save_path outputs/metrics/COD10K/seed42
python diagnostics/summarize_multiseed.py --results outputs/predictions/COD10K/seed42/results.json outputs/predictions/COD10K/seed3407/results.json outputs/predictions/COD10K/seed114147/results.json --output outputs/metrics/COD10K/three_seed_summary.csv
```

Evaluation includes per-image Dice/IoU at threshold 0.5, structure measure, weighted F-measure, enhanced alignment and MAE. The evaluator also exposes threshold-sweep overlap scores. Select the metric column that matches the desired aggregation convention. Three-seed summaries use sample standard deviation (`ddof=1`).

## Repository structure

```text
TriEvoMoELoRA.py       Model, search controller, SARM and SAOED
train.py              Training and checkpoint selection
test.py               Prediction and evaluation of a saved checkpoint
eval.py               Metrics from saved probability maps
dataset.py            Dataset pairing and transforms
reviewer_utils.py     Metrics and checkpoint compatibility helpers
diagnostics/          Rank, sensitivity, expert and multiseed analysis
sam2/                 Pinned official SAM 2 / SAM2.1 Python sources and configs
tests/                Component and regression tests
docs/                 Data, reproducibility and validation documentation
licenses/             Upstream license, NOTICE and source checksums
```

## Acknowledgments

TriEvo-MoE-LoRA builds on [ACO-MoE-LoRA](https://github.com/lky0234/ACO-MoE-LoRA), [SAM 2](https://github.com/facebookresearch/sam2), [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) and [PySODMetrics](https://github.com/lartpang/PySODMetrics).

## License

See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing and redistribution requirements.
