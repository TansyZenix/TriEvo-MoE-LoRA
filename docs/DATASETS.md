# Dataset protocol

Prepare each dataset from its official provider and follow its access and redistribution terms. This repository does not include data or invent new random train/test splits.

| Domain | Dataset | Training images | Test images |
| --- | --- | ---: | ---: |
| Polyp | Kvasir-SEG | 900 | 100 |
| Polyp | CVC-ClinicDB | 550 | 62 |
| Polyp | CVC-ColonDB | 0 | 380 |
| Polyp | CVC-300 | 0 | 60 |
| Polyp | ETIS | 0 | 196 |
| COD | COD10K | 3,040 | 2,026 |
| COD | CAMO | 1,000 | 250 |
| COD | CHAMELEON | 0 | 76 |
| COD | NC4K | 0 | 4,121 |
| SOD | DUTS-TR / DUTS-TE | 10,553 | 5,019 |
| SOD | DUT-OMRON | 0 | 5,168 |
| SOD | HKU-IS | 0 | 4,447 |
| SOD | PASCAL-S | 0 | 850 |
| SOD | ECSSD | 0 | 1,000 |
| MAS | MAS3K | 1,769 | 1,141 |
| MAS | RMAS | 2,514 | 500 |

Counts above transcribe the supplied manuscript supplement. Confirm the actual image lists against the official split release before running; counts alone do not identify a split. No official split manifests were available in the local code, so the repository cannot certify the exact historical training image lists.

Train one Polyp model on Kvasir-SEG + CVC-ClinicDB, one COD model on COD10K + CAMO, one SOD model on DUTS-TR, one MAS3K model on MAS3K and one RMAS model on RMAS. Each seed trains these five domain models independently. The 16 datasets are evaluation datasets, not 16 separate training runs per seed.

## Files and pairing

Use RGB images and single-channel binary foreground masks, conventionally encoded as 0 and 255. Supported image extensions are `.jpg`, `.jpeg` and `.png` (lowercase). An image `example.jpg` pairs with mask `example.png` by its stem. Equal counts are insufficient: filenames must align. Training validates the sorted stem pairs and applies resize, horizontal/vertical flips, tensor conversion and ImageNet normalization. The retained training implementation uses bicubic mask resizing, so soft boundary values may occur; changing this transform would create a different training protocol. Evaluation restores predictions to the original ground-truth resolution.

Directory names are examples: pass the actual image and mask directories through CLI arguments. Multiple training sources must be listed in the same order for `--train_image_paths` and `--train_mask_paths`. Do not add CHAMELEON, NC4K or held-out Polyp test images to training.

Useful official entry points: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/), [COD10K](https://github.com/DengPingFan/COD10K), [DUTS](http://saliencydetection.net/duts/), [MAS3K](https://github.com/LinHong-HIT/MAS3K). Obtain remaining datasets through the original dataset publication/provider; this release has not independently verified every download URL.
