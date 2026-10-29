# Manuscript alignment and reproducibility

Prepared against the supplied main manuscript and supplementary source on 2026-10-29. The release preserves the current local algorithm rather than substituting a reimplementation from an unrelated repository.

## Method mapping

| Manuscript component | Code |
| --- | --- |
| Gradient-sensitivity EMA, prediction uncertainty and layer credit | `ACOHyperparamOptimizer`, `collect_lora_fisher_scores`, `calculate_prediction_uncertainty` |
| Sample spectral entropy and learned rank | `InstanceRankModulator.compute_complexity_feature`, `InstanceRankModulator.forward` |
| Straight-through channel masks | `ACOConvLoRA.build_sample_rank_mask` |
| Paired channel importance and cyclic expert groups | `ACOConvLoRA.update_importance_order`, `get_expert_channel_masks` |
| Sparse routing and auxiliary constraints | `ACOMoeGate`, `ACOConvLoRA.forward` |
| Cross-expert direction correlation | `ACOConvLoRA._factor_decorrelation_loss` |
| Weighted BCE + weighted IoU | `train.structure_loss` |
| Saved topology and warmup restoration | `reviewer_utils.restore_checkpoint_model` |

## Release main preset

| Setting | Value |
| --- | --- |
| Backbone | Official SAM2.1 Hiera Large, frozen pretrained encoder with attention LoRA |
| Input / epochs / batch | 352 x 352 / 50 / 12 |
| Main seeds | 42, 3407, 114147 |
| Optimizer / LR / weight decay | AdamW / 0.001 / 0.0001 |
| Schedule | Cosine, minimum LR 0.000001 |
| Rank / alpha / top-k candidates | [8,16,24,32] / [8,16,24,32,48] / [1,2,3] |
| Ants / evaporation / mutation / diversification | 24 / 0.22 / 0.12 / 20 epochs |
| Experts / maximum rank / minimum sample rank | 8 per decoder layer / 32 / 4 |
| Fisher EMA / update interval / uncertainty gamma | 0.9 / 5 epochs / 0.5 |
| Orthogonality / smoothness / efficiency | 0.01 / 0.01 / 0.001 |
| Orthogonality warmup | 10 epochs |
| Initial outer adaptation weight | 0.6 |
| Extra Dice / boundary weights | 0 / 0 for main; 1 / 1 for separate Polyp design study |
| Checkpoint | `best.pth`, maximum training IoU by default |
| Inference | No TTA, no postprocessing by default |

The manuscript fixes some but not all optimizer engineering details; retained LR and weight decay values come from the local training code. The `aco_fulltrain` protocol name is retained for compatibility and means using all supplied official training images. It does not mean reproducing the ACO-MoE-LoRA baseline.

## Differences requiring disclosure

1. **Software version:** the supplement lists PyTorch 2.1.0 and SAM2.1 together. The pinned official SAM2.1 implementation requires torch >=2.3.1. This release targets torch 2.5.1 / torchvision 0.20.1 instead of claiming that incompatible combination works.
2. **Checkpoint selection:** the supplement says IoU without defining a validation split. This release uses full-training IoU; it does not claim held-out validation selection. Test masks are never consulted during training. The previous local code selected minimum training loss; that mode remains available explicitly.
3. **Search timing:** the retained trainer selects the topology at the first batch of an epoch and updates controller feedback from epoch summaries. The manuscript's step notation should not be read as a fresh independently evaluated topology at every minibatch.
4. **Encoder granularity:** the local implementation inserts LoRA into matching attention `qkv`/`proj` linears throughout Hiera and shares an encoder rank/alpha configuration. The measured Hiera-Large graph has 99 LoRA linear wrappers. The manuscript's reference to four encoder modules should not be taken as four individual linear insertions. The decoder has four distinct adaptive stages.
5. **Adaptation schedule:** the retained scheduler adjusts the outer adaptation weight using recent training performance. Orthogonality and rank modulation have separate epoch warmups. The manuscript does not fully specify the outer schedule.
6. **Parameter counts:** 8.24M and 3.63% are manuscript results. They are not enforced or asserted by the source. See `VALIDATION.md` for the parameter count measured from this release and its implications.
7. **Evaluation:** fixed-threshold Dice/IoU and threshold-sweep Dice/IoU coexist in the code. Specify the metric column used. Main three-seed aggregation and the separate Polyp design study must not be mixed. The exact historical main TTA/overlap aggregation choices are not fully established by the supplied manuscript, so no-TTA fixed-threshold evaluation is the explicit release default.
8. **Historical checkpoints:** no trained weights or official image split manifests were supplied for this release. Code changes to backbone and defaults require new training; historical tables cannot be certified from source inspection alone.

## Separate Polyp design controls

Use batch 16, learning rate 0.001, 50 epochs, BF16, extra Dice and boundary weights 1, and no TTA. Train each control independently under otherwise matching settings. Supported controls replace Fisher credit with uniform credit, permute sample ranks within a batch, use fixed ranks [6,11,10,7], or randomize expert grouping. The manuscript's orthogonality sweep uses separate runs with changed `--lambda_orth`.

The released CLI does not supply independent on/off toggles for every entry in the complete eight-row component ablation. Design-control modes are not equivalent to disabling an entire component. Reproducing that full component table needs the exact historical ablation configurations, which were not identified in the selected main source.

## Scope of the local release validation

Component checks and synthetic training/inference exercise executable behavior. They do not validate dataset access, the contents of official pretrained weights, 50-epoch convergence or the manuscript's numerical results. See [VALIDATION.md](VALIDATION.md) for actual commands and observations.
