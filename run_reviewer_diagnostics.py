"""Optional diagnostics on fixed validation data; never chooses a test checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import FullDataset
from TriEvoMoELoRA import TriEvoMoELoRA
from reviewer_utils import load_legacy_checkpoint, restore_checkpoint_model, write_csv
from train import validate, metric_oriented_loss, set_seed
from diagnostics.rank_logger import RankLogger
from diagnostics.expert_overlap import expert_overlap
from diagnostics.analyze_rank_complexity import analyze as analyze_rank
from diagnostics.analyze_layer_credit import analyze as analyze_credit
from diagnostics.profile_compute import profile


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = load_legacy_checkpoint(args.checkpoint, device)
    config = checkpoint.get('config', {})
    model = TriEvoMoELoRA(num_experts=config.get('num_experts', 8)).to(device)
    restore_checkpoint_model(model, checkpoint)
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    loader = None
    if args.rank_analysis or args.layer_credit:
        if not all((args.image_path, args.mask_path, args.val_manifest)):
            raise ValueError('Rank/credit analysis requires image_path, mask_path and fixed val_manifest')
        dataset = FullDataset(args.image_path, args.mask_path, 352, 'val')
        with open(args.val_manifest, encoding='utf-8') as stream:
            manifest = {tuple(json.loads(line)) for line in stream if line.strip()}
        indices = [i for i, pair in enumerate(zip(dataset.images, dataset.gts))
                   if tuple(str(Path(p).resolve()) for p in pair) in manifest]
        if not indices:
            raise ValueError('No images match the validation manifest')
        if args.max_samples:
            indices = indices[:args.max_samples]
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.rank_analysis:
        logger = RankLogger(output / 'rank_diagnostics.csv')
        if logger.path.exists():
            raise FileExistsError('Use a new output directory to avoid mixing diagnostic runs')
        metrics = validate(model, loader, device, args.amp, rank_logger=logger)
        logger.flush()
        analyze_rank(logger.path, output)
        write_csv(output / 'validation_metrics.csv', [metrics])
        summary.append('Rank diagnostics and phase-separated correlations exported.')
    if args.expert_overlap:
        rows = expert_overlap(model, checkpoint.get('epoch', -1))
        write_csv(output / 'expert_overlap.csv', rows)
        if rows:
            write_csv(output / 'expert_overlap_summary.csv', [dict(
                variant=model.aco_layers[0].grouping_mode,
                mean_cross_expert_factor_correlation=float(np.mean([r['mean_abs_corr'] for r in rows])))])
        summary.append('A-factor correlations measured under the saved grouping.')
    if args.layer_credit:
        scores, count = {}, 0
        for batch in loader:
            model.zero_grad(set_to_none=True)
            x, target = batch['image'].to(device), batch['label'].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and args.amp == 'bf16'):
                pred, _ = model(x, update_hyperparams=False)
            # Diagnostics only; no optimizer step, no routing update.
            loss = metric_oriented_loss(pred.float(), target,
                                        config.get('dice_loss_weight', 1.0), config.get('boundary_loss_weight', 1.0))
            loss.backward()
            for key, value in model.collect_lora_fisher_scores().items():
                scores[key] = scores.get(key, 0.0) + value * len(x)
            count += len(x)
        scores = {key: value / count for key, value in scores.items()}
        model.zero_grad(set_to_none=True)
        def evaluate_fixed():
            benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
            set_seed(args.seed)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = deterministic
            return validate(model, loader, device, args.amp)
        corr = analyze_credit(model, evaluate_fixed, scores,
                              output / 'layer_credit_diagnostic.csv')
        write_csv(output / 'layer_credit_correlations.csv', [dict(loss_source='validation_main_loss', **corr)])
        summary.append('Credit uses validation main-loss gradients; removal uses the same fixed images. It is not causal attribution.')
    if args.profile:
        values = profile(model, args.batch_size, 352, args.warmup, args.repeat, args.amp)
        compute = Path(args.checkpoint).parent / 'training_compute.csv'
        if compute.exists():
            with compute.open(encoding='utf-8') as stream:
                training = list(csv.DictReader(stream))
            values['training_GPU_hours'] = float(training[-1]['GPU_hours']) if training and training[-1]['GPU_hours'] else None
        (output / 'compute_profile.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
        write_csv(output / 'compute_profile.csv', [values])
        print(values)
        summary.append('Only TriEvo exists in this checkout. External baselines must use the same device/precision/protocol. FLOPs remain unmeasured.')
    (output / 'summary.md').write_text('\n\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='validation')
    parser.add_argument('--image-path')
    parser.add_argument('--mask-path')
    parser.add_argument('--val-manifest')
    parser.add_argument('--output', default='reviewer_diagnostics')
    parser.add_argument('--rank-analysis', action='store_true')
    parser.add_argument('--expert-overlap', action='store_true')
    parser.add_argument('--layer-credit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--amp', choices=['off', 'bf16'], default='bf16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=100)
    main(parser.parse_args())
