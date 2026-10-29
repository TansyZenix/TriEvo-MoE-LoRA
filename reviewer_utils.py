"""Validation, metrics and trusted local checkpoint compatibility helpers."""
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BufferedCSV:
    def __init__(self, path, flush_size=512):
        self.path = Path(path)
        self.flush_size = flush_size
        self.rows = []

    def add(self, rows):
        self.rows.extend(rows)
        if len(self.rows) >= self.flush_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows.clear()


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'ACOMoELoRA':
            module = 'TriEvoMoELoRA'
            if name == 'ACOMoELoRA':
                name = 'TriEvoMoELoRA'
        return super().find_class(module, name)


class _LegacyPickle:
    __name__ = 'pickle'
    Unpickler = _LegacyUnpickler
    load = pickle.load
    loads = pickle.loads


def load_legacy_checkpoint(path, device='cpu'):
    """Load trusted project checkpoints; remap the former pickle module name."""
    return torch.load(path, map_location=device, weights_only=False, pickle_module=_LegacyPickle)


def fixed_split(train_datasets, val_datasets, directory, ratio=0.1, split_seed=2026):
    """Immutable image/mask manifests, independent of the model training seed."""
    if not 0 < ratio < 1:
        raise ValueError('val_ratio must be between zero and one')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pairs = [(str(Path(i).resolve()), str(Path(m).resolve()))
             for ds in train_datasets for i, m in zip(ds.images, ds.gts)]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate training samples across datasets')
    identity = json.dumps(pairs, ensure_ascii=False)
    meta_path = directory / 'split.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['pairs'] != identity or meta['ratio'] != ratio or meta['split_seed'] != split_seed:
            raise ValueError('Existing split differs from current data/config; choose another split_dir')
        train_idx, val_idx = meta['train_indices'], meta['val_indices']
    else:
        rng = np.random.RandomState(split_seed)
        train_idx, val_idx, offset = [], [], 0
        for ds in train_datasets:
            if len(ds) < 2:
                raise ValueError('Each training dataset needs at least two images for validation')
            shuffled = rng.permutation(len(ds))
            count = min(len(ds) - 1, max(1, int(round(len(ds) * ratio))))
            val_idx.extend((shuffled[:count] + offset).tolist())
            train_idx.extend((shuffled[count:] + offset).tolist())
            offset += len(ds)
        train_idx, val_idx = sorted(train_idx), sorted(val_idx)
        meta = dict(pairs=identity, ratio=ratio, split_seed=split_seed,
                    train_indices=train_idx, val_indices=val_idx)
        # Exclusive creation prevents silently overwriting another seed's manifest.
        with meta_path.open('x', encoding='utf-8') as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
    for name, indices in [('train', train_idx), ('val', val_idx)]:
        text = ''.join(json.dumps(pairs[i], ensure_ascii=False) + '\n' for i in indices)
        manifest = directory / (name + '.txt')
        if manifest.exists() and manifest.read_text(encoding='utf-8') != text:
            raise ValueError(f'Manifest has been modified: {manifest}')
        manifest.write_text(text, encoding='utf-8')
    return Subset(ConcatDataset(train_datasets), train_idx), Subset(ConcatDataset(val_datasets), val_idx)


class SegmentationMetrics:
    """Existing SOD protocol plus fixed-threshold, per-image Dice/IoU."""
    def __init__(self, threshold=0.5):
        import py_sod_metrics as sm
        self.threshold = threshold
        self.meters = {'S_alpha': sm.Smeasure(), 'F_beta_w': sm.WeightedFmeasure(),
                       'F_beta': sm.Fmeasure(), 'E_phi': sm.Emeasure(), 'MAE': sm.MAE()}
        self.dice, self.iou = [], []

    def step(self, pred, gt):
        pred, gt = np.asarray(pred, dtype=np.uint8), np.asarray(gt, dtype=np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f'Prediction/GT shape mismatch: {pred.shape} != {gt.shape}')
        for meter in self.meters.values():
            meter.step(pred=pred, gt=gt)
        p, g = pred > self.threshold * 255, gt > 127
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        denom = p.sum() + g.sum()
        self.dice.append(float(2 * inter / denom) if denom else 1.0)
        self.iou.append(float(inter / union) if union else 1.0)

    def merge(self, sample):
        """Merge already-computed samples, preserving the full E-measure curve."""
        if self.threshold != sample.threshold:
            raise ValueError('Cannot merge different metric thresholds')
        for key, meter in sample.meters.items():
            for field, values in vars(meter).items():
                if isinstance(values, list):
                    getattr(self.meters[key], field).extend(values)
        self.dice.extend(sample.dice)
        self.iou.extend(sample.iou)

    def results(self):
        if not self.iou:
            raise ValueError('No evaluation images')
        fm = self.meters['F_beta'].get_results()['fm']
        em = self.meters['E_phi'].get_results()['em']
        result = dict(S_alpha=float(self.meters['S_alpha'].get_results()['sm']),
                      F_beta_w=float(self.meters['F_beta_w'].get_results()['wfm']),
                      F_beta=float(fm['adp']), E_phi=float(em['curve'].mean()),
                      E_phi_max=float(em['curve'].max()),
                      MAE=float(self.meters['MAE'].get_results()['mae']),
                      mDice=float(np.mean(self.dice)), mIoU=float(np.mean(self.iou)))
        result['composite'] = (result['S_alpha'] + result['F_beta_w'] + result['E_phi'] + 1 - result['MAE']) / 4
        return result


def selection_score(metrics, name):
    # iou/loss remain CLI-compatible but refer exclusively to validation.
    key = {'iou': 'mIoU', 'loss': 'loss'}.get(name, name)
    return -metrics[key] if key in ('loss', 'MAE') else metrics[key]


def configure_model(model, config):
    fixed_ranks = [int(value) for value in config.get('fixed_ranks', '').split(',') if value]
    if fixed_ranks and (len(fixed_ranks) != len(model.aco_layers) or any(rank < 4 or rank > 32 for rank in fixed_ranks)):
        raise ValueError('fixed_ranks must contain one rank in [4, 32] per decoder layer')
    for index, layer in enumerate(model.aco_layers):
        layer.aco_gate.routing_policy = config.get('routing_policy', 'legacy_train_only')
        layer.rank_modulator.complexity_type = config.get('complexity_type', 'paper_rfft')
        layer.importance_type = config.get('importance_type', 'paper_svd')
        layer.sarm_mode = config.get('sarm_mode', 'learned_fft')
        layer.grouping_mode = config.get('grouping_mode', 'importance_grouping')
        layer.fixed_rank = fixed_ranks[index] if fixed_ranks else config.get('fixed_rank', 8)
        layer.diagnostics_enabled = config.get('diagnostics', False)
        layer.grouping_seed = config.get('grouping_seed', 2026)
        if 'lambda_orth' in config:
            layer.orthogonality_weight = config['lambda_orth']
    optimizer = model.aco_optimizer
    optimizers = getattr(optimizer, 'optimizers', [optimizer])
    for item in optimizers:
        item.credit_mode = config.get('credit_mode', 'fisher_weighted_credit')


def restore_checkpoint_model(model, checkpoint):
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    config = checkpoint.get('config')
    if config:
        configure_model(model, config)
    else:
        # Historical checkpoints used channel-averaged rFFT and A*B importance.
        configure_model(model, dict(complexity_type='legacy_rfft', importance_type='legacy_ab_norm'))
    hp = checkpoint.get('current_hyperparams') or checkpoint.get('best_hyperparams')
    if hp:
        model._apply_encoder_hyperparams(hp['encoder'])
        for layer, params in zip(model.aco_layers, hp['decoder'].get('layers', [hp['decoder']] * len(model.aco_layers))):
            layer.set_hyperparams(**params)
        model.current_encoder_params, model.current_decoder_params = hp['encoder'], hp['decoder']
    if 'epoch' in checkpoint:
        total_epochs = (config or {}).get('epoch', 50)
        for layer in model.aco_layers:
            # Restore warmup state without reordering the saved rank_order buffer.
            layer.rank_modulator_active = checkpoint['epoch'] >= total_epochs * layer.rank_modulator_warmup_ratio
    return model


def print_metrics(label, metrics):
    print(label + ' | ' + ' | '.join(f'{key}: {value:.6f}' for key, value in metrics.items()))


def experiment_protocol(config):
    ignored = {'seed', 'save_path', 'resume', 'resume_path', 'diagnostics', 'diagnostics_interval', 'num_workers'}
    return {key: value for key, value in config.items() if key not in ignored}
