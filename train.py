import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import math
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from TriEvoMoELoRA import TriEvoMoELoRA, AdaptiveTrainingScheduler
from reviewer_utils import (SegmentationMetrics, write_csv,
                            configure_model, load_legacy_checkpoint, restore_checkpoint_model,
                            print_metrics, BufferedCSV, experiment_protocol)
from diagnostics.rank_logger import RankLogger
from diagnostics.expert_overlap import expert_overlap
import time
import warnings
import copy

warnings.filterwarnings('ignore')

# === Argument Parsing ===
parser = argparse.ArgumentParser("TriEvo-MoE-LoRA Training Script")
parser.add_argument('--training_protocol', choices=['aco_fulltrain'], default='aco_fulltrain')
parser.add_argument("--hiera_path", type=str, required=True,
                    help="Path to the SAM2 pretrained Hiera model")
parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                    help="SAM2 configuration relative to the sam2 package")
parser.add_argument("--train_image_paths", type=str, nargs='+', required=True,
                    help="List of paths to training images")
parser.add_argument("--train_mask_paths", type=str, nargs='+', required=True,
                    help="List of paths to training masks")
parser.add_argument('--save_path', type=str, required=True,
                    help="Directory to save checkpoints")
parser.add_argument("--epoch", type=int, default=50,
                    help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument('--encoder_lr_scale', type=float, default=1.0,
                    help='Encoder LoRA learning rate relative to decoder learning rate')
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--dice_loss_weight", default=0.0, type=float)
parser.add_argument("--boundary_loss_weight", default=0.0, type=float)
parser.add_argument("--num_experts", default=8, type=int)
parser.add_argument("--top_k", default=2, type=int)
parser.add_argument("--aco_loss_weight", default=0.6, type=float)
parser.add_argument("--use_multi_start", action="store_true",
                    help="Enable Multi-Start ACO Optimizer to avoid local optima")
parser.add_argument("--aco_diversification_interval", default=20, type=int,
                    help="Interval for ACO diversification search")
parser.add_argument("--aco_num_ants", default=24, type=int,
                    help="Number of ants in ACO")
parser.add_argument("--aco_evaporation_rate", default=0.22, type=float,
                    help="ACO pheromone evaporation rate")
parser.add_argument("--aco_mutation_prob", default=0.12, type=float,
                    help="ACO mutation probability")
parser.add_argument("--aco_cost_penalty_weight", default=0.003, type=float,
                    help="Penalty for high-rank/high-top-k ACO candidates")
parser.add_argument("--aco_uncertainty_warmup_ratio", default=0.2, type=float,
                    help="Fraction of epochs before uncertainty-guided search starts")
parser.add_argument("--aco_fisher_update_interval", default=5, type=int,
                    help="Epoch interval for Fisher credit signal updates")
parser.add_argument("--lr_eta_min", default=1e-6, type=float,
                    help="Minimum learning rate for cosine annealing")
parser.add_argument('--checkpoint_metric', choices=['iou', 'train_main_loss'], default='iou')
parser.add_argument("--resume", action="store_true",
                    help="Resume training from a checkpoint")
parser.add_argument("--resume_path", type=str, default=None,
                    help="Path to the checkpoint used for resume")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--amp', nargs='?', const='bf16', choices=['off', 'bf16'], default='bf16',
                    help='Enable CUDA BF16 mixed precision; bare --amp selects bf16')

parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True, help='Enable TF32 CUDA matmul and cuDNN')
parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False, help='Enable cuDNN autotune; relaxes deterministic convolution')
parser.add_argument('--diagnostics', action='store_true')
parser.add_argument('--diagnostics_interval', type=int, default=100)
parser.add_argument('--complexity_type', choices=['paper_rfft', 'fft_energy_entropy', 'legacy_rfft'], default='paper_rfft')
parser.add_argument('--importance_type', choices=['paper_svd', 'a_row_l2_norm', 'legacy_ab_norm'], default='paper_svd')
parser.add_argument('--sarm_mode', choices=['learned_fft', 'random_rank', 'fixed_mean_rank'], default='learned_fft')
parser.add_argument('--grouping_mode', choices=['importance_grouping', 'random_grouping', 'sequential_grouping'], default='importance_grouping')
parser.add_argument('--credit_mode', choices=['fisher_weighted_credit', 'uniform_credit', 'global_credit'], default='fisher_weighted_credit')
parser.add_argument('--fixed_rank', type=int, default=8)
parser.add_argument('--fixed_ranks', default='', help='Four comma-separated calibrated decoder ranks')
parser.add_argument('--routing_policy', choices=['consistent', 'legacy_train_only'], default='consistent')
parser.add_argument('--grouping_seed', type=int, default=2026)
parser.add_argument('--lambda_orth', type=float, default=0.01)

def set_random_seed(seed):
    """Sets random seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


set_seed = set_random_seed


def capture_random_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_state(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


@torch.no_grad()
def validate(model, dataloader, device, amp='bf16', dice_weight=1.0, boundary_weight=1.0,
             rank_logger=None, epoch=0):
    model.eval()
    metrics = SegmentationMetrics()
    total_loss, count = 0.0, 0
    for step, batch in enumerate(dataloader):
        image = batch['image'].to(device, non_blocking=True)
        mask = batch['label'].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and amp == 'bf16'):
            output, _ = model(image, update_hyperparams=False)
        output = output.float()
        total_loss += float(metric_oriented_loss(output, mask, dice_weight, boundary_weight)) * len(image)
        count += len(image)
        pred = output.sigmoid().cpu().numpy()
        gt = mask.cpu().numpy()
        for p, g in zip(pred, gt):
            metrics.step(np.clip(p[0] * 255, 0, 255).round().astype(np.uint8),
                         np.clip(g[0] * 255, 0, 255).round().astype(np.uint8))
        if rank_logger:
            rank_logger.record(model, batch, epoch, step, 'validation')
    result = metrics.results()
    result['loss'] = total_loss / count
    return result

def seed_worker(worker_id):
    """Seeds each DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def structure_loss(pred, mask):
    """
    Computes Structure Loss, combining Weighted BCE and Weighted IoU.
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def soft_dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    pred = pred.flatten(1)
    mask = mask.flatten(1)
    inter = (pred * mask).sum(dim=1)
    denom = pred.sum(dim=1) + mask.sum(dim=1)
    return (1 - (2 * inter + 1) / (denom + 1)).mean()

def boundary_loss(pred, mask):
    pred = torch.sigmoid(pred)
    pred_boundary = torch.abs(
        F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1) - pred
    )
    mask_boundary = torch.abs(
        F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask
    )
    return F.l1_loss(pred_boundary, mask_boundary)

def metric_oriented_loss(pred, mask, dice_weight=1.0, boundary_weight=1.0):
    return (
        structure_loss(pred, mask)
        + dice_weight * soft_dice_loss(pred, mask)
        + boundary_weight * boundary_loss(pred, mask)
    )

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) metric.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def calculate_prediction_uncertainty(pred):
    """
    Calculates mean binary prediction entropy from logits.
    """
    prob = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
    entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    return (entropy / math.log(2.0)).mean().item()

def time_text(seconds):
    """Converts seconds to HH:MM:SS format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

class ACOTrainer:
    """
    Trainer class managing the training loop, logging, and ACO scheduling.
    """

    def __init__(self, model, optimizer, initial_aco_weight=1.0,
                 dice_loss_weight=1.0, boundary_loss_weight=1.0, amp="bf16"):
        self.model = model
        self.amp_enabled = amp != "off" and next(model.parameters()).is_cuda
        self.amp_dtype = torch.bfloat16
        # BF16 needs no scaling; support PyTorch versions predating torch.amp.GradScaler.
        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.optimizer = optimizer
        self.dice_loss_weight = dice_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.aco_scheduler = AdaptiveTrainingScheduler(
            initial_aco_weight=initial_aco_weight,
            cooldown_epochs=10
        )
        self.aco_loss_history = []
        self.performance_history = []
        self.epoch_performances = []
        self.training_metric_history = []
        self.best_performance = 0.0
        self.best_hyperparams = None

        # Track history of expert utilization
        self.expert_usage_history = []
        self.layer_parameter_history = []
        self.rank_logger = None
        self.fisher_logger = None
        self.diagnostics_interval = 100

    def train_epoch(self, dataloader, epoch, total_epochs, device):
        """Executes a single training epoch."""
        self.model.train()
        self.model.set_epoch(epoch)
        epoch_total_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aco_loss = 0.0
        epoch_iou = 0.0
        epoch_dice, epoch_mae, samples = 0.0, 0.0, 0
        num_batches = len(dataloader)
        current_aco_weight = self.aco_scheduler.step(
            self.performance_history[-1] if self.performance_history else 0.0,
            self.performance_history)

        # Store batch-wise metrics for statistics
        batch_performances = []
        fisher_score_sums = {}
        fisher_score_count = 0
        prediction_uncertainties = []
        adaptation_stat_sums = {}
        adaptation_stat_count = 0

        for i, batch in enumerate(dataloader):
            x = batch['image'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Note: Update hyperparameters via ACO only on the first batch of the epoch
            with torch.autocast(device_type=device.type, dtype=self.amp_dtype,
                                enabled=self.amp_enabled):
                outputs, aco_loss = self.model(x, update_hyperparams=(i == 0))
            outputs = outputs.float()
            aco_loss = aco_loss.float()
            if self.rank_logger and i % self.diagnostics_interval == 0:
                self.rank_logger.record(self.model, batch, epoch, i)
            if hasattr(self.model, 'collect_adaptation_stats'):
                batch_adaptation_stats = self.model.collect_adaptation_stats()
                for layer_name, layer_stats in batch_adaptation_stats.items():
                    layer_accumulator = adaptation_stat_sums.setdefault(layer_name, {})
                    for key, value in layer_stats.items():
                        layer_accumulator[key] = layer_accumulator.get(key, 0.0) + value
                adaptation_stat_count += 1

            main_loss = metric_oriented_loss(
                outputs,
                target,
                dice_weight=self.dice_loss_weight,
                boundary_weight=self.boundary_loss_weight
            )

            # Calculate metrics
            batch_iou = calculate_iou(outputs, target)
            batch_performances.append(batch_iou)
            epoch_iou += batch_iou
            with torch.no_grad():
                probability = outputs.sigmoid()
                prediction, truth = probability > .5, target > .5
                intersection = (prediction & truth).flatten(1).sum(1).float()
                denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
                dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
                epoch_dice += float(dice.sum())
                epoch_mae += float((probability - target).abs().flatten(1).mean(1).sum())
                samples += len(x)

            total_loss = main_loss + current_aco_weight * aco_loss

            # Maintain a sliding window of ACO loss history
            self.aco_loss_history.append(aco_loss.item())
            if len(self.aco_loss_history) > 100:
                self.aco_loss_history.pop(0)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if hasattr(self.model, 'collect_lora_fisher_scores'):
                batch_fisher_scores = self.model.collect_lora_fisher_scores()
                if self.fisher_logger and i % self.diagnostics_interval == 0:
                    self.fisher_logger.add([dict(epoch=epoch, step=i, layer=key, fisher_score=value,
                                                 loss_source='L_total') for key, value in batch_fisher_scores.items()])
                for key, value in batch_fisher_scores.items():
                    fisher_score_sums[key] = fisher_score_sums.get(key, 0.0) + value
                fisher_score_count += 1
                prediction_uncertainties.append(calculate_prediction_uncertainty(outputs))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_total_loss += total_loss.item()
            epoch_main_loss += main_loss.item()
            epoch_aco_loss += aco_loss.item()

            if i % 50 == 0:
                print(f"Epoch:{epoch + 1}-{i + 1}/{num_batches} | "
                      f"Total: {total_loss.item():.4f} | "
                      f"Main: {main_loss.item():.4f} | "
                      f"ACO: {aco_loss.item():.4f} | "
                      f"IoU: {batch_iou:.4f} | "
                      f"ACO Weight: {current_aco_weight:.4f}")

        # Compute epoch statistics
        self.last_train_metrics = dict(mDice=epoch_dice / samples, MAE=epoch_mae / samples)
        print_metrics(f'Training epoch {epoch + 1}', self.last_train_metrics)
        avg_epoch_iou = epoch_iou / num_batches
        epoch_std_iou = np.std(batch_performances) if batch_performances else 0

        # Update global performance history
        self.performance_history.append(avg_epoch_iou)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

        # Process and record expert load statistics
        clean_stats = {}
        try:
            current_expert_stats = self.model.get_expert_statistics()
            if current_expert_stats:
                for layer_name, stats in current_expert_stats.items():
                    if 'usage_ratio' in stats:
                        # Convert Tensor to list for serialization
                        clean_stats[layer_name] = stats['usage_ratio'].cpu().tolist()

            self.expert_usage_history.append({
                'epoch': epoch,
                'stats': clean_stats
            })
        except Exception as e:
            print(f"Warning: Failed to record expert stats: {e}")

        # Snapshot current hyperparameters
        current_hyperparams_snapshot = copy.deepcopy(self.model.get_current_hyperparams())
        avg_fisher_scores = {
            key: value / max(fisher_score_count, 1)
            for key, value in fisher_score_sums.items()
        }
        avg_prediction_uncertainty = (
            float(np.mean(prediction_uncertainties)) if prediction_uncertainties else 0.0
        )
        avg_adaptation_stats = {
            layer_name: {
                key: value / max(adaptation_stat_count, 1)
                for key, value in layer_stats.items()
            }
            for layer_name, layer_stats in adaptation_stat_sums.items()
        }
        self.epoch_performances.append({
            'epoch': epoch,
            'mean_iou': avg_epoch_iou,
            'std_iou': epoch_std_iou,
            'hyperparams': current_hyperparams_snapshot,
            'expert_usage': clean_stats,
            'layer_fisher_scores': avg_fisher_scores,
            'prediction_uncertainty': avg_prediction_uncertainty,
            'adaptation_stats': avg_adaptation_stats
        })
        self.layer_parameter_history.extend(
            self._build_layer_parameter_records(
                epoch=epoch,
                mean_iou=avg_epoch_iou,
                hyperparams=current_hyperparams_snapshot,
                fisher_scores=avg_fisher_scores,
                prediction_uncertainty=avg_prediction_uncertainty,
                adaptation_stats=avg_adaptation_stats
            )
        )

        return (epoch_total_loss / num_batches,
                epoch_main_loss / num_batches,
                epoch_aco_loss / num_batches,
                avg_epoch_iou)

    def _build_layer_parameter_records(self, epoch, mean_iou, hyperparams,
                                       fisher_scores, prediction_uncertainty,
                                       adaptation_stats=None):
        adaptation_stats = adaptation_stats or {}
        pheromone_snapshot = {}
        if hasattr(self.model.aco_optimizer, 'get_layer_pheromone_snapshot'):
            pheromone_snapshot = self.model.aco_optimizer.get_layer_pheromone_snapshot()

        records = []
        encoder_params = hyperparams.get('encoder') or {}
        encoder_pheromone = pheromone_snapshot.get('encoder', {})
        records.append({
            'epoch': epoch,
            'layer': 'encoder',
            'rank': encoder_params.get('rank'),
            'alpha': encoder_params.get('alpha'),
            'top_k': None,
            'mean_iou': mean_iou,
            'fisher_score': fisher_scores.get('encoder', 0.0),
            'credit_score': encoder_pheromone.get('credit', 0.0),
            'prediction_uncertainty': prediction_uncertainty,
            'sample_rank': encoder_params.get('rank', 0.0),
            'rank_coefficient': 1.0,
            'input_complexity': 0.0,
            'orthogonality_loss': 0.0,
            'rank_smoothness_loss': 0.0,
            'rank_efficiency_loss': 0.0,
            'rank_pheromone': encoder_pheromone.get('rank', []),
            'alpha_pheromone': encoder_pheromone.get('alpha', []),
            'topk_pheromone': []
        })

        decoder_params = hyperparams.get('decoder') or {}
        decoder_layers = decoder_params.get('layers') or [
            decoder_params for _ in getattr(self.model, 'aco_layers', [])
        ]
        for layer_idx, layer_params in enumerate(decoder_layers):
            layer_name = f'decoder_{layer_idx}'
            layer_pheromone = pheromone_snapshot.get(layer_name, {})
            layer_adaptation = adaptation_stats.get(layer_name, {})
            records.append({
                'epoch': epoch,
                'layer': layer_name,
                'rank': layer_params.get('rank'),
                'alpha': layer_params.get('alpha'),
                'top_k': layer_params.get('top_k'),
                'mean_iou': mean_iou,
                'fisher_score': fisher_scores.get(layer_name, 0.0),
                'credit_score': layer_pheromone.get('credit', 0.0),
                'prediction_uncertainty': prediction_uncertainty,
                'sample_rank': layer_adaptation.get('sample_ranks', 0.0),
                'rank_coefficient': layer_adaptation.get('rank_coefficients', 0.0),
                'input_complexity': layer_adaptation.get('complexity', 0.0),
                'orthogonality_loss': layer_adaptation.get('orthogonality_loss', 0.0),
                'rank_smoothness_loss': layer_adaptation.get('rank_smoothness_loss', 0.0),
                'rank_efficiency_loss': layer_adaptation.get('rank_efficiency_loss', 0.0),
                'rank_pheromone': layer_pheromone.get('rank', []),
                'alpha_pheromone': layer_pheromone.get('alpha', []),
                'topk_pheromone': layer_pheromone.get('top_k', [])
            })
        return records

    def export_layer_parameter_history(self, save_path):
        if not self.layer_parameter_history:
            return

        json_path = os.path.join(save_path, 'layer_parameter_history.json')
        csv_path = os.path.join(save_path, 'layer_parameter_history.csv')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_parameter_history, f, ensure_ascii=False, indent=2)

        csv_fields = [
            'epoch', 'layer', 'rank', 'alpha', 'top_k', 'mean_iou',
            'fisher_score', 'credit_score', 'prediction_uncertainty',
            'sample_rank', 'rank_coefficient', 'input_complexity',
            'orthogonality_loss', 'rank_smoothness_loss', 'rank_efficiency_loss',
            'rank_pheromone', 'alpha_pheromone', 'topk_pheromone'
        ]
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for record in self.layer_parameter_history:
                row = dict(record)
                for key in ['rank_pheromone', 'alpha_pheromone', 'topk_pheromone']:
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                writer.writerow(row)

        self._plot_layer_parameter_history(save_path)
        print(f"Layer parameter history saved: {json_path}")
        print(f"Layer parameter CSV saved: {csv_path}")

    def _plot_layer_parameter_history(self, save_path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Skip layer parameter visualization: matplotlib unavailable ({e})")
            return

        layers = sorted({record['layer'] for record in self.layer_parameter_history})
        epochs = sorted({record['epoch'] for record in self.layer_parameter_history})
        metrics = [
            ('rank', 'Rank'),
            ('sample_rank', 'Mean Sample Rank'),
            ('fisher_score', 'Fisher Score'),
            ('credit_score', 'Credit Score'),
            ('prediction_uncertainty', 'Prediction Uncertainty')
        ]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_key, title) in zip(axes, metrics):
            for layer in layers:
                values = []
                for epoch in epochs:
                    record = next(
                        (item for item in self.layer_parameter_history
                         if item['layer'] == layer and item['epoch'] == epoch),
                        None
                    )
                    values.append(record.get(metric_key) if record else np.nan)
                ax.plot(epochs, values, marker='o', linewidth=1.5, label=layer)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Epoch')
        axes[0].legend(loc='best', fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(save_path, 'layer_parameter_history.png')
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Layer parameter visualization saved: {plot_path}")

    def record_epoch_metrics(self, epoch, total_loss, main_loss, aco_loss, iou, lr, epoch_time):
        self.training_metric_history.append({
            'epoch': epoch,
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aco_loss': aco_loss,
            'iou': iou,
            'lr': lr,
            'epoch_time': epoch_time,
            **self.last_train_metrics
        })

    def export_training_metric_history(self, save_path, plot=True):
        if not self.training_metric_history:
            return

        json_path = os.path.join(save_path, 'training_metrics.json')
        csv_path = os.path.join(save_path, 'training_metrics.csv')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_metric_history, f, ensure_ascii=False, indent=2)

        csv_fields = ['epoch', 'total_loss', 'main_loss', 'aco_loss', 'iou', 'mDice', 'MAE', 'lr', 'epoch_time']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(self.training_metric_history)

        if plot:
            self._plot_training_metric_history(save_path)
        print(f"Training metrics saved: {json_path}")
        print(f"Training metrics CSV saved: {csv_path}")

    def _plot_training_metric_history(self, save_path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Skip training metric visualization: matplotlib unavailable ({e})")
            return

        epochs = [record['epoch'] + 1 for record in self.training_metric_history]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(epochs, [record['total_loss'] for record in self.training_metric_history],
                     marker='o', label='Total Loss')
        axes[0].plot(epochs, [record['main_loss'] for record in self.training_metric_history],
                     marker='o', label='Main Loss')
        axes[0].plot(epochs, [record['aco_loss'] for record in self.training_metric_history],
                     marker='o', label='ACO Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend(loc='best')

        axes[1].plot(epochs, [record['iou'] for record in self.training_metric_history],
                     marker='o', color='tab:green')
        axes[1].set_ylabel('IoU')

        axes[2].plot(epochs, [record['lr'] for record in self.training_metric_history],
                     marker='o', color='tab:orange')
        axes[2].set_ylabel('LR')
        axes[2].set_xlabel('Epoch')

        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(save_path, 'training_metrics.png')
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Training metric visualization saved: {plot_path}")

    def update_aco_after_epoch(self, epoch):
        """
        Updates the ACO Optimizer based on the overall performance of the epoch.
        """
        if len(self.epoch_performances) == 0:
            return

        current_performance = self.epoch_performances[-1]
        performance_metric = current_performance['mean_iou']

        # Retrieve the configuration used in this epoch
        current_hyperparams = current_performance['hyperparams']
        solution = (current_hyperparams['encoder'], current_hyperparams['decoder'])

        # Update Pheromones
        self.model.aco_optimizer.update_pheromone(
            [(solution, performance_metric)],
            layer_scores=current_performance.get('layer_fisher_scores'),
            prediction_uncertainty=current_performance.get('prediction_uncertainty')
        )

        # Track Global Best
        if performance_metric > self.best_performance:
            self.best_performance = performance_metric
            self.best_hyperparams = current_hyperparams
            print(f"New Optimal Configuration: IoU={performance_metric:.4f}")

        # Log ACO internal statistics periodically
        if epoch % 10 == 0:
            stats = self.model.aco_optimizer.get_optimizer_stats()
            print(f"ACO Stats: Best Perf={stats['best_performance']:.4f}, "
                  f"Stagnation={stats['stagnation_count']}, Diversity={stats['current_diversity']:.3f}")

    def get_best_hyperparams(self):
        """Returns the best hyperparameters found during training."""
        return self.best_hyperparams or self.model.get_current_hyperparams()

    def get_training_stats(self):
        """Returns current training statistics."""
        return {
            'current_aco_weight': self.aco_scheduler.aco_weight,
            'recent_aco_loss_avg': np.mean(self.aco_loss_history[-20:]) if self.aco_loss_history else 0,
            'recent_performance_avg': np.mean(self.performance_history[-10:]) if self.performance_history else 0,
            'performance_trend': self._calculate_performance_trend()
        }

    def _calculate_performance_trend(self):
        """Calculates the trend of performance improvement over recent epochs."""
        if len(self.performance_history) < 5:
            return 0
        recent = self.performance_history[-5:]
        return (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0

def main(args):
    args.implementation_version = 'trievo_paper_release_20261029'

    set_random_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = not args.cudnn_benchmark
    torch.use_deterministic_algorithms(not args.cudnn_benchmark)
    if args.amp != "off" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unsupported on this GPU; use --amp off")
    print(f"Performance: amp={args.amp}, tf32={args.tf32}, "
          f"cudnn_benchmark={args.cudnn_benchmark}, workers={args.num_workers}")

    if len(args.train_image_paths) != len(args.train_mask_paths):
        raise ValueError("Number of image paths must match number of mask paths")

    if args.diagnostics_interval < 1:
        raise ValueError('diagnostics_interval must be positive')
    datasets = [FullDataset(img_path, mask_path, 352, mode='train')
                for img_path, mask_path in zip(args.train_image_paths, args.train_mask_paths)]
    combined_dataset = ConcatDataset(datasets)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=False,
                            worker_init_fn=seed_worker, generator=generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ACO Optimizer Configuration
    aco_config = {
        'rank_candidates': [8, 16, 24, 32],
        'alpha_candidates': [8.0, 16.0, 24.0, 32.0, 48.0],
        'top_k_candidates': [1, 2, 3],
        'num_ants': args.aco_num_ants,
        'evaporation_rate': args.aco_evaporation_rate,
        'diversification_interval': args.aco_diversification_interval,
        'mutation_prob': args.aco_mutation_prob,
        'cost_penalty_weight': args.aco_cost_penalty_weight,
        'uncertainty_warmup_ratio': args.aco_uncertainty_warmup_ratio,
        'fisher_update_interval': args.aco_fisher_update_interval,
        'total_epochs': args.epoch
    }

    aco_config.update(alpha=1.5, beta=2.0)

    # Model Initialization
    model = TriEvoMoELoRA(
        checkpoint_path=args.hiera_path,
        model_cfg=args.model_cfg,
        num_experts=args.num_experts,
        use_multi_start=args.use_multi_start,
        aco_optimizer_config=aco_config
    )
    model.to(device)
    model.aco_optimizer.paper_fixed_beta = False
    configure_model(model, vars(args))

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not 0 < args.encoder_lr_scale <= 1:
        raise ValueError('encoder_lr_scale must be in (0, 1]')
    parameter_groups = [
        {'params': [p for name, p in model.named_parameters()
                    if p.requires_grad and name.startswith('encoder.')],
         'lr': args.lr * args.encoder_lr_scale},
        {'params': [p for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith('encoder.')],
         'lr': args.lr},
    ]
    optimizer = opt.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr_eta_min)

    trainer = ACOTrainer(
        model,
        optimizer,
        args.aco_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        boundary_loss_weight=args.boundary_loss_weight,
        amp=args.amp
    )

    os.makedirs(args.save_path, exist_ok=True)
    validation_history = []
    best_validation_score = -float('inf')
    if args.diagnostics:
        trainer.rank_logger = RankLogger(os.path.join(args.save_path, 'rank_diagnostics.csv'))
        trainer.fisher_logger = BufferedCSV(os.path.join(args.save_path, 'fisher_diagnostics.csv'))
        trainer.diagnostics_interval = args.diagnostics_interval
    overlap_logger = BufferedCSV(os.path.join(args.save_path, 'expert_overlap.csv')) if args.diagnostics else None

    best_loss = float('inf')
    best_iou = 0.0
    best_checkpoint_hyperparams = None
    start_epoch = 0

    if args.resume:
        resume_path = args.resume_path or os.path.join(args.save_path, 'best.pth')
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = load_legacy_checkpoint(resume_path, device)
        if 'random_state' not in checkpoint:
            raise ValueError('This legacy checkpoint has no RNG state; exact resume requires a new run')
        if checkpoint.get('config', {}).get('training_protocol', 'holdout') != args.training_protocol:
            raise ValueError('Resume must use the same training protocol')
        if checkpoint.get('config', {}).get('implementation_version') != args.implementation_version:
            raise ValueError('Resume requires the same implementation version; start a fresh run after paper alignment')
        if checkpoint.get('config') and checkpoint['config'] != vars(args):
            for key in ('complexity_type', 'importance_type', 'sarm_mode', 'grouping_mode', 'credit_mode', 'lambda_orth', 'grouping_seed', 'fixed_rank', 'checkpoint_metric', 'seed', 'dice_loss_weight', 'boundary_loss_weight', 'aco_loss_weight', 'routing_policy', 'encoder_lr_scale', 'num_workers', 'amp', 'tf32', 'cudnn_benchmark'):
                if checkpoint['config'].get(key) != getattr(args, key):
                    raise ValueError(f'Resume configuration mismatch: {key}')
        restore_checkpoint_model(model, checkpoint)
        configure_model(model, vars(args))
        best_validation_score = checkpoint.get('best_validation_score', -float('inf'))
        validation_history = checkpoint.get('validation_history', [])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_loss = checkpoint.get('loss', best_loss)
        best_iou = checkpoint.get('iou', best_iou)
        start_epoch = checkpoint.get('epoch', -1) + 1
        trainer.best_hyperparams = checkpoint.get('best_hyperparams')
        best_checkpoint_hyperparams = checkpoint.get('current_hyperparams') or checkpoint.get('best_hyperparams')
        trainer.best_performance = best_iou
        trainer.layer_parameter_history = checkpoint.get('layer_parameter_history', [])
        trainer.training_metric_history = checkpoint.get('training_metric_history', [])
        trainer.performance_history = checkpoint.get('performance_history', [])
        trainer.aco_loss_history = checkpoint.get('aco_loss_history', [])
        trainer.epoch_performances = checkpoint.get('epoch_performances', [])
        model.training_iteration = checkpoint.get('training_iteration', 0)
        if 'aco_scheduler_state' in checkpoint:
            trainer.aco_scheduler.__dict__.update(checkpoint['aco_scheduler_state'])

        if 'aco_optimizer_state' in checkpoint:
            model.aco_optimizer = checkpoint['aco_optimizer_state']
            configure_model(model, vars(args))

        print(f"Resumed training from: {resume_path}")
        print(f"Resume start epoch: {start_epoch + 1}/{args.epoch}")
        print(f"Previous best IoU: {best_iou:.4f}, Loss: {best_loss:.4f}")
        restore_random_state(checkpoint['random_state'], generator)

    start_time = time.time()

    # Logging Information
    print(f"Starting training with {len(trainable_params)} trainable parameters")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} total parameters")
    print(f"Trainable: {sum(p.numel() for p in trainable_params):,} parameters")
    print(f"Using {'Multi-Start ' if args.use_multi_start else ''}Enhanced ACO Optimizer")
    print(f"ACO Configuration: {aco_config}")
    print(f"Random Seed: {args.seed}")

    # Training Loop
    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()

        # Train one epoch
        avg_total_loss, avg_main_loss, avg_aco_loss, avg_iou = trainer.train_epoch(
            dataloader, epoch, args.epoch, device)

        # Update ACO based on epoch performance
        trainer.update_aco_after_epoch(epoch)
        if trainer.rank_logger:
            trainer.rank_logger.flush()
            trainer.fisher_logger.flush()
            overlap_logger.add(expert_overlap(model, epoch))
            overlap_logger.flush()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[-1]

        # Timing and Logging
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        progress = (epoch + 1) / args.epoch
        estimated_total_time = total_time / progress if progress > 0 else 0

        print(f"Epoch {epoch + 1}/{args.epoch} | "
              f"Total: {avg_total_loss:.4f} | "
              f"Main: {avg_main_loss:.4f} | "
              f"ACO: {avg_aco_loss:.4f} | "
              f"IoU: {avg_iou:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {time_text(epoch_time)} |"
              f"Total: {time_text(total_time)} |"
              f"Estimated: {time_text(estimated_total_time)}"
              )
        trainer.record_epoch_metrics(
            epoch=epoch,
            total_loss=avg_total_loss,
            main_loss=avg_main_loss,
            aco_loss=avg_aco_loss,
            iou=avg_iou,
            lr=current_lr,
            epoch_time=epoch_time
        )
        trainer.export_training_metric_history(args.save_path, plot=False)

        # Save Best Model
        score = avg_iou if args.checkpoint_metric == 'iou' else -avg_main_loss
        is_best_checkpoint = score > best_validation_score

        if is_best_checkpoint:
            best_validation_score = score
            best_iou = avg_iou
            best_loss = avg_main_loss
            best_path = os.path.join(args.save_path, 'best.pth')

            current_hyperparams = copy.deepcopy(model.get_current_hyperparams())
            best_checkpoint_hyperparams = current_hyperparams
            aco_best_hyperparams = trainer.get_best_hyperparams()

            torch.save({
                'epoch': epoch,
                'training_iteration': model.training_iteration,
                'performance_history': trainer.performance_history,
                'aco_loss_history': trainer.aco_loss_history,
                'epoch_performances': trainer.epoch_performances,
                'aco_scheduler_state': trainer.aco_scheduler.__dict__,
                'config': vars(args),
                'validation_metrics': None,
                'validation_history': validation_history,
                'best_validation_score': best_validation_score,
                'split_manifest': None,
                'selection_source': 'train_iou' if args.checkpoint_metric == 'iou' else 'train_main_loss',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': trainer.scaler.state_dict(),
                'amp': args.amp,
                'tf32': args.tf32,
                'cudnn_benchmark': args.cudnn_benchmark,
                'loss': best_loss,
                'iou': best_iou,
                'current_hyperparams': current_hyperparams,
                'best_hyperparams': current_hyperparams,
                'aco_best_hyperparams': aco_best_hyperparams,
                'layer_parameter_history': trainer.layer_parameter_history,
                'training_metric_history': trainer.training_metric_history,
                'seed': args.seed,
                'random_state': capture_random_state(generator),
                'dice_loss_weight': args.dice_loss_weight,
                'boundary_loss_weight': args.boundary_loss_weight,
                'checkpoint_metric': args.checkpoint_metric,
                'aco_optimizer_state': model.aco_optimizer,
            }, best_path)
            print(
                f'[Best Model Saved.] Metric: {args.checkpoint_metric} | '
                f'IoU: {best_iou:.4f}, Loss: {best_loss:.4f}'
            )
            with open(os.path.join(args.save_path, 'results.json'), 'w', encoding='utf-8') as stream:
                json.dump(dict(dataset='training', seed=args.seed,
                               metrics=dict(loss=avg_main_loss, mIoU=avg_iou),
                               protocol=experiment_protocol(vars(args))), stream, indent=2)

        # Log Expert Statistics
        try:
            stats = model.get_expert_statistics()
            if stats:
                print("Expert Statistics:")
                for layer_name, layer_stats in stats.items():
                    if isinstance(layer_stats, dict) and 'min_usage' in layer_stats:
                        print(f"  {layer_name}: Min Usage={layer_stats['min_usage']:.3f}, "
                              f"Max Usage={layer_stats['max_usage']:.3f}, "
                              f"Std Usage={layer_stats['std_usage']:.3f}")
        except Exception as e:
            print(f"Failed to retrieve expert statistics: {e}")

        print("-" * 80)

    runtime = time.time() - start_time
    write_csv(os.path.join(args.save_path, 'training_compute.csv'), [dict(
        seed=args.seed, elapsed_seconds=runtime, GPU_hours=runtime / 3600 if device.type == 'cuda' else None,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        completed_epochs=max(0, args.epoch - start_epoch), amp=args.amp)])

    trainer.export_layer_parameter_history(args.save_path)
    trainer.export_training_metric_history(args.save_path)

    # Final Summary
    final_best_hyperparams = best_checkpoint_hyperparams or model.get_current_hyperparams()
    print("\n=== Training completed ===")
    print(f"{args.training_protocol} selected checkpoint topology:")
    print(f"  ACO-LoRA-A (Encoder): {final_best_hyperparams['encoder']}")
    print(f"  ACO-LoRA-B (Decoder): {final_best_hyperparams['decoder']}")
    print(f"Training IoU at selected checkpoint: {best_iou:.4f}")


if __name__ == "__main__":

    main(parser.parse_args())
