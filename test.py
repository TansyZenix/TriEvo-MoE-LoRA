import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from TriEvoMoELoRA import TriEvoMoELoRA
from dataset import TestDataset
import cv2
import time
import json
from reviewer_utils import (load_legacy_checkpoint, restore_checkpoint_model, SegmentationMetrics,
                            write_csv, print_metrics, experiment_protocol)

# === Argument Parsing ===
parser = argparse.ArgumentParser("TriEvo-MoE-LoRA Inference Script")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="Name of the dataset (used for logging)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the model checkpoint file")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="Directory containing test images")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="Directory containing ground truth masks")
parser.add_argument("--save_path", type=str, required=True,
                    help="Directory to save the predicted masks")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for binary segmentation")
parser.add_argument("--use_tta", type=int, default=0,
                    help="Use multi-scale and flip test-time augmentation")
parser.add_argument("--tta_scales", type=str, default="0.75,1.0,1.25",
                    help="Comma-separated TTA input scales")
parser.add_argument("--use_postprocess", type=int, default=0,
                    help="Use grayscale probability-map refinement before saving")
parser.add_argument("--postprocess_gamma", type=float, default=0.9,
                    help="Gamma used by probability-map refinement")
parser.add_argument('--seed', type=int, default=None, help='Defaults to the checkpoint training seed')
parser.add_argument('--metric_workers', type=int, default=4)


def resolve_checkpoint_hyperparams(checkpoint):
    """Returns the exact topology used by the checkpoint weights."""
    if checkpoint.get("current_hyperparams"):
        return checkpoint["current_hyperparams"]

    checkpoint_epoch = checkpoint.get("epoch")
    history = checkpoint.get("layer_parameter_history") or []
    epoch_records = [
        record for record in history
        if record.get("epoch") == checkpoint_epoch
    ]
    encoder_record = next(
        (record for record in epoch_records if record.get("layer") == "encoder"),
        None,
    )
    decoder_records = sorted(
        (
            record for record in epoch_records
            if str(record.get("layer", "")).startswith("decoder_")
        ),
        key=lambda record: int(record["layer"].split("_")[-1]),
    )
    if encoder_record and decoder_records:
        layers = [
            {
                "rank": record["rank"],
                "alpha": record["alpha"],
                "top_k": record["top_k"],
            }
            for record in decoder_records
        ]
        return {
            "encoder": {
                "rank": encoder_record["rank"],
                "alpha": encoder_record["alpha"],
            },
            "decoder": {
                **layers[0],
                "layers": layers,
            },
        }

    return checkpoint.get("best_hyperparams")


def prediction_to_uint8(prediction):
    """Preserves probability information for grayscale segmentation metrics."""
    if not np.isfinite(prediction).all():
        raise ValueError('Prediction contains NaN/Inf; refusing to save an invalid mask')
    return np.clip(prediction * 255.0, 0, 255).round().astype(np.uint8)


def refine_prediction_map(prediction, gamma=0.9):
    """Normalizes and lightly smooths a grayscale probability map for SOD metrics."""
    refined = prediction.astype(np.float32)
    pred_min = float(refined.min())
    pred_max = float(refined.max())
    if pred_max - pred_min > 1e-6:
        refined = (refined - pred_min) / (pred_max - pred_min)

    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    refined = np.clip(refined, 0.0, 1.0)
    refined = np.power(refined, gamma)
    return np.clip(refined, 0.0, 1.0)


def parse_tta_scales(scales_text):
    scales = [
        float(item.strip())
        for item in scales_text.split(",")
        if item.strip()
    ]
    return scales or [1.0]


def scale_to_stride_size(height, width, scale, stride=32):
    scaled_h = max(stride, int(round(height * scale / stride) * stride))
    scaled_w = max(stride, int(round(width * scale / stride) * stride))
    return scaled_h, scaled_w


def finite_model_output(model, image):
    output, _ = model(image, update_hyperparams=False)
    if not torch.isfinite(output).all():
        print(f'Non-finite output at {tuple(image.shape[-2:])}; retrying in FP32', flush=True)
        with torch.autocast(device_type=image.device.type, enabled=False):
            output, _ = model(image.float(), update_hyperparams=False)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Model output contains NaN/Inf even in FP32')
    return output.float()


def tta_predict(model, image, scales):
    """Fuse finite logits in FP32 under the caller's inference precision."""
    _, _, base_h, base_w = image.shape
    accumulated = None
    count = 0

    for scale in scales:
        if scale == 1.0:
            scaled_image = image
        else:
            scaled_h, scaled_w = scale_to_stride_size(base_h, base_w, scale)
            scaled_image = F.interpolate(
                image,
                size=(scaled_h, scaled_w),
                mode='bilinear',
                align_corners=False
            )

        output = finite_model_output(model, scaled_image)
        output = F.interpolate(output, size=(base_h, base_w), mode='bilinear', align_corners=False)
        if accumulated is None:
            accumulated = output.clone()
        else:
            accumulated += output
        count += 1

        flipped_image = torch.flip(scaled_image, dims=[3])
        flipped_output = finite_model_output(model, flipped_image)
        flipped_output = torch.flip(flipped_output, dims=[3])
        flipped_output = F.interpolate(
            flipped_output,
            size=(base_h, base_w),
            mode='bilinear',
            align_corners=False
        )
        accumulated += flipped_output
        count += 1

    return accumulated / count


def set_deterministic(seed=42):
    """
    Sets seeds for all random number generators to ensure reproducible results.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def measure_prediction(prediction, gt, threshold, output_path):
    sample = SegmentationMetrics(threshold)
    sample.step(prediction, gt)
    encoded, buffer = cv2.imencode('.png', prediction)
    if not encoded:
        raise OSError(f'Failed to encode prediction: {output_path}')
    buffer.tofile(output_path)
    return sample


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_legacy_checkpoint(args.checkpoint, 'cpu')
    seed = args.seed if args.seed is not None else checkpoint.get('seed', checkpoint.get('config', {}).get('seed', 42))
    set_deterministic(seed)
    model = TriEvoMoELoRA(
        num_experts=checkpoint.get('config', {}).get('num_experts', 8),
        model_cfg=checkpoint.get('config', {}).get('model_cfg', 'configs/sam2.1/sam2.1_hiera_l.yaml')).to(device)
    restore_checkpoint_model(model, checkpoint)
    model.eval()

    # Apply optimal hyperparameters from the best training epoch
    checkpoint_hyperparams = resolve_checkpoint_hyperparams(checkpoint)
    if checkpoint_hyperparams:
        print("Loading the exact hyperparameters used by the checkpoint weights:")
        print(
            f"  Encoder: rank={checkpoint_hyperparams['encoder']['rank']}, "
            f"alpha={checkpoint_hyperparams['encoder']['alpha']}"
        )
        decoder_params = checkpoint_hyperparams["decoder"]
        decoder_layers = decoder_params.get("layers") or [
            decoder_params for _ in model.aco_layers
        ]
        print(
            "  Decoder: " + ", ".join(
                f"L{i}:r{params['rank']}/a{params['alpha']}/k{params['top_k']}"
                for i, params in enumerate(decoder_layers)
            )
        )

        model._apply_encoder_hyperparams(checkpoint_hyperparams['encoder'])
        for layer, layer_params in zip(model.aco_layers, decoder_layers):
            layer.set_hyperparams(
                rank=layer_params['rank'],
                alpha=layer_params['alpha'],
                top_k=layer_params['top_k']
            )
        model.current_encoder_params = checkpoint_hyperparams["encoder"]
        model.current_decoder_params = checkpoint_hyperparams["decoder"]
    else:
        # Fallback: Use default parameters or those set explicitly in the model class
        print("Warning: No 'best_hyperparams' found in checkpoint. Using default test parameters.")
        model.set_test_hyperparams()

    # Disable stochastic behaviors (e.g., noisy gating) for deterministic inference
    model.set_deterministic_mode()

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=max(1, args.metric_workers)) as metric_pool:
        test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
        os.makedirs(args.save_path, exist_ok=True)

        print(f"Testing on {test_loader.size} images")
        print(f"Using device: {device}")
        tta_scales = parse_tta_scales(args.tta_scales)
        print(f"TTA: {'enabled' if args.use_tta else 'disabled'} | scales={tta_scales}")
        print(
            f"Postprocess: {'enabled' if args.use_postprocess else 'disabled'} | "
            f"gamma={args.postprocess_gamma}"
        )

        total_time = 0
        iou_scores = []
        metric_meter = SegmentationMetrics(args.threshold)
        per_image = []
        pending = deque()

        def collect_sample():
            name, inference_time, future = pending.popleft()
            sample_meter = future.result()
            metric_meter.merge(sample_meter)
            metrics = sample_meter.results()
            iou_scores.append(metrics['mIoU'])
            per_image.append(dict(sample_id=name, seed=seed, elapsed_seconds=inference_time, **metrics))

        for i in range(test_loader.size):
            start_time = time.time()

            image, gt, name = test_loader.load_data()
            image = image.to(device)
            # Ground truth preparation
            gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)

            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda',
                                dtype=torch.bfloat16):
                if args.use_tta:
                    res = tta_predict(model, image, tta_scales)
                else:
                    res = finite_model_output(model, image)

            # Resize prediction to match ground truth resolution
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = torch.sigmoid(res).squeeze().cpu().numpy()
            if args.use_postprocess:
                res = refine_prediction_map(res, gamma=args.postprocess_gamma)

            inference_time = time.time() - start_time
            total_time += inference_time

            # Save the grayscale probability map required by eval.py metrics.
            res_output = prediction_to_uint8(res)
            output_path = os.path.join(args.save_path, os.path.splitext(name)[0] + ".png")
            future = metric_pool.submit(measure_prediction, res_output, gt, args.threshold, output_path)
            pending.append((name, inference_time, future))
            if len(pending) >= max(1, args.metric_workers) * 2:
                collect_sample()

            if i % 10 == 0:
                print(f"Processed {i + 1}/{test_loader.size}: {name} "
                      f"Time: {inference_time:.3f}s")

        while pending:
            collect_sample()

        # Compute Aggregate Statistics
        avg_time = total_time / test_loader.size
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)

        print(f"\n=== Test Results ===")
        print(f"Dataset: {args.dataset_name}")
        print(f"Average IoU: {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"Average inference time: {avg_time:.3f}s per image")
        print(f"Results saved to: {args.save_path}")

        # Log statistics to file
        stats_path = os.path.join(args.save_path, "test_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Test Images: {test_loader.size}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Std IoU: {std_iou:.4f}\n")
            f.write(f"Average Time: {avg_time:.3f}s\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write(f"TTA: {args.use_tta}\n")
            f.write(f"TTA Scales: {args.tta_scales}\n")
            f.write(f"Postprocess: {args.use_postprocess}\n")
            f.write(f"Postprocess Gamma: {args.postprocess_gamma}\n")
        metrics = metric_meter.results()
        print_metrics('Test metrics (fixed-threshold Dice/IoU)', metrics)
        write_csv(os.path.join(args.save_path, 'per_image_metrics.csv'), per_image)
        write_csv(os.path.join(args.save_path, 'test_metrics.csv'), [dict(dataset=args.dataset_name,
                  seed=seed, **metrics)])
        with open(os.path.join(args.save_path, 'results.json'), 'w', encoding='utf-8') as stream:
            json.dump(dict(dataset=args.dataset_name, seed=seed, metrics=metrics,
                           protocol=dict(TTA=args.use_tta, scales=args.tta_scales, threshold=args.threshold,
                                         dice_iou='per_image_binary_mean', empty_both=1.0,
                                         E_phi='threshold_curve_mean', F_beta='adaptive_beta2_0.3',
                                         postprocess=args.use_postprocess, gamma=args.postprocess_gamma,
                                         method=experiment_protocol(checkpoint.get('config', {}))
                                         )), stream, indent=2)


if __name__ == "__main__":
    main(parser.parse_args())
