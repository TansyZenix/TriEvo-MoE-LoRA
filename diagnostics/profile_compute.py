import time
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch


def profile(model, batch_size=1, size=352, warmup=30, repeat=100, amp='bf16'):
    if warmup < 30 or repeat < 100:
        raise ValueError('Profile requires warmup >= 30 and repeat >= 100')
    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    times = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and amp == 'bf16'):
        for _ in range(warmup):
            model(x, update_hyperparams=False)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeat):
            sync()
            start = time.perf_counter()
            model(x, update_hyperparams=False)
            sync()
            times.append(time.perf_counter() - start)
    return dict(total_params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                latency_mean_ms=float(np.mean(times) * 1000), latency_std_ms=float(np.std(times, ddof=1) * 1000),
                latency_median_ms=float(np.median(times) * 1000), latency_p95_ms=float(np.percentile(times, 95) * 1000),
                images_per_second=float(batch_size / np.mean(times)),
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                torch_version=torch.__version__, amp=amp if device.type == 'cuda' else 'off', batch_size=batch_size, image_size=size,
                warmup=warmup, repeat=repeat, TTA=False, FLOPs=None, training_GPU_hours=None,
                tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark)


if __name__ == '__main__':
    import argparse
    import json
    from TriEvoMoELoRA import TriEvoMoELoRA
    from reviewer_utils import load_legacy_checkpoint, restore_checkpoint_model, write_csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--size', type=int, default=352)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--amp', choices=['off', 'bf16'], default='bf16')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    checkpoint = load_legacy_checkpoint(args.checkpoint, device)
    model = TriEvoMoELoRA(num_experts=checkpoint.get('config', {}).get('num_experts', 8)).to(device)
    restore_checkpoint_model(model, checkpoint)
    values = profile(model, args.batch_size, args.size, args.warmup, args.repeat, args.amp)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'compute_profile.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    write_csv(output / 'compute_profile.csv', [values])
    print(values)
