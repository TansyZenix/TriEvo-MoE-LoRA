import os
import cv2
import py_sod_metrics
import argparse
import csv
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from reviewer_utils import SegmentationMetrics, write_csv, print_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--pred_path", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str, required=True,
                    help="path to the ground truth masks")
parser.add_argument("--save_path", type=str, default=None,
                    help="directory to save evaluation metrics")
parser.add_argument("--result_file", type=str, default=None,
                    help="csv file path to save evaluation metrics")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--protocol_file', default=None, help='test.py results.json with seed/protocol metadata')
parser.add_argument('--metric_workers', type=int, default=4)
args = parser.parse_args()

pred_root = args.pred_path
mask_root = args.gt_path
metadata_file = args.protocol_file or os.path.join(args.pred_path, 'results.json')
metadata = {}
if os.path.isfile(metadata_file):
    with open(metadata_file, encoding='utf-8') as stream:
        metadata = json.load(stream)
threshold = metadata.get('protocol', {}).get('threshold', 0.5)
mask_name_list = sorted(name for name in os.listdir(mask_root) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
canonical_meter = SegmentationMetrics(threshold)
per_image = []
def measure_mask(mask_name):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, os.path.splitext(mask_name)[0] + '.png')
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imdecode(np.fromfile(pred_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or pred is None:
        raise FileNotFoundError(f'Unreadable prediction or mask: {pred_path}, {mask_path}')
    sample_meter = SegmentationMetrics(threshold)
    sample_meter.step(pred, mask)
    curves = py_sod_metrics.FmeasureV2(metric_handlers={
        'iou': py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True),
        'dice': py_sod_metrics.DICEHandler(with_adaptive=True, with_dynamic=True),
    })
    curves.step(pred=pred, gt=mask)
    return mask_name, sample_meter, curves.get_results()


with ThreadPoolExecutor(max_workers=max(1, args.metric_workers)) as pool:
    # Bounded chunks avoid retaining all full-resolution masks/curves in futures.
    chunk_size = max(1, args.metric_workers) * 2
    auxiliary = {'iou': [], 'dice': []}
    for start in range(0, len(mask_name_list), chunk_size):
        for mask_name, sample_meter, curves in pool.map(measure_mask, mask_name_list[start:start + chunk_size]):
            canonical_meter.merge(sample_meter)
            per_image.append(dict(sample_id=mask_name, **sample_meter.results()))
            for key in auxiliary:
                auxiliary[key].append(curves[key]['dynamic'])
        print(f'Processed {min(start + chunk_size, len(mask_name_list))}/{len(mask_name_list)} masks', flush=True)

fmv2 = {key: {'dynamic': np.mean(values, axis=0)} for key, values in auxiliary.items()}
canonical = canonical_meter.results()
sm, wfm, mae = canonical['S_alpha'], canonical['F_beta_w'], canonical['MAE']
fm = {'adp': canonical['F_beta']}
em = canonical_meter.meters['E_phi'].get_results()['em']


curr_results = {
    "meandice": canonical['mDice'],
    "meaniou": canonical['mIoU'],
    'Smeasure': sm,
    "wFmeasure": wfm,  # For Marine Animal Segmentation
    "adpFm": fm["adp"], # For Camouflaged Object Detection
    "meanEm": em["curve"].mean(),
    "maxEm": em["curve"].max(),
    "MAE": mae,
}

print(f'{args.dataset_name} (per-image Dice/IoU, threshold={threshold})')
print("mDice:        ", format(curr_results['meandice'], '.3f'))
print("mIoU:         ", format(curr_results['meaniou'], '.3f'))
print("S_{alpha}:    ", format(curr_results['Smeasure'], '.3f'))
print("F^{w}_{beta}: ", format(curr_results['wFmeasure'], '.3f'))
print("F_{beta}:     ", format(curr_results['adpFm'], '.3f'))
print("E_{phi}:      ", format(curr_results['meanEm'], '.3f'))
print("E^{max}_{phi}:", format(curr_results['maxEm'], '.3f'))
print("MAE:          ", format(curr_results['MAE'], '.3f'))

if args.save_path:
    os.makedirs(args.save_path, exist_ok=True)
    metrics_path = os.path.join(args.save_path, "val_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Dataset: {args.dataset_name}\n")
        file.write(f"mDice: {curr_results['meandice']:.3f}\n")
        file.write(f"mIoU: {curr_results['meaniou']:.3f}\n")
        file.write(f"Smeasure: {curr_results['Smeasure']:.3f}\n")
        file.write(f"wFmeasure: {curr_results['wFmeasure']:.3f}\n")
        file.write(f"Fmeasure: {curr_results['adpFm']:.3f}\n")
        file.write(f"meanEm: {curr_results['meanEm']:.3f}\n")
        file.write(f"maxEm: {curr_results['maxEm']:.3f}\n")
        file.write(f"MAE: {curr_results['MAE']:.3f}\n")

if args.result_file:
    os.makedirs(os.path.dirname(args.result_file) or ".", exist_ok=True)
    with open(args.result_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset", "mDice", "mIoU", "Smeasure", "wFmeasure",
                "Fmeasure", "meanEm", "maxEm", "MAE"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": args.dataset_name,
            "mDice": curr_results['meandice'],
            "mIoU": curr_results['meaniou'],
            "Smeasure": curr_results['Smeasure'],
            "wFmeasure": curr_results['wFmeasure'],
            "Fmeasure": curr_results['adpFm'],
            "meanEm": curr_results['meanEm'],
            "maxEm": curr_results['maxEm'],
            "MAE": curr_results['MAE'],
        })

# All primary outputs use the same fixed-threshold Dice/IoU protocol.
# Historical threshold-sweep metrics remain explicitly named auxiliary columns.
output_dir = args.save_path or args.pred_path
os.makedirs(output_dir, exist_ok=True)
print_metrics('Evaluation (fixed threshold Dice/IoU)', canonical)
write_csv(os.path.join(output_dir, 'per_image_metrics.csv'), per_image)
write_csv(os.path.join(output_dir, 'metrics_full_precision.csv'), [dict(dataset=args.dataset_name,
    mDice_threshold_mean=float(fmv2['dice']['dynamic'].mean()),
    mIoU_threshold_mean=float(fmv2['iou']['dynamic'].mean()), **canonical)])
protocol = dict(metadata.get('protocol', {}), threshold=threshold,
                dice_iou='per_image_binary_mean', empty_both=1.0,
                E_phi='threshold_curve_mean', F_beta='adaptive_beta2_0.3')
with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as stream:
    json.dump(dict(dataset=args.dataset_name, seed=args.seed if args.seed is not None else metadata.get('seed'),
                   metrics=canonical, protocol=protocol), stream, indent=2)
