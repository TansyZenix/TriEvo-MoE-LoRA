import argparse
import sys
import json
from collections import defaultdict
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reviewer_utils import write_csv


def summarize(files, output):
    groups = defaultdict(list)
    for filename in files:
        record = json.loads(Path(filename).read_text(encoding='utf-8'))
        if record.get('seed') is None:
            raise ValueError(f'Missing seed: {filename}')
        groups[(record['dataset'], json.dumps(record.get('protocol', {}), sort_keys=True))].append(record)
    rows = []
    for (dataset, protocol), records in groups.items():
        if len({r['seed'] for r in records}) != len(records):
            raise ValueError('Duplicate seed within dataset/protocol')
        for metric in ('S_alpha', 'F_beta_w', 'E_phi', 'MAE', 'mDice', 'mIoU'):
            values = [r['metrics'][metric] for r in records]
            mean, std = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float('nan')
            rows.append(dict(dataset=dataset, protocol=protocol, metric=metric, seeds=len(values), mean=mean,
                             std=std, display=f'{mean:.6f} ± {std:.6f}'))
            print(dataset, metric, rows[-1]['display'])
    write_csv(output, rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.results, args.output)
