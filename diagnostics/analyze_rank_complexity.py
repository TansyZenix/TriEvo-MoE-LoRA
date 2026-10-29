import argparse
import sys
import csv
from collections import defaultdict
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr
from reviewer_utils import write_csv


def analyze(path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(path, encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty rank diagnostics')
    groups = defaultdict(list)
    for row in rows:
        # Keep phase separate; training augmentation differs from validation.
        phase = row.get('phase', 'unknown')
        groups[(phase, 'overall', 'all')].append(row)
        groups[(phase, 'layer', row['layer_id'])].append(row)
        groups[(phase, 'dataset', row['dataset'])].append(row)
    result = []
    for (phase, scope, name), values in groups.items():
        entropy, rank, coef = [np.array([float(r[key]) for r in values]) for key in ('fft_entropy', 'active_rank', 'rank_coefficient')]
        def correlation(y):
            return float(spearmanr(entropy, y).statistic) if len(y) > 1 and np.ptp(entropy) and np.ptp(y) else float('nan')
        result.append(dict(phase=phase, scope=scope, name=name, samples=len(values),
                           mean_rank=float(rank.mean()), rho_rank=correlation(rank), rho_coefficient=correlation(coef)))
    write_csv(output / 'rank_correlations.csv', result)
    entropy, rank, coef = [np.array([float(r[key]) for r in rows]) for key in ('fft_entropy', 'active_rank', 'rank_coefficient')]
    for filename, x, y in [('rank_histogram.png', rank, None), ('rank_vs_fft_entropy.png', entropy, rank), ('rank_vs_rho.png', entropy, coef)]:
        fig, ax = plt.subplots()
        if y is None:
            ax.hist(x, bins=np.arange(int(rank.min()) - .5, int(rank.max()) + 1.5))
            ax.set(xlabel='Active rank', ylabel='Count')
        else:
            ax.scatter(x, y, s=4, alpha=.3)
            ax.set(xlabel='2D Fourier energy entropy', ylabel='Active rank' if 'entropy' in filename else 'Rank coefficient')
        fig.savefig(output / filename, dpi=160, bbox_inches='tight')
        plt.close(fig)
    for row in result:
        print(row)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)
