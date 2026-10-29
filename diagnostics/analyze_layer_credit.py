from contextlib import contextmanager
import numpy as np
from scipy.stats import spearmanr, pearsonr
from reviewer_utils import write_csv


@contextmanager
def remove_layer(model, name):
    if name == 'encoder':
        from TriEvoMoELoRA import DynamicLoRALayer
        modules = [m for m in model.encoder.modules() if isinstance(m, DynamicLoRALayer)]
        previous = [m.current_rank for m in modules]
        for module in modules:
            module.current_rank = 0
        try:
            yield
        finally:
            for module, rank in zip(modules, previous):
                module.current_rank = rank
    else:
        layer = model.aco_layers[int(name.split('_')[-1])]
        previous = layer.adaptation_enabled
        layer.adaptation_enabled = False
        try:
            yield
        finally:
            layer.adaptation_enabled = previous


def analyze(model, evaluate, fisher, output):
    # evaluate uses the same fixed validation images and never updates routing.
    base = evaluate()['mIoU']
    rows = []
    for name, score in fisher.items():
        with remove_layer(model, name):
            perturbed = evaluate()['mIoU']
        rows.append(dict(layer=name, fisher_score=score, base_mIoU=base,
                         removed_mIoU=perturbed, empirical_delta=base - perturbed))
    write_csv(output, rows)
    x, y = np.array([r['fisher_score'] for r in rows]), np.array([r['empirical_delta'] for r in rows])
    valid = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    correlations = dict(Spearman=float(spearmanr(x, y).statistic) if valid else float('nan'),
                        Pearson=float(pearsonr(x, y).statistic) if valid else float('nan'))
    print('Layer-removal diagnostic:', correlations)
    return correlations
