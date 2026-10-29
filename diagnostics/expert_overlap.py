import torch
import torch.nn.functional as F


@torch.no_grad()
def expert_overlap(model, epoch):
    rows = []
    for index, layer in enumerate(model.aco_layers):
        factor = F.normalize(layer.lora_A.index_select(0, layer.rank_order[:layer.current_rank]).flatten(1).float(), dim=1)
        masks = layer.get_expert_channel_masks(device=factor.device).bool()
        for first in range(layer.num_experts):
            for second in range(first + 1, layer.num_experts):
                a, b = factor[masks[first]], factor[masks[second]]
                if not a.numel() or not b.numel():
                    continue
                corr = (a @ b.T).abs()
                rows.append(dict(epoch=epoch, layer=f'decoder_{index}', variant=layer.grouping_mode,
                                 expert_i=first, expert_j=second,
                                 mean_abs_corr=float(corr.mean()), max_abs_corr=float(corr.max())))
    return rows
