import json
from reviewer_utils import BufferedCSV


class RankLogger(BufferedCSV):
    def record(self, model, batch, epoch, step, phase='train'):
        rows = []
        for index, layer in enumerate(model.aco_layers):
            stats = layer.last_adaptation_stats
            if not stats:
                continue
            values = {k: stats[k].float().cpu().tolist() for k in ('complexity', 'sample_ranks', 'rank_coefficients')}
            selected = layer.aco_gate.last_selected_experts.cpu().tolist()
            for sample, sample_id in enumerate(batch['sample_id']):
                rows.append(dict(epoch=epoch, step=step, phase=phase,
                                 sample_id=sample_id, dataset=batch['dataset'][sample],
                                 layer_id=f'decoder_{index}', fft_entropy=values['complexity'][sample],
                                 rank_coefficient=values['rank_coefficients'][sample],
                                 active_rank=values['sample_ranks'][sample], ceiling_rank=layer.current_rank,
                                 selected_expert_count=len(selected[sample]),
                                 selected_expert_ids=json.dumps(selected[sample])))
        self.add(rows)
