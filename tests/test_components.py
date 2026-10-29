import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from TriEvoMoELoRA import InstanceRankModulator, ACOConvLoRA, ACOHyperparamOptimizer, ACOMoeGate
from reviewer_utils import (fixed_split, selection_score, SegmentationMetrics, load_legacy_checkpoint,
                            BufferedCSV, configure_model)
from diagnostics.analyze_layer_credit import remove_layer
from diagnostics.summarize_multiseed import summarize
from train import ACOTrainer, validate, parser

torch.set_num_threads(1)


class TinyDataset:
    def __init__(self, n):
        self.images = [f'image_{i}.png' for i in range(n)]
        self.gts = [f'mask_{i}.png' for i in range(n)]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return i


class RandomDataset:
    def __len__(self):
        return 8

    def __getitem__(self, i):
        import random
        return i, random.random(), np.random.random(), torch.rand(())


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aco_layers = torch.nn.ModuleList([ACOConvLoRA(3, 1, max_rank=16, num_experts=4)])
        self.aco_optimizer = SimpleNamespace()
        self.training_iteration = 0
    def set_epoch(self, epoch):
        self.aco_layers[0].set_epoch(epoch)
    def forward(self, x, update_hyperparams=False):
        return self.aco_layers[0](x)
    def get_expert_statistics(self):
        return {}
    def get_current_hyperparams(self):
        return dict(encoder=dict(rank=16, alpha=16), decoder=dict(rank=16, alpha=16, top_k=2))
    def collect_lora_fisher_scores(self):
        return {'decoder_0': self.aco_layers[0].get_lora_fisher_score()}
    def _apply_encoder_hyperparams(self, params):
        pass
    def set_deterministic_mode(self):
        pass


class ReviewerTests(unittest.TestCase):
    def test_seed_and_epoch_resume_reproducibility(self):
        import random
        from train import set_random_seed, seed_worker, capture_random_state, restore_random_state
        from torch.utils.data import DataLoader
        def loader(generator):
            return DataLoader(RandomDataset(), batch_size=2, shuffle=True, num_workers=2,
                              generator=generator, worker_init_fn=seed_worker, persistent_workers=False)
        def epoch(generator):
            return [torch.cat([value.double().reshape(-1) for value in batch]) for batch in loader(generator)]
        set_random_seed(3407)
        generator = torch.Generator().manual_seed(3407)
        first = epoch(generator)
        state = capture_random_state(generator)
        expected = (random.random(), np.random.random(), torch.rand(4), epoch(generator))
        set_random_seed(3407)
        repeated = epoch(torch.Generator().manual_seed(3407))
        for left, right in zip(first, repeated):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        restore_random_state(state, generator)
        resumed = (random.random(), np.random.random(), torch.rand(4), epoch(generator))
        self.assertEqual(expected[:2], resumed[:2])
        torch.testing.assert_close(expected[2], resumed[2], rtol=0, atol=0)
        for left, right in zip(expected[3], resumed[3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_metric_merge_matches_serial(self):
        serial, merged = SegmentationMetrics(.7), SegmentationMetrics(.7)
        rng = np.random.RandomState(3407)
        for size in (16, 23, 31):
            pred = rng.randint(0, 256, (size, size), dtype=np.uint8)
            gt = (rng.rand(size, size) > .5).astype(np.uint8) * 255
            serial.step(pred, gt)
            sample = SegmentationMetrics(.7)
            sample.step(pred, gt)
            merged.merge(sample)
        self.assertEqual(serial.results(), merged.results())

    def test_inference_seed_and_threaded_metrics_repeat(self):
        from unittest.mock import patch
        import test as inference
        from train import set_seed
        set_seed(3407)
        model = TinyModel()
        image = torch.randn(1, 3, 32, 32)
        mask = (np.arange(256).reshape(16, 16) > 100).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            torch.save(dict(model_state_dict=model.state_dict(), config={}, seed=3407,
                            current_hyperparams=model.get_current_hyperparams()), checkpoint)
            results = []
            for workers in (1, 4):
                output = root / str(workers)
                args = inference.parser.parse_args(['--dataset_name', 'tiny', '--checkpoint', str(checkpoint),
                    '--test_image_path', 'unused', '--test_gt_path', 'unused', '--save_path', str(output),
                    '--use_tta', '1', '--tta_scales', '1.0', '--metric_workers', str(workers)])
                samples = iter([(image.clone(), mask.copy(), f'{i}.png') for i in range(4)])
                loader = SimpleNamespace(size=4, load_data=lambda: next(samples))
                with patch('test.TriEvoMoELoRA', return_value=TinyModel()), \
                     patch('test.TestDataset', return_value=loader):
                    inference.main(args)
                results.append(json.loads((output / 'results.json').read_text(encoding='utf-8')))
                self.assertEqual(results[-1]['seed'], 3407)
            self.assertEqual(results[0]['metrics'], results[1]['metrics'])
            for i in range(4):
                self.assertEqual((root / '1' / f'{i}.png').read_bytes(), (root / '4' / f'{i}.png').read_bytes())

    def test_deterministic_mode_preserves_caller_seed(self):
        from TriEvoMoELoRA import TriEvoMoELoRA
        torch.manual_seed(3407)
        model = TinyModel()
        state = torch.get_rng_state()
        TriEvoMoELoRA.set_deterministic_mode(model)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(model.aco_layers[0].aco_gate.noisy_gating)

    def test_sparse_eval_matches_all_experts(self):
        from unittest.mock import patch
        layer = ACOConvLoRA(3, 1, max_rank=16, num_experts=8).eval()
        layer.routing_policy = 'consistent'
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            original_gate = layer.aco_gate.forward
            def all_experts(*args, **kwargs):
                result = original_gate(*args, **kwargs)
                layer.aco_gate.last_selected_experts = torch.arange(8).reshape(1, -1)
                return result
            with patch.object(layer.aco_gate, 'forward', side_effect=all_experts):
                expected, _ = layer(x)
            actual, _ = layer(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_auxiliary_schedule_uses_epochs_and_respects_small_weight(self):
        from TriEvoMoELoRA import AdaptiveTrainingScheduler
        for initial in (0.0, 0.05, 1.0):
            scheduler = AdaptiveTrainingScheduler(initial_aco_weight=initial)
            for _ in range(30):
                self.assertLessEqual(scheduler.step(.5, [.5] * 12), initial)
        model = TinyModel()
        trainer = ACOTrainer(model, torch.optim.AdamW(model.parameters()), amp='off')
        batch = dict(image=torch.randn(2, 3, 16, 16), label=torch.rand(2, 1, 16, 16))
        trainer.train_epoch([batch, batch, batch], 0, 2, torch.device('cpu'))
        self.assertEqual(trainer.aco_scheduler.epoch_count, 1)
        trainer.train_epoch([batch, batch], 1, 2, torch.device('cpu'))
        self.assertEqual(trainer.aco_scheduler.epoch_count, 2)

    def test_gate_train_eval_consistency(self):
        import copy
        gate = ACOMoeGate(3, 4, top_k=2, noisy_gating=False)
        with torch.no_grad():
            gate.w_gate.weight.zero_()
            gate.pheromone.copy_(torch.tensor([0.2, 0.5, 2.0, 4.0]))
            gate.expert_usage.copy_(torch.tensor([8., 6., 4., 2.]))
            gate.total_selections.fill_(10)
        inference = copy.deepcopy(gate).eval()
        state = {key: value.clone() for key, value in inference.state_dict().items()}
        x = torch.randn(2, 3, 8, 8)
        energy = torch.tensor([[4., 3., 2., 1.]]).expand(2, -1)
        train_gates, loss = gate(x, expert_energy=energy)
        eval_gates, _ = inference(x, expert_energy=energy)
        torch.testing.assert_close(train_gates, eval_gates)
        torch.testing.assert_close(inference(x, expert_energy=energy)[0], eval_gates)
        for key, value in inference.state_dict().items():
            torch.testing.assert_close(value, state[key])
        (train_gates[:, 3].sum() + loss).backward()
        self.assertTrue(torch.isfinite(gate.w_gate.weight.grad).all())

    def test_gate_checkpoint_policy(self):
        from reviewer_utils import restore_checkpoint_model
        model = TinyModel()
        checkpoint = dict(model_state_dict=model.state_dict(), config=dict(epoch=50))
        restore_checkpoint_model(model, checkpoint)
        gate = model.aco_layers[0].aco_gate.eval()
        self.assertEqual(gate.routing_policy, 'legacy_train_only')
        x = torch.randn(2, 3, 8, 8)
        before, _ = gate(x)
        with torch.no_grad():
            gate.pheromone.copy_(torch.arange(1, gate.num_experts + 1).float())
            gate.expert_usage.fill_(100)
            gate.total_selections.fill_(1)
        torch.testing.assert_close(gate(x)[0], before)
        checkpoint['config']['routing_policy'] = 'consistent'
        restore_checkpoint_model(model, checkpoint)
        self.assertEqual(model.aco_layers[0].aco_gate.routing_policy, 'consistent')

    def test_fft_rank(self):
        x = torch.randn(2, 3, 8, 8)
        energy = torch.fft.rfft2(x, norm='ortho').abs().square().mean(dim=1)
        p = energy / (energy.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        expected = -(p * (p + 1e-8).log()).sum(dim=(-2, -1)) / np.log(8 * 5)
        torch.testing.assert_close(InstanceRankModulator.compute_complexity_feature(x), expected)
        self.assertEqual(InstanceRankModulator().complexity_type, 'paper_rfft')
        for x in (torch.zeros(2, 3, 8, 8), torch.ones(2, 3, 8, 8), torch.randn(2, 3, 8, 8)):
            feature = InstanceRankModulator.compute_complexity_feature(x)
            self.assertEqual(feature.shape, (2,))
            self.assertTrue(torch.isfinite(feature).all())
            self.assertTrue(((feature >= 0) & (feature <= 1)).all())
        mod = InstanceRankModulator()
        for ceiling in (2, 4, 16):
            rank, _, _ = mod(torch.randn(3, 3, 8, 8), ceiling)
            self.assertTrue(((rank >= min(4, ceiling)) & (rank <= ceiling)).all())
            torch.testing.assert_close(rank, rank.round(), rtol=0, atol=0)

    def test_importance_and_decorrelation(self):
        layer = ACOConvLoRA(1, 1, kernel_size=1, max_rank=16, num_experts=2)
        with torch.no_grad():
            layer.lora_A.copy_(torch.arange(16).reshape(16, 1, 1, 1))
            layer.lora_B.fill_(1)
        self.assertEqual(layer.importance_type, 'paper_svd')
        layer.update_importance_order(force=True)
        self.assertEqual(layer.rank_order[0], 15)
        with torch.no_grad():
            layer.lora_B[:, 1] = 100
        layer.update_importance_order(force=True)
        self.assertEqual(layer.rank_order[0], 1)
        masks = torch.eye(2)
        high = layer._factor_decorrelation_loss(torch.ones(2, 2, 1, 1), masks)
        low = layer._factor_decorrelation_loss(torch.eye(2).reshape(2, 2, 1, 1), masks)
        self.assertGreater(float(high), float(low))
        a = torch.randn(8, 12)
        u, s, _ = torch.linalg.svd(a, full_matrices=False)
        torch.testing.assert_close((u.square() * s.square()[None]).sum(1).sqrt(), a.norm(dim=1))

    def test_split_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            train, val = fixed_split([TinyDataset(20)], [TinyDataset(20)], directory)
            train2, val2 = fixed_split([TinyDataset(20)], [TinyDataset(20)], directory)
            self.assertEqual(val.indices, val2.indices)
            self.assertFalse(set(train.indices) & set(val.indices))
            self.assertEqual(len(train) + len(val), 20)
            with self.assertRaises(ValueError):
                fixed_split([TinyDataset(21)], [TinyDataset(21)], directory)
        # A worse training metric cannot override a better validation score.
        best = selection_score({'mIoU': .8, 'train_iou': .1}, 'iou')
        self.assertGreater(best, selection_score({'mIoU': .7, 'train_iou': .99}, 'iou'))
        self.assertGreater(selection_score({'loss': .1}, 'loss'), selection_score({'loss': .2}, 'loss'))

    def test_metrics_and_legacy_pickle(self):
        meter = SegmentationMetrics()
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2:5, 2:5] = 255
        meter.step(image, image)
        self.assertEqual(meter.results()['mDice'], 1)
        self.assertEqual(meter.results()['mIoU'], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'model_state_dict': {'x': torch.ones(2)}}, path)
            self.assertTrue(torch.equal(load_legacy_checkpoint(path)['model_state_dict']['x'], torch.ones(2)))
            import sys
            import types
            module = types.ModuleType('ACOMoELoRA')
            module.ACOHyperparamOptimizer = ACOHyperparamOptimizer
            original = ACOHyperparamOptimizer.__module__
            sys.modules['ACOMoELoRA'] = module
            try:
                ACOHyperparamOptimizer.__module__ = 'ACOMoELoRA'
                torch.save({'aco_optimizer_state': ACOHyperparamOptimizer()}, path)
            finally:
                ACOHyperparamOptimizer.__module__ = original
                del sys.modules['ACOMoELoRA']
            self.assertIsInstance(load_legacy_checkpoint(path)['aco_optimizer_state'], ACOHyperparamOptimizer)

    def test_trainer_validation_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters())
            trainer = ACOTrainer(model, optimizer, amp='off')
            from diagnostics.rank_logger import RankLogger
            trainer.rank_logger = RankLogger(Path(directory) / 'rank.csv')
            trainer.fisher_logger = BufferedCSV(Path(directory) / 'fisher.csv')
            batch = dict(image=torch.randn(2, 3, 32, 32), label=torch.rand(2, 1, 32, 32),
                         sample_id=['a', 'b'], dataset=['tiny', 'tiny'])
            trainer.train_epoch([batch], 0, 1, torch.device('cpu'))
            trainer.record_epoch_metrics(0, 1, 1, 1, .5, .001, 1)
            trainer.export_training_metric_history(directory, plot=False)
            before = model.aco_layers[0].aco_gate.pheromone.clone()
            result = validate(model, [batch], torch.device('cpu'), 'off', rank_logger=trainer.rank_logger)
            self.assertTrue(all(np.isfinite(value) for value in result.values()))
            torch.testing.assert_close(before, model.aco_layers[0].aco_gate.pheromone)
            trainer.rank_logger.flush()
            trainer.fisher_logger.flush()
            self.assertTrue((Path(directory) / 'rank.csv').exists())
            with remove_layer(model, 'decoder_0'):
                self.assertFalse(model.aco_layers[0].adaptation_enabled)
            self.assertTrue(model.aco_layers[0].adaptation_enabled)


    def test_fisher_ema_before_normalization(self):
        optimizer = ACOHyperparamOptimizer()
        optimizer.set_epoch(0)
        optimizer.update_credit_signals({'encoder': 100.0, 'decoder_0': 1.0})
        optimizer.set_epoch(5)
        optimizer.update_credit_signals({'encoder': 1.0, 'decoder_0': 1.0})
        self.assertAlmostEqual(optimizer.fisher_ema_scores['encoder'], 90.1)
        self.assertAlmostEqual(optimizer.layer_credit_scores['encoder'], 90.1 / 91.1)

    def test_multiseed(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for seed, value in [(42, .6), (3407, .8), (114514, .7)]:
                path = Path(directory) / f'{seed}.json'
                path.write_text(json.dumps(dict(dataset='tiny', seed=seed, metrics={k: value for k in
                    ('S_alpha', 'F_beta_w', 'E_phi', 'MAE', 'mDice', 'mIoU')})), encoding='utf-8')
                files.append(path)
            rows = summarize(files, Path(directory) / 'summary.csv')
            self.assertAlmostEqual(rows[0]['mean'], .7)
            self.assertAlmostEqual(rows[0]['std'], .1)

    def test_profile_and_rank_plots(self):
        from diagnostics.profile_compute import profile
        from diagnostics.analyze_rank_complexity import analyze
        from diagnostics.rank_logger import RankLogger
        with tempfile.TemporaryDirectory() as directory:
            model = TinyModel()
            configure_model(model, dict())
            model.eval()
            batch = dict(image=torch.randn(2, 3, 8, 8), sample_id=['a', 'b'], dataset=['tiny', 'tiny'])
            with torch.no_grad():
                model(batch['image'])
            logger = RankLogger(Path(directory) / 'rank.csv')
            logger.record(model, batch, 1, 1, 'validation')
            logger.flush()
            analyze(logger.path, directory)
            self.assertTrue((Path(directory) / 'rank_histogram.png').exists())
            result = profile(model, size=8, amp='off')
            self.assertGreater(result['images_per_second'], 0)
            self.assertIsNone(result['FLOPs'])



    def test_eval_cli_and_prediction_exports(self):
        import subprocess
        import sys
        import csv
        from PIL import Image
        from unittest.mock import patch
        import test as inference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt = root / 'pred', root / 'gt'
            pred.mkdir()
            gt.mkdir()
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            Image.fromarray(mask).save(pred / 'a.png')
            Image.fromarray(mask).save(gt / 'a.png')
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'eval.py'),
                '--dataset_name', 'tiny', '--pred_path', str(pred), '--gt_path', str(gt), '--save_path', str(root / 'eval'),
                '--result_file', str(root / 'eval' / 'legacy.csv'), '--seed', '42'], capture_output=True, text=True,
                encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((root / 'eval' / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['metrics']['mDice'], 1)
            self.assertEqual(metrics['seed'], 42)
            with (root / 'eval' / 'legacy.csv').open(encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row['mDice']), metrics['metrics']['mDice'])
            self.assertEqual(float(row['mIoU']), metrics['metrics']['mIoU'])
            with (root / 'eval' / 'per_image_metrics.csv').open(encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            model = TinyModel()
            checkpoint = root / 'model.pt'
            torch.save(dict(model_state_dict=model.state_dict(), config={}, seed=42,
                            current_hyperparams=model.get_current_hyperparams()), checkpoint)
            args = inference.parser.parse_args(['--dataset_name', 'tiny', '--checkpoint', str(checkpoint),
                '--test_image_path', str(pred), '--test_gt_path', str(gt), '--save_path', str(root / 'test'), '--use_tta', '0'])
            loader = SimpleNamespace(size=1, load_data=lambda: (torch.randn(1, 3, 32, 32), mask, 'a.jpeg'))
            with patch('test.TriEvoMoELoRA', return_value=TinyModel()), patch('test.TestDataset', return_value=loader):
                inference.main(args)
            self.assertTrue((root / 'test' / 'a.png').exists())
            self.assertTrue((root / 'test' / 'test_metrics.csv').exists())

    def test_matched_rank_and_overlap(self):
        from diagnostics.expert_overlap import expert_overlap
        model = TinyModel().eval()
        layer = model.aco_layers[0]
        x = torch.randn(4, 3, 16, 16)
        with torch.no_grad():
            layer(x)
            ranks = layer.last_adaptation_stats['sample_ranks'].clone()
            layer.sarm_mode = 'random_rank'
            layer(x)
            torch.testing.assert_close(ranks.sort().values, layer.last_adaptation_stats['sample_ranks'].sort().values)
        self.assertTrue(all(np.isfinite(r['mean_abs_corr']) for r in expert_overlap(model, 1)))

    def test_eval_threshold_and_output_consistency(self):
        import subprocess
        import sys
        import csv
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt, output = root / 'pred', root / 'gt', root / 'eval'
            pred.mkdir()
            gt.mkdir()
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            Image.fromarray(mask).save(gt / 'a.png')
            Image.fromarray((mask > 0).astype(np.uint8) * 160).save(pred / 'a.png')
            (pred / 'results.json').write_text(json.dumps(dict(seed=42, protocol=dict(threshold=.7))), encoding='utf-8')
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'eval.py'),
                '--dataset_name', 'tiny', '--pred_path', str(pred), '--gt_path', str(gt),
                '--save_path', str(output), '--result_file', str(output / 'eval_metrics.csv')],
                capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            record = json.loads((output / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(record['metrics']['mDice'], 0.0)
            self.assertEqual(record['protocol']['threshold'], .7)
            with (output / 'eval_metrics.csv').open(encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            for column, key in [('mDice', 'mDice'), ('mIoU', 'mIoU'), ('Smeasure', 'S_alpha'),
                                ('wFmeasure', 'F_beta_w'), ('meanEm', 'E_phi'), ('MAE', 'MAE')]:
                self.assertEqual(float(row[column]), record['metrics'][key])

    def test_restore_early_sarm_checkpoint(self):
        from reviewer_utils import restore_checkpoint_model
        model = TinyModel()
        checkpoint = dict(model_state_dict=model.state_dict(), config=dict(epoch=50), epoch=1)
        restore_checkpoint_model(model, checkpoint)
        self.assertFalse(model.aco_layers[0].rank_modulator_active)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 unavailable')
    def test_cuda_bf16_training_and_validation(self):
        model = TinyModel().cuda()
        trainer = ACOTrainer(model, torch.optim.AdamW(model.parameters()), amp='bf16')
        batch = dict(image=torch.randn(2, 3, 32, 32), label=torch.rand(2, 1, 32, 32))
        values = trainer.train_epoch([batch], 1, 2, torch.device('cuda'))
        self.assertTrue(all(np.isfinite(v) for v in values))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        metrics = validate(model, [batch], torch.device('cuda'))
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_diagnostic_runner(self):
        from unittest.mock import patch
        import run_reviewer_diagnostics as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinyModel()
            checkpoint = root / 'model.pt'
            torch.save(dict(model_state_dict=model.state_dict(), config=dict(epoch=10, num_experts=4), epoch=5,
                            current_hyperparams=model.get_current_hyperparams()), checkpoint)
            class Dataset:
                images = [str(root / f'image{i}.png') for i in range(2)]
                gts = [str(root / f'mask{i}.png') for i in range(2)]
                def __len__(self):
                    return 2
                def __getitem__(self, i):
                    generator = torch.Generator().manual_seed(i)
                    return dict(image=torch.randn(3, 32, 32, generator=generator),
                                label=torch.rand(1, 32, 32, generator=generator),
                                sample_id=self.images[i], dataset='tiny')
            manifest = root / 'val.txt'
            manifest.write_text(''.join(json.dumps(pair) + '\n' for pair in zip(Dataset.images, Dataset.gts)), encoding='utf-8')
            args = SimpleNamespace(seed=42, checkpoint=str(checkpoint), output=str(root / 'diagnostic'),
                rank_analysis=True, layer_credit=True, expert_overlap=True, profile=False, image_path='unused',
                mask_path='unused', val_manifest=str(manifest), max_samples=2, batch_size=2, amp='off', dataset='tiny')
            with patch('run_reviewer_diagnostics.TriEvoMoELoRA', return_value=TinyModel()), \
                 patch('run_reviewer_diagnostics.FullDataset', return_value=Dataset()):
                runner.main(args)
            for name in ('rank_diagnostics.csv', 'expert_overlap.csv', 'layer_credit_diagnostic.csv', 'summary.md'):
                self.assertTrue((root / 'diagnostic' / name).exists())


    def test_main_protocol_and_training_iou_checkpoint(self):
        from unittest.mock import patch
        import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser.parse_args(['--hiera_path', 'unused', '--train_image_paths', 'unused',
                '--train_mask_paths', 'unused', '--save_path', str(root), '--epoch', '2',
                '--num_workers', '0', '--amp', 'off', '--training_protocol', 'aco_fulltrain',
                '--encoder_lr_scale', '0.1'])
            tiny_model = TinyModel()
            tiny_model.encoder = torch.nn.Linear(2, 2)
            def fake_train(trainer, loader, epoch, *rest):
                self.assertEqual(len(loader.dataset), 10)
                if epoch == 0:
                    self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], args.lr * .1)
                    self.assertAlmostEqual(trainer.optimizer.param_groups[1]['lr'], args.lr)
                    grouped = [p for group in trainer.optimizer.param_groups for p in group['params']]
                    self.assertEqual(len(grouped), len({id(p) for p in grouped}))
                    self.assertEqual({id(p) for p in grouped},
                                     {id(p) for p in tiny_model.parameters() if p.requires_grad})
                trainer.last_train_metrics = dict(mDice=.5, MAE=.5)
                trainer.optimizer.step()
                return 1, .2 if epoch == 0 else .8, 0, .1 if epoch == 0 else .9
            with patch('train.FullDataset', return_value=TinyDataset(10)), \
                 patch('train.TriEvoMoELoRA', return_value=tiny_model) as constructor, \
                 patch('train.validate') as validation, \
                 patch.object(ACOTrainer, 'train_epoch', fake_train), \
                 patch.object(ACOTrainer, '_plot_training_metric_history'):
                train.main(args)
                validation.assert_not_called()
                config = constructor.call_args.kwargs['aco_optimizer_config']
                self.assertEqual(config['rank_candidates'], [8, 16, 24, 32])
                self.assertEqual(config['alpha_candidates'], [8, 16, 24, 32, 48])
                self.assertEqual(config['top_k_candidates'], [1, 2, 3])
                self.assertEqual(config['beta'], 2.0)
            path = root / 'best.pth'
            checkpoint = load_legacy_checkpoint(path)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertIsNone(checkpoint['validation_metrics'])
            self.assertIsNone(checkpoint['split_manifest'])
            self.assertEqual(checkpoint['selection_source'], 'train_iou')
            self.assertTrue((root / "training_metrics.csv").is_file())
            self.assertEqual(checkpoint["config"]["aco_num_ants"], 24)


if __name__ == '__main__':
    unittest.main()
