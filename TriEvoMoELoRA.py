import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import math
from typing import Optional, List, Tuple, Dict
import numpy as np
import random


class DoubleConv(nn.Module):
    """
    Standard double convolution block: (Conv2d => BN => ReLU) * 2.
    Used in the upsampling path of the U-Net-like architecture.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None,dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """
    Upscaling module followed by a double convolution.
    Uses bilinear interpolation for upsampling and handles padding for tensor alignment.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class ACOHyperparamOptimizer:
    """
    Ant Colony Optimization (ACO) algorithm for hyperparameter search.

    It searches for the optimal configuration of Rank and Alpha (for LoRA)
    and Top-K (for MoE) by simulating ant trails (pheromones) on the parameter space.
    """

    def __init__(self,
                 rank_candidates: List[int] = [8, 16, 24, 32],
                 alpha_candidates: List[float] = [8.0, 16.0, 24.0, 32.0, 48.0],
                 top_k_candidates: List[int] = [1, 2, 3],
                 num_ants: int = 24,
                 evaporation_rate: float = 0.22,
                 alpha: float = 1.5,
                 beta: float = 2,
                 q0: float = 0.7,
                 diversification_interval: int = 20,
                 mutation_prob: float = 0.12,
                 cost_penalty_weight: float = 0.003,
                 performance_window: int = 5,
                 total_epochs: int = 50,
                 num_decoder_layers: int = 4,
                 uncertainty_temperature_gamma: float = 0.5,
                 uncertainty_warmup_ratio: float = 0.2,
                 fisher_ema_decay: float = 0.9,
                 fisher_update_interval: int = 5,
                 **kwargs):

        self.rank_candidates = rank_candidates
        self.alpha_candidates = alpha_candidates
        self.top_k_candidates = top_k_candidates
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.diversification_interval = diversification_interval
        self.mutation_prob = mutation_prob

        # --- Pheromone Matrices ---
        # Encoder Pheromones
        self.enc_pheromone_rank = np.ones(len(rank_candidates))
        self.enc_pheromone_alpha = np.ones(len(alpha_candidates))

        # Decoder Pheromones
        self.dec_pheromone_rank = np.ones(len(rank_candidates))
        self.dec_pheromone_alpha = np.ones(len(alpha_candidates))
        self.dec_pheromone_topk = np.ones(len(top_k_candidates))

        # --- Tracking State ---
        self.history_performance = []
        self.best_solution = None
        self.best_performance = -float('inf')
        self.stagnation_count = 0
        self.iteration_count = 0
        self.epoch_count = 0

        # --- Diversity & Heuristics ---
        self.solution_diversity = []
        self.convergence_threshold = 0.1
        self.last_diversification_epoch = -diversification_interval

        self.performance_window = performance_window
        self.hyperparam_performances = {}
        self.cost_penalty_weight = cost_penalty_weight
        self.total_epochs = total_epochs
        self.num_decoder_layers = num_decoder_layers
        self.uncertainty_temperature_gamma = uncertainty_temperature_gamma
        self.uncertainty_warmup_ratio = uncertainty_warmup_ratio
        self.fisher_ema_decay = fisher_ema_decay
        self.fisher_update_interval = max(int(fisher_update_interval), 1)
        self.prediction_uncertainty = 0.0

        # UCGE-Gate: per-layer decoder pheromones for Fisher-decomposed credit.
        self.dec_layer_pheromone_rank = np.ones((num_decoder_layers, len(rank_candidates)))
        self.dec_layer_pheromone_alpha = np.ones((num_decoder_layers, len(alpha_candidates)))
        self.dec_layer_pheromone_topk = np.ones((num_decoder_layers, len(top_k_candidates)))
        self.layer_credit_scores = self._default_layer_credit_scores()
        self.fisher_ema_scores = None

    def set_epoch(self, epoch: int):
        self.epoch_count = epoch

    def _default_layer_credit_scores(self):
        credits = {'encoder': 1.0}
        for layer_idx in range(self.num_decoder_layers):
            credits[f'decoder_{layer_idx}'] = 1.0
        return self._normalize_layer_scores(credits)

    def _normalize_layer_scores(self, layer_scores=None):
        if not layer_scores:
            layer_scores = {'encoder': 1.0}
            for layer_idx in range(self.num_decoder_layers):
                layer_scores[f'decoder_{layer_idx}'] = 1.0

        cleaned = {}
        encoder_score = 0.0
        decoder_scores = [0.0] * self.num_decoder_layers

        for key, value in layer_scores.items():
            score = max(float(value), 0.0)
            if key == 'encoder' or key.startswith('encoder'):
                encoder_score += score
            elif key == 'decoder':
                for layer_idx in range(self.num_decoder_layers):
                    decoder_scores[layer_idx] += score / max(self.num_decoder_layers, 1)
            elif key.startswith('decoder_'):
                try:
                    layer_idx = int(key.split('_')[-1])
                except ValueError:
                    continue
                if 0 <= layer_idx < self.num_decoder_layers:
                    decoder_scores[layer_idx] += score

        cleaned['encoder'] = encoder_score
        for layer_idx, score in enumerate(decoder_scores):
            cleaned[f'decoder_{layer_idx}'] = score

        total = sum(cleaned.values())
        if total <= 0:
            return self._normalize_layer_scores(None)
        return {key: value / total for key, value in cleaned.items()}

    def update_credit_signals(self, layer_scores=None, prediction_uncertainty=None):
        """Updates UCGE-Gate training-only signals used by the ACO search."""
        if getattr(self, 'credit_mode', 'fisher_weighted_credit') != 'fisher_weighted_credit':
            # Global-only reward and uniform per-layer credit are equivalent here.
            self.layer_credit_scores = self._default_layer_credit_scores()
            layer_scores = None
        should_update_fisher = self.epoch_count % self.fisher_update_interval == 0
        if layer_scores is not None and should_update_fisher:
            raw_scores = {key: max(float(value), 0.0) for key, value in layer_scores.items()}
            if self.fisher_ema_scores is None:
                self.fisher_ema_scores = raw_scores
            else:
                self.fisher_ema_scores = {
                    key: (
                        self.fisher_ema_decay * self.fisher_ema_scores.get(key, 0.0)
                        + (1.0 - self.fisher_ema_decay) * raw_scores.get(key, 0.0)
                    )
                    for key in raw_scores
                }
            self.layer_credit_scores = self._normalize_layer_scores(self.fisher_ema_scores)
        if prediction_uncertainty is not None:
            self.prediction_uncertainty = float(np.clip(prediction_uncertainty, 0.0, 1.0))

    def _current_temperature(self, layer_name=None):
        cycle_progress = (self.epoch_count % 10) / 10.0
        base_temp = 0.5 + 1.5 * (0.5 * (1 + math.cos(math.pi * 2 * cycle_progress)))
        warmup_epochs = self.total_epochs * self.uncertainty_warmup_ratio
        if self.epoch_count < warmup_epochs:
            return base_temp
        temperature = base_temp * (
            1.0 + self.uncertainty_temperature_gamma * self.prediction_uncertainty
        )
        if layer_name is not None and self.layer_credit_scores:
            max_credit = max(self.layer_credit_scores.values()) + 1e-8
            relative_credit = self.layer_credit_scores.get(layer_name, 0.0) / max_credit
            temperature *= 0.5 + 0.5 * relative_credit
        return temperature

    def select_hyperparams(self, current_iteration: int) -> Tuple[Dict, Dict]:
        """Main entry point for generating hyperparameter candidates for the current iteration."""
        self.iteration_count = current_iteration

        # Check for Diversified Search Trigger
        epochs_since_last = self.epoch_count - getattr(self, 'last_diversification_epoch', -50)
        if epochs_since_last >= self.diversification_interval and random.random() < 0.3:
            self.last_diversification_epoch = self.epoch_count
            print(f"Epoch {self.epoch_count}: Triggering diversified search...")
            return self._diversification_search()

        solutions = []
        for ant in range(self.num_ants):

            # 1. Generate Encoder Solution
            encoder_solution = self._ant_search(
                self.enc_pheromone_rank,
                self.enc_pheromone_alpha,
                None,
                is_encoder=True,
                layer_name='encoder'
            )

            # 2. Generate Decoder Solutions layer by layer (UCGE-Gate)
            decoder_solution = [
                self._ant_search(
                    self.dec_layer_pheromone_rank[layer_idx],
                    self.dec_layer_pheromone_alpha[layer_idx],
                    self.dec_layer_pheromone_topk[layer_idx],
                    is_encoder=False,
                    layer_name=f'decoder_{layer_idx}'
                )
                for layer_idx in range(self.num_decoder_layers)
            ]

            solutions.append((encoder_solution, decoder_solution))

        # Select the best candidate from the ant population
        encoder_solution, decoder_solution = self._select_best_expected_solution(solutions)
        return self._solution_to_params(encoder_solution, decoder_solution)

    def _ant_search(self, pheromone_rank, pheromone_alpha, pheromone_topk,
                    is_encoder=False, layer_name=None):
        """Simulates a single ant constructing a solution path."""
        solution = []
        progress = self.epoch_count / self.total_epochs

        # Dynamic Beta Adjustment:
        # Start with low Beta (0.5) to encourage random exploration.
        # Increase Beta (up to 2.5) over time to rely more on heuristic/pheromone guidance.
        current_beta = self.beta if getattr(self, 'paper_fixed_beta', False) else 0.5 + 2.0 * progress

        # 1. Rank Selection
        # Early training phase encourages randomness
        if progress < 0.3 and random.random() < 0.3:
            rank_idx = random.randint(0, len(self.rank_candidates) - 1)
        else:
            rank_idx = self._enhanced_selection(
                pheromone_rank, np.ones_like(pheromone_rank),
                beta=current_beta, layer_name=layer_name
            )

        selected_rank = self.rank_candidates[rank_idx]
        solution.append(rank_idx)

        # 2. Alpha Selection
        # Mutation probability decreases as training progresses
        mutation_prob = 0.3 if is_encoder else 0.2
        current_mutation = max(0.15, mutation_prob * (1 - progress))

        if random.random() < current_mutation:
            alpha_idx = random.randint(0, len(self.alpha_candidates) - 1)
        else:
            heuristic_alpha = np.ones_like(pheromone_alpha)
            target_alpha = selected_rank * 2.0

            for i, val in enumerate(self.alpha_candidates):
                dist = abs(val - target_alpha)
                sigma = target_alpha * 3.0
                score = np.exp(-0.5 * (dist / sigma) ** 2)
                heuristic_alpha[i] = 0.5 + 1.0 * score

            # Inject noise into heuristic
            heuristic_alpha *= np.random.uniform(0.6, 1.4, size=len(heuristic_alpha))
            alpha_idx = self._enhanced_selection(
                pheromone_alpha, heuristic_alpha,
                beta=current_beta, layer_name=layer_name
            )

        solution.append(alpha_idx)

        # 3. Top-K Selection (Decoder Only)
        if pheromone_topk is not None:
            heuristic_topk = np.array([1.0 + 2.0 / k for k in self.top_k_candidates])
            noise = np.random.uniform(0.8, 1.2, size=len(heuristic_topk))
            heuristic_topk *= noise

            if progress < 0.3 and random.random() < 0.3:
                topk_idx = random.randint(0, len(self.top_k_candidates) - 1)
            else:
                topk_idx = self._enhanced_selection(
                    pheromone_topk, heuristic_topk,
                    beta=current_beta, layer_name=layer_name
                )
            solution.append(topk_idx)
        else:
            solution.append(-1)

        return solution

    def _enhanced_selection(self, pheromone, heuristic, beta=None, layer_name=None):
        """Probabilistic selection based on Pheromone^Alpha * Heuristic^Beta."""
        if beta is None:
            beta = self.beta

        progress = self.epoch_count / self.total_epochs

        # Temperature Control for Softmax
        # Oscillates to periodically encourage exploration
        current_temp = self._current_temperature(layer_name=layer_name)
        epsilon = 0.15

        # Calculate probability distribution
        probs = (pheromone ** self.alpha) * (heuristic ** beta)
        probs = probs ** (1.0 / current_temp)

        if np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / np.sum(probs)

        # Epsilon-greedy mixing
        n_actions = len(probs)
        probs = (1 - epsilon) * probs + (epsilon / n_actions)
        probs = probs / np.sum(probs)

        if progress > 0.98:
            return np.argmax(probs)

        return np.random.choice(len(probs), p=probs)

    def _diversification_search(self):
        """Executes a search aimed at escaping local optima."""
        # Strategy 1: Pure Random Search (30% chance or if no best solution exists)
        if self.best_solution is None or random.random() < 0.3:
            enc_sol = [
                random.randint(0, len(self.rank_candidates) - 1),
                random.randint(0, len(self.alpha_candidates) - 1),
                -1
            ]
            dec_sol = [
                random.randint(0, len(self.rank_candidates) - 1),
                random.randint(0, len(self.alpha_candidates) - 1),
                random.randint(0, len(self.top_k_candidates) - 1)
            ]
            dec_sol = [dec_sol[:] for _ in range(self.num_decoder_layers)]
            return self._solution_to_params(enc_sol, dec_sol)

        # Strategy 2: Local Perturbation around Best Known Solution
        else:
            best_enc, best_dec = self.best_solution

            try:
                e_r_idx = self.rank_candidates.index(best_enc['rank'])
                e_a_idx = self.alpha_candidates.index(best_enc['alpha'])

                best_dec_layer = self._get_decoder_layers(best_dec)[0]
                d_r_idx = self.rank_candidates.index(best_dec_layer['rank'])
                d_a_idx = self.alpha_candidates.index(best_dec_layer['alpha'])
                d_k_idx = self.top_k_candidates.index(best_dec_layer['top_k'])

                enc_sol = [
                    self._perturb_index(e_r_idx, len(self.rank_candidates)),
                    self._perturb_index(e_a_idx, len(self.alpha_candidates)),
                    -1
                ]

                dec_sol = [
                    self._perturb_index(d_r_idx, len(self.rank_candidates)),
                    self._perturb_index(d_a_idx, len(self.alpha_candidates)),
                    self._perturb_index(d_k_idx, len(self.top_k_candidates))
                ]
                dec_sol = [dec_sol[:] for _ in range(self.num_decoder_layers)]

                return self._solution_to_params(enc_sol, dec_sol)

            except ValueError:
                # If an index error occurs, fall back to random.
                return self._diversification_search()

    def _perturb_index(self, current_idx, max_len):
        """Applies a small step perturbation (-1, 0, +1) to an index."""
        step = random.choice([-1, 0, 1])
        new_idx = current_idx + step
        return max(0, min(new_idx, max_len - 1))

    def _low_pheromone_search(self, pheromone_rank, pheromone_alpha, pheromone_topk):
        """Search biased toward low pheromone areas"""
        solution = []

        # Select rank with lower pheromone levels
        rank_probs = 1.0 / (pheromone_rank + 1e-8)
        rank_probs = rank_probs / np.sum(rank_probs)
        solution.append(np.random.choice(len(rank_probs), p=rank_probs))

        # Select alphas with lower pheromone levels
        alpha_probs = 1.0 / (pheromone_alpha + 1e-8)
        alpha_probs = alpha_probs / np.sum(alpha_probs)
        solution.append(np.random.choice(len(alpha_probs), p=alpha_probs))

        # top_k Selection
        if pheromone_topk is not None:
            topk_probs = 1.0 / (pheromone_topk + 1e-8)
            topk_probs = topk_probs / np.sum(topk_probs)
            solution.append(np.random.choice(len(topk_probs), p=topk_probs))
        else:
            solution.append(-1)

        return solution

    def _calculate_diversity(self, solutions):
        """Calculates population diversity to monitor convergence."""
        if len(solutions) <= 1:
            return 1.0

        encoded_solutions = []
        for enc_sol, dec_sol in solutions:
            encoded = []
            flat_dec_sol = []
            if dec_sol and isinstance(dec_sol[0], (list, tuple)):
                for layer_sol in dec_sol:
                    flat_dec_sol.extend(layer_sol)
            else:
                flat_dec_sol = dec_sol
            for val in enc_sol + flat_dec_sol:
                if val is None:
                    encoded.append(-1)
                else:
                    encoded.append(val)
            encoded_solutions.append(encoded)

        encoded_solutions = np.array(encoded_solutions, dtype=np.float32)

        total_distance = 0
        count = 0
        for i in range(len(encoded_solutions)):
            for j in range(i + 1, len(encoded_solutions)):
                total_distance += np.linalg.norm(encoded_solutions[i] - encoded_solutions[j])
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        # Normalize by approximate max distance
        max_possible_distance = np.sqrt(len(encoded_solutions[0]) * (max(len(self.rank_candidates),
                                                                         len(self.alpha_candidates),
                                                                         len(self.top_k_candidates)) ** 2))

        return min(avg_distance / max_possible_distance, 1.0)

    def _select_best_expected_solution(self, solutions):
        """Selects a solution from the ant population weighted by pheromone levels."""
        if not solutions:
            return ([0, 0, -1], [[0, 0, 0] for _ in range(self.num_decoder_layers)])

        solution_scores = []
        for enc_sol, dec_sol in solutions:
            score = 0
            # Encoder Score
            score += self.enc_pheromone_rank[enc_sol[0]] * self.alpha
            score += self.enc_pheromone_alpha[enc_sol[1]] * self.beta

            # Decoder Score
            decoder_solutions = dec_sol if dec_sol and isinstance(dec_sol[0], (list, tuple)) else [dec_sol]
            for layer_idx, layer_sol in enumerate(decoder_solutions):
                layer_idx = min(layer_idx, self.num_decoder_layers - 1)
                layer_credit = self.layer_credit_scores.get(f'decoder_{layer_idx}', 1.0)
                score += layer_credit * self.dec_layer_pheromone_rank[layer_idx, layer_sol[0]] * self.alpha
                score += layer_credit * self.dec_layer_pheromone_alpha[layer_idx, layer_sol[1]] * self.beta
                if layer_sol[2] != -1:
                    score += layer_credit * self.dec_layer_pheromone_topk[layer_idx, layer_sol[2]] * self.beta

            solution_scores.append(score)

        scores = np.array(solution_scores)
        # Prevent numerical overflow (Softmax stability)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        selected_idx = np.random.choice(len(solutions), p=probs)
        return solutions[selected_idx]

    def _select_based_on_performance_history(self, solutions):
        """Select the optimal solution based on historical performance data"""
        best_idx = None
        best_expected_performance = -float('inf')

        for i, (enc_sol, dec_sol) in enumerate(solutions):
            hyperparam_key = self._get_hyperparam_key_from_solution(enc_sol, dec_sol)

            if hyperparam_key in self.hyperparam_performances:
                # Calculate expected performance
                performances = self.hyperparam_performances[hyperparam_key]
                avg_performance = np.mean(performances)

                # Consider performance stability
                if len(performances) > 1:
                    std_performance = np.std(performances)
                    stability_factor = 1.0 / (1.0 + std_performance)
                    expected_performance = avg_performance * stability_factor
                else:
                    expected_performance = avg_performance

                # Consider usage frequency and avoid overusing the same combination
                frequency = len(performances)
                # The more times it is used, the smaller the exploration factor becomes.
                exploration_factor = max(0.5, 1.0 - (frequency * 0.1))

                final_score = expected_performance * exploration_factor

                if final_score > best_expected_performance:
                    best_expected_performance = final_score
                    best_idx = i

        return best_idx

    def _get_hyperparam_key_from_solution(self, encoder_solution, decoder_solution):
        """Generates a string key for the solution for caching/hashing."""
        rank_idx = encoder_solution[0]
        alpha_idx = encoder_solution[1]
        encoder_rank = self.rank_candidates[rank_idx]
        encoder_alpha = self.alpha_candidates[alpha_idx]

        decoder_solutions = (
            decoder_solution
            if decoder_solution and isinstance(decoder_solution[0], (list, tuple))
            else [decoder_solution]
        )
        dec_parts = []
        for layer_idx, layer_solution in enumerate(decoder_solutions):
            dec_rank_idx = layer_solution[0]
            dec_alpha_idx = layer_solution[1]
            dec_topk_idx = layer_solution[2]
            decoder_rank = self.rank_candidates[dec_rank_idx]
            decoder_alpha = self.alpha_candidates[dec_alpha_idx]
            decoder_topk = self.top_k_candidates[dec_topk_idx] if dec_topk_idx != -1 else 2
            dec_parts.append(f"d{layer_idx}_r{decoder_rank}_a{decoder_alpha}_k{decoder_topk}")

        return f"enc_r{encoder_rank}_a{encoder_alpha}_" + "_".join(dec_parts)

    def _solution_to_params(self, encoder_solution, decoder_solution):
        """Converts indices back to actual hyperparameter values."""
        enc_rank_idx = encoder_solution[0]
        enc_alpha_idx = encoder_solution[1]

        encoder_params = {
            'rank': self.rank_candidates[enc_rank_idx],
            'alpha': self.alpha_candidates[enc_alpha_idx]
        }

        decoder_solutions = (
            decoder_solution
            if decoder_solution and isinstance(decoder_solution[0], (list, tuple))
            else [decoder_solution for _ in range(self.num_decoder_layers)]
        )
        layer_params = []
        for layer_solution in decoder_solutions:
            dec_rank_idx = layer_solution[0]
            dec_alpha_idx = layer_solution[1]
            dec_topk_idx = layer_solution[2]
            layer_params.append({
                'rank': self.rank_candidates[dec_rank_idx],
                'alpha': self.alpha_candidates[dec_alpha_idx],
                'top_k': self.top_k_candidates[dec_topk_idx]
            })

        first_layer = layer_params[0]

        decoder_params = {
            'rank': first_layer['rank'],
            'alpha': first_layer['alpha'],
            'top_k': first_layer['top_k'],
            'layers': layer_params
        }

        return encoder_params, decoder_params

    def _get_decoder_layers(self, decoder_params):
        layers = decoder_params.get('layers') if isinstance(decoder_params, dict) else None
        if layers:
            return layers
        return [decoder_params for _ in range(self.num_decoder_layers)]

    def _calculate_fitness(self, iou, encoder_params, decoder_params):
        """
        Calculates fitness score: IOU minus a penalty for model complexity.
        Fitness = IOU - (Penalty_Weight * Complexity)
        """
        max_rank = max(self.rank_candidates)
        max_topk = max(self.top_k_candidates)

        # 1. Normalize costs
        enc_cost = encoder_params['rank'] / max_rank
        decoder_layers = self._get_decoder_layers(decoder_params)
        dec_rank_norm = np.mean([layer['rank'] / max_rank for layer in decoder_layers])
        dec_topk_norm = np.mean([(layer['top_k'] / max_topk) ** 1.5 for layer in decoder_layers])

        # 2. Weighted Complexity
        dec_cost = 0.5 * dec_rank_norm + 0.5 * dec_topk_norm
        total_complexity = 0.45 * enc_cost + 0.5 * dec_cost

        # 3. Dynamic Penalty Weighting
        # Penalty increases as training stabilizes (Warmup strategy)
        base_penalty = 0.0005
        target_penalty = 0.002
        warmup_epochs = self.total_epochs * 0.4

        if self.epoch_count < warmup_epochs:
            penalty_weight = base_penalty
        else:
            ratio = (self.epoch_count - warmup_epochs) / (self.total_epochs - warmup_epochs)
            penalty_weight = base_penalty + (target_penalty - base_penalty) * ratio

        penalty = penalty_weight * total_complexity
        return iou - penalty

    def update_pheromone(self, solutions_performance: List[Tuple[Dict, float]],
                         layer_scores: Dict[str, float] = None,
                         prediction_uncertainty: float = None):
        """Updates the pheromone trails based on the performance (IoU)."""
        if not solutions_performance:
            return
        self.update_credit_signals(layer_scores, prediction_uncertainty)

        # 1. Pheromone Evaporation (Global)
        self.enc_pheromone_rank *= (1 - self.evaporation_rate)
        self.enc_pheromone_alpha *= (1 - self.evaporation_rate)
        self.dec_pheromone_rank *= (1 - self.evaporation_rate)
        self.dec_pheromone_alpha *= (1 - self.evaporation_rate)
        self.dec_pheromone_topk *= (1 - self.evaporation_rate)
        self.dec_layer_pheromone_rank *= (1 - self.evaporation_rate)
        self.dec_layer_pheromone_alpha *= (1 - self.evaporation_rate)
        self.dec_layer_pheromone_topk *= (1 - self.evaporation_rate)

        # Archive performance
        for (enc_params, dec_params), raw_iou in solutions_performance:
            key = self._get_hyperparam_key(enc_params, dec_params)
            if key not in self.hyperparam_performances:
                self.hyperparam_performances[key] = []
            self.hyperparam_performances[key].append(raw_iou)

        # 2. Calculate Fitness & Rank Solutions
        scored_solutions = []
        for (enc_params, dec_params), raw_iou in solutions_performance:
            key = self._get_hyperparam_key(enc_params, dec_params)
            avg_iou = np.mean(self.hyperparam_performances[key])
            fitness = self._calculate_fitness(avg_iou, enc_params, dec_params)
            scored_solutions.append({'params': (enc_params, dec_params), 'fitness': fitness, 'raw_iou': avg_iou})

        sorted_solutions = sorted(scored_solutions, key=lambda x: x['fitness'], reverse=True)

        # Update Best Global Solution
        if sorted_solutions[0]['raw_iou'] > self.best_performance:
            self.best_performance = sorted_solutions[0]['raw_iou']
            self.best_solution = sorted_solutions[0]['params']
            self.stagnation_count = 0
            print(f"Epoch {self.epoch_count}: Performance Improved: {self.best_performance:.4f}")
        else:
            self.stagnation_count += 1

        # 3. Deposit Pheromone (Differential Update)
        # Better solutions deposit more pheromone
        fits = [s['fitness'] for s in sorted_solutions]
        min_f, max_f = min(fits), max(fits)
        fit_range = max_f - min_f + 1e-9

        # Only update for the top 50% solutions
        top_count = max(1, len(sorted_solutions) // 2)
        for item in sorted_solutions[:top_count]:
            (enc_params, dec_params) = item['params']
            delta = ((item['fitness'] - min_f) / fit_range) * 2.0
            self._deposit_encoder_pheromone(enc_params, delta)
            self._deposit_decoder_pheromone(dec_params, delta)

        # 4. Elitist Strategy
        # The global best solution always deposits pheromone
        progress = self.epoch_count / self.total_epochs
        elitist_weight = 0.5 + 2.5 * progress

        if self.best_solution:
            enc_best, dec_best = self.best_solution
            self._deposit_encoder_pheromone(enc_best, elitist_weight)
            self._deposit_decoder_pheromone(dec_best, elitist_weight)

        # 5. Pheromone Clamping (Max-Min Ant System concept)
        max_val = 20.0
        min_val = 10.0 * (1 - progress) + 0.01

        for pheromone in [self.enc_pheromone_rank, self.enc_pheromone_alpha,
                          self.dec_pheromone_rank, self.dec_pheromone_alpha, self.dec_pheromone_topk,
                          self.dec_layer_pheromone_rank, self.dec_layer_pheromone_alpha,
                          self.dec_layer_pheromone_topk]:
            np.clip(pheromone, min_val, max_val, out=pheromone)

    def _deposit_encoder_pheromone(self, params, delta):
        r_idx = self.rank_candidates.index(params['rank'])
        a_idx = self.alpha_candidates.index(params['alpha'])
        credit = self.layer_credit_scores.get('encoder', 1.0)
        self.enc_pheromone_rank[r_idx] += delta * credit
        self.enc_pheromone_alpha[a_idx] += delta * credit

    def _deposit_decoder_pheromone(self, params, delta):
        for layer_idx, layer_params in enumerate(self._get_decoder_layers(params)):
            if layer_idx >= self.num_decoder_layers:
                break
            credit = self.layer_credit_scores.get(f'decoder_{layer_idx}', 1.0)
            r_idx = self.rank_candidates.index(layer_params['rank'])
            a_idx = self.alpha_candidates.index(layer_params['alpha'])
            k_idx = self.top_k_candidates.index(layer_params['top_k'])
            self.dec_layer_pheromone_rank[layer_idx, r_idx] += delta * credit
            self.dec_layer_pheromone_alpha[layer_idx, a_idx] += delta * credit
            self.dec_layer_pheromone_topk[layer_idx, k_idx] += delta * credit

        self.dec_pheromone_rank = self.dec_layer_pheromone_rank.mean(axis=0)
        self.dec_pheromone_alpha = self.dec_layer_pheromone_alpha.mean(axis=0)
        self.dec_pheromone_topk = self.dec_layer_pheromone_topk.mean(axis=0)


    def _get_hyperparam_key(self, encoder_params, decoder_params):
        """Generates unique key for caching performance history."""
        enc_str = f"e_r{encoder_params['rank']}_a{encoder_params['alpha']}"
        dec_parts = []
        for layer_idx, layer_params in enumerate(self._get_decoder_layers(decoder_params)):
            dec_parts.append(
                f"d{layer_idx}_r{layer_params['rank']}_a{layer_params['alpha']}_k{layer_params['top_k']}"
            )
        dec_str = "_".join(dec_parts)
        return f"{enc_str}_{dec_str}"

    def get_hyperparam_performance_stats(self):
        """Returns statistics of the optimization process for logging."""
        stats = {}
        for hyperparam_key, performances in self.hyperparam_performances.items():
            if performances:
                stats[hyperparam_key] = {
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'count': len(performances),
                    'recent': performances[-5:]
                }
        return stats

    def _soft_reset_pheromone(self):
        """Soft Reset Pheromone"""
        print(f"Epoch {self.epoch_count}: Detected stagnation, executing soft reset pheromone...")
        def smooth(matrix):
            return (matrix + np.ones_like(matrix)) / 2.0

        self.enc_pheromone_rank = smooth(self.enc_pheromone_rank)
        self.enc_pheromone_alpha = smooth(self.enc_pheromone_alpha)

        self.dec_pheromone_rank = smooth(self.dec_pheromone_rank)
        self.dec_pheromone_alpha = smooth(self.dec_pheromone_alpha)
        self.dec_pheromone_topk = smooth(self.dec_pheromone_topk)
        self.dec_layer_pheromone_rank = smooth(self.dec_layer_pheromone_rank)
        self.dec_layer_pheromone_alpha = smooth(self.dec_layer_pheromone_alpha)
        self.dec_layer_pheromone_topk = smooth(self.dec_layer_pheromone_topk)

    def _hard_reset_pheromone(self):
        """Hard Reset Pheromone (Restart)"""
        print(f"Epoch {self.epoch_count}: Perform a hard reset (Restart)...")

        self.enc_pheromone_rank = np.ones(len(self.rank_candidates))
        self.enc_pheromone_alpha = np.ones(len(self.alpha_candidates))

        self.dec_pheromone_rank = np.ones(len(self.rank_candidates))
        self.dec_pheromone_alpha = np.ones(len(self.alpha_candidates))
        self.dec_pheromone_topk = np.ones(len(self.top_k_candidates))
        self.dec_layer_pheromone_rank = np.ones((self.num_decoder_layers, len(self.rank_candidates)))
        self.dec_layer_pheromone_alpha = np.ones((self.num_decoder_layers, len(self.alpha_candidates)))
        self.dec_layer_pheromone_topk = np.ones((self.num_decoder_layers, len(self.top_k_candidates)))

        self.stagnation_count = 0

    def get_best_hyperparams(self) -> Tuple[Dict, Dict]:
        """Obtain optimal hyperparameters"""
        if self.best_solution is None:
            return {'rank': 16, 'alpha': 1.0}, {'rank': 16, 'alpha': 1.0, 'top_k': 2}
        return self.best_solution

    def get_optimizer_stats(self):
        """Report the pheromone intensity for the Encoder and Decoder respectively"""
        diversity = self.solution_diversity[-1] if hasattr(self,'solution_diversity') and self.solution_diversity else 0.0

        stats = {
            'best_performance': self.best_performance,
            'stagnation_count': self.stagnation_count,
            'iteration_count': self.iteration_count,
            'epoch_count': self.epoch_count,
            'current_diversity': diversity,
            # === Encoder ===
            'enc_pheromone_rank_mean': np.mean(self.enc_pheromone_rank),
            'enc_pheromone_alpha_mean': np.mean(self.enc_pheromone_alpha),
            # === Decoder ===
            'dec_pheromone_rank_mean': np.mean(self.dec_pheromone_rank),
            'dec_pheromone_alpha_mean': np.mean(self.dec_pheromone_alpha),
            'dec_pheromone_topk_mean': np.mean(self.dec_pheromone_topk),
            'prediction_uncertainty': self.prediction_uncertainty,
            'layer_credit_scores': dict(self.layer_credit_scores)
        }

        return stats

    def get_layer_pheromone_snapshot(self):
        """Exports encoder and per-decoder-layer pheromones for logging/visualization."""
        snapshot = {
            'encoder': {
                'rank': self.enc_pheromone_rank.tolist(),
                'alpha': self.enc_pheromone_alpha.tolist(),
                'credit': float(self.layer_credit_scores.get('encoder', 0.0))
            }
        }
        for layer_idx in range(self.num_decoder_layers):
            snapshot[f'decoder_{layer_idx}'] = {
                'rank': self.dec_layer_pheromone_rank[layer_idx].tolist(),
                'alpha': self.dec_layer_pheromone_alpha[layer_idx].tolist(),
                'top_k': self.dec_layer_pheromone_topk[layer_idx].tolist(),
                'credit': float(self.layer_credit_scores.get(f'decoder_{layer_idx}', 0.0))
            }
        return snapshot

class MultiStartACOOptimizer:
    """
    Multi-start ACO Optimizer to Avoid Local Optima
    """

    def __init__(self, num_starts=3, **aco_kwargs):
        self.num_starts = num_starts
        self.optimizers = [ACOHyperparamOptimizer(**aco_kwargs) for _ in range(num_starts)]
        self.total_epochs = self.optimizers[0].total_epochs
        self.current_optimizer_idx = 0
        self.optimizer_performance = [0] * num_starts
        self.epoch_count = 0

    def set_epoch(self, epoch: int):
        """Set the current epoch"""
        self.epoch_count = epoch
        for optimizer in self.optimizers:
            optimizer.set_epoch(epoch)

    def select_hyperparams(self, current_iteration: int) -> Tuple[Dict, Dict]:
        return self.optimizers[self.current_optimizer_idx].select_hyperparams(current_iteration)

    def update_pheromone(self, solutions_performance: List[Tuple[Dict, float]],
                         layer_scores: Dict[str, float] = None,
                         prediction_uncertainty: float = None):
        self.optimizers[self.current_optimizer_idx].update_pheromone(
            solutions_performance,
            layer_scores=layer_scores,
            prediction_uncertainty=prediction_uncertainty
        )

        # Update the performance records of the current optimizer
        if solutions_performance:
            best_perf = max(solutions_performance, key=lambda x: x[1])[1]
            self.optimizer_performance[self.current_optimizer_idx] = best_perf

            # Periodically switch to a more efficient optimizer
            if self.epoch_count % 100 == 0:
                best_idx = np.argmax(self.optimizer_performance)
                if best_idx != self.current_optimizer_idx:
                    print(f"Epoch {self.epoch_count}: Switch to the optimizer {best_idx}")
                    self.current_optimizer_idx = best_idx

    def get_best_hyperparams(self) -> Tuple[Dict, Dict]:
        """Select the optimal solution from all optimizers"""
        best_performance = -float('inf')
        best_solution = None

        for optimizer in self.optimizers:
            if optimizer.best_performance > best_performance:
                best_performance = optimizer.best_performance
                best_solution = optimizer.best_solution

        return best_solution if best_solution else ({'rank': 16, 'alpha': 1.0}, {'rank': 16, 'alpha': 1.0, 'top_k': 2})

    def get_layer_pheromone_snapshot(self):
        return self.optimizers[self.current_optimizer_idx].get_layer_pheromone_snapshot()


class DynamicLoRALayer(nn.Module):
    """
    Dynamic Low-Rank Adaptation (LoRA) Layer.
    Allows changing rank and alpha during runtime without re-instantiating weights.
    """

    def __init__(self, in_dim, out_dim, max_rank=32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_rank = max_rank

        # Pre-allocate max parameters to support dynamic rank adjustments without reallocation.
        self.lora_A = nn.Parameter(torch.zeros(max_rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, max_rank))

        # Kaiming Init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.current_rank = 16
        self.current_alpha = 16.0

    def set_hyperparams(self, rank: int, alpha: float):
        self.current_rank = min(rank, self.max_rank)
        self.current_alpha = alpha

    def forward(self, x):
        if self.current_rank == 0:
            return 0

        # Slicing: Use only the required subset of parameters
        lora_A = self.lora_A[:self.current_rank]
        lora_B = self.lora_B[:, :self.current_rank]

        lora_update = F.linear(F.linear(x, lora_A), lora_B)
        scaling = self.current_alpha / self.current_rank
        return scaling * lora_update


class DynamicLoRAWrapper(nn.Module):
    """
    Dynamic LoRA Wrapper
    """

    def __init__(self, layer, max_rank=32):
        super().__init__()
        self.layer = layer
        self.lora = DynamicLoRALayer(
            layer.in_features, layer.out_features, max_rank
        )

    def set_hyperparams(self, rank: int, alpha: float):
        self.lora.set_hyperparams(rank, alpha)

    def forward(self, x):
        return self.layer(x) + self.lora(x)


class InstanceRankModulator(nn.Module):
    """Predicts sample rank from 2D Fourier energy entropy, not semantic difficulty."""

    def __init__(self, hidden_dim=32, min_rank=4):
        super().__init__()
        self.min_rank = min_rank
        self.complexity_type = 'paper_rfft'
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def compute_complexity_feature(x, complexity_type='paper_rfft'):
        """2D Fourier energy entropy; not singular-value entropy or semantic difficulty.

        Default: channel-averaged rFFT power entropy, following supplement Eq. 8.
        fft_energy_entropy preserves the joint channel-frequency variant.
        """
        if complexity_type == 'fft_energy_entropy':
            energy = torch.fft.fft2(x.float(), dim=(-2, -1), norm='ortho').abs().square().flatten(1)
            probabilities = energy / (energy.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(probabilities * (probabilities + 1e-8).log()).sum(dim=1)
            return (entropy / max(math.log(max(energy.shape[1], 2)), 1e-8)).clamp(0, 1).to(x.dtype)
        spectrum = torch.fft.rfft2(x.float(), norm='ortho')
        energy = spectrum.abs().pow(2).mean(dim=1)
        probabilities = energy / (energy.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        entropy = -(probabilities * (probabilities + 1e-8).log()).sum(dim=(-2, -1))
        max_entropy = math.log(max(energy.shape[-2] * energy.shape[-1], 2))
        return (entropy / max_entropy).clamp(0.0, 1.0).to(dtype=x.dtype)

    def forward(self, x, rank_upper):
        complexity = self.compute_complexity_feature(x, self.complexity_type)
        coefficients = self.mlp(complexity.unsqueeze(-1)).squeeze(-1)
        min_rank = min(self.min_rank, rank_upper)
        continuous_rank = min_rank + coefficients * (rank_upper - min_rank)
        discrete_rank = torch.round(continuous_rank)
        # Subtract first so the forward value stays exactly integer-valued,
        # including BF16; reassociation could activate one extra mask channel.
        straight_through_rank = discrete_rank.detach() + (continuous_rank - continuous_rank.detach())
        if self.complexity_type == 'legacy_rfft':
            straight_through_rank = discrete_rank.detach() + continuous_rank - continuous_rank.detach()
        return straight_through_rank, coefficients, complexity


class ACOMoeGate(nn.Module):
    """
    Gating Network for Mixture-of-Experts (MoE) augmented with ACO pheromones.
    """

    def __init__(self, input_dim, num_experts, top_k=2, noisy_gating=True):
        super(ACOMoeGate, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.routing_policy = 'consistent'

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.w_gate = nn.Linear(64, num_experts, bias=False)

        if noisy_gating:
            self.w_noise = nn.Linear(64, num_experts, bias=False)
            nn.init.normal_(self.w_noise.weight, std=0.01)

        # Pheromone tracking
        self.pheromone = nn.Parameter(torch.ones(num_experts), requires_grad=False)
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_selections', torch.tensor(0))

        nn.init.normal_(self.w_gate.weight, std=1e-3)

        self.evaporation_rate = 0.3
        self.alpha_gate = 1.0
        self.alpha = 1.5
        self.beta = 2.5
        self.Q = 0.1

    def set_top_k(self, top_k: int):
        self.top_k = min(top_k, self.num_experts)

    def forward(self, x, iteration=0, expert_energy=None):
        batch_size, channels, height, width = x.shape

        # 1. Feature Extraction
        x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        features = self.feature_proj(x_pooled)
        clean_logits = self.w_gate(features)
        if expert_energy is not None:
            normalized_energy = expert_energy / (expert_energy.sum(dim=1, keepdim=True) + 1e-8)
            clean_logits = clean_logits + normalized_energy

        # 2. Frequency Penalty (Load Balancing)
        if (self.training or self.routing_policy == 'consistent') and self.total_selections > 0:
            usage_ratio = self.expert_usage / (self.total_selections + 1e-6)
            eta = 2.0
            target_usage = self.top_k / self.num_experts
            freq_penalty = eta * (usage_ratio - target_usage)
            freq_penalty = torch.relu(freq_penalty)
            clean_logits = clean_logits - freq_penalty.unsqueeze(0)

        # 3. Noise Injection
        if self.noisy_gating and self.training:
            noise_logits = self.w_noise(features)
            noise_std = F.softplus(noise_logits) + 0.1
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits

        # if not self.training:
        #     probabilities = torch.softmax(logits, dim=-1)
        # else:
        #     # Combine with Pheromone during training
        #     heuristic = torch.softmax(logits, dim=-1)
        #     pheromone_expanded = self.pheromone.unsqueeze(0).expand(batch_size, -1)
        #
        #     # Mixed Fractions
        #     selection_scores = (pheromone_expanded ** 1.0) * (heuristic ** 2.0)
        #     # Normalized probability
        #     probabilities = selection_scores / (selection_scores.sum(dim=1, keepdim=True) + 1e-8)

        # 4. Use the same learned routing biases during training and inference.
        if self.training or self.routing_policy == 'consistent':
            # Convert pheromones to the Log space
            log_pheromone = torch.log(self.pheromone + 1e-8)

            # Extend to the Batch dimension
            log_pheromone_expanded = log_pheromone.unsqueeze(0).expand(batch_size, -1)

            final_logits = logits + self.alpha_gate * log_pheromone_expanded

            probabilities = torch.softmax(final_logits, dim=-1)

        else:
            probabilities = torch.softmax(logits, dim=-1)

        # 5. Top-K Selection
        top_probs, top_indices = torch.topk(probabilities, self.top_k, dim=1)
        top_probs = top_probs / (top_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Create Gate Matrix
        gates = torch.zeros_like(probabilities)
        gates.scatter_(1, top_indices, top_probs)
        self.last_selected_experts = top_indices.detach()

        # 6. Loss Calculation & Pheromone Update
        if self.training:
            loss = 1.5 * self.compute_load_balancing_loss(gates, clean_logits)
            self._update_pheromone_detached(top_indices, top_probs)
            self._update_expert_usage_detached(top_indices)
        else:
            loss = torch.tensor(0.0, device=x.device)

        return gates, loss

    def compute_load_balancing_loss(self, gates, logits):
        """Computes auxiliary loss for expert load balancing."""
        # CV Squared Loss
        importance = gates.sum(dim=0)
        cv_loss = self.cv_squared(importance)

        # Entropy Loss (Encourages certainty)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        entropy_loss = -0.1 * entropy

        return cv_loss + entropy_loss

    def cv_squared(self, x):
        """Coefficient of Variation squared."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _update_pheromone_detached(self, selected_experts, qualities):
        """Updates pheromone levels without tracking gradients."""
        with torch.no_grad():
            batch_size = selected_experts.shape[0]
            # Global Evaporation
            new_pheromone = self.pheromone.data * (1 - self.evaporation_rate)

            # Batch reduction avoids per-selection host synchronization.
            new_pheromone.scatter_add_(
                0, selected_experts.reshape(-1),
                (self.Q * qualities.detach()).reshape(-1).to(new_pheromone.dtype)
            )

            # Clamp and Normalize
            new_pheromone = torch.clamp(new_pheromone, min=0.1, max=5.0)
            if self.total_selections % 100 == 0:
                new_pheromone = new_pheromone / new_pheromone.sum() * self.num_experts

            self.pheromone.data = new_pheromone

    def _update_expert_usage_detached(self, selected_experts):
        with torch.no_grad():
            batch_size = selected_experts.shape[0]

            decay_factor = 0.95
            self.expert_usage *= decay_factor
            self.total_selections = self.total_selections * decay_factor + batch_size

            self.expert_usage.scatter_add_(
                0, selected_experts.reshape(-1),
                torch.ones_like(selected_experts, dtype=self.expert_usage.dtype).reshape(-1)
            )

    def get_expert_stats(self):
        """Obtain expert statistical information"""
        if self.total_selections > 0:
            usage_ratio = self.expert_usage / self.total_selections
            return {
                'usage_ratio': usage_ratio,
                'min_usage': usage_ratio.min().item(),
                'max_usage': usage_ratio.max().item(),
                'std_usage': usage_ratio.std().item(),
                'pheromone_mean': self.pheromone.mean().item(),
                'pheromone_std': self.pheromone.std().item()
            }
        return None


class AdaptiveExpert(nn.Module):
    """
    Adaptive Expert Network.

    This module adapts dynamic input channels to a fixed hidden dimension for processing,
    then projects them back. It allows the MoE layer to handle inputs with varying
    channel counts or rank-dependent outputs.
    """

    def __init__(self, input_channels, hidden_channels=32, dropout_rate=0.0):
        super(AdaptiveExpert, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Project varying input dimension to fixed hidden dimension
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Core processing convolution
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Project back to original input dimension
        self.output_proj = nn.Conv2d(hidden_channels, input_channels, 1, bias=False)

    def forward(self, x):
        # Projection & Pre-processing
        x = self.input_proj(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Context Processing
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Output Projection
        x = self.output_proj(x)
        return x


class ACOConvLoRA(nn.Module):
    """
    Convolutional LoRA Layer integrated with ACO-MoE gating.

    This layer combines:
    1. A frozen base convolution.
    2. A dynamic LoRA branch (Low-Rank Adaptation) where the rank is determined by ACO.
    3. A Mixture-of-Experts (MoE) system that processes the LoRA features, gated by ACO pheromones.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, max_rank=32,
                 num_experts=8, max_top_k=4, min_rank=4,
                 importance_reorder_interval=5, orthogonality_weight=0.01,
                 rank_smoothness_weight=0.01, rank_efficiency_weight=0.001,
                 rank_modulator_warmup_ratio=0.2,
                 orthogonality_warmup_epochs=10):
        super(ACOConvLoRA, self).__init__()

        # Frozen Base Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)

        self.max_rank = max_rank
        self.num_experts = num_experts
        self.max_top_k = max_top_k
        self.min_rank = min_rank
        self.importance_reorder_interval = importance_reorder_interval
        self.orthogonality_weight = orthogonality_weight
        self.rank_smoothness_weight = rank_smoothness_weight
        self.rank_efficiency_weight = rank_efficiency_weight
        self.rank_modulator_warmup_ratio = rank_modulator_warmup_ratio
        self.orthogonality_warmup_epochs = orthogonality_warmup_epochs

        # Dynamic State (Controlled by ACO)
        self.current_rank = 16
        self.current_alpha = 16.0
        self.current_top_k = 2

        # ACO-MoE Gating Mechanism
        self.aco_gate = ACOMoeGate(
            input_dim=in_channels,
            num_experts=num_experts,
            top_k=self.current_top_k
        )

        # LoRA Weights - Pre-allocated with maximum capacity
        self.lora_A = nn.Parameter(
            torch.zeros(max_rank, in_channels, kernel_size, kernel_size))
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels, max_rank, 1, 1))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.register_buffer('rank_order', torch.arange(max_rank, dtype=torch.long))
        self.rank_modulator = InstanceRankModulator(min_rank=min_rank)
        self.rank_modulator_active = True
        self.orthogonality_warmup_scale = 1.0
        self.last_adaptation_stats = {}
        self.sarm_mode = 'learned_fft'
        self.grouping_mode = 'importance_grouping'
        self.importance_type = 'paper_svd'
        self.fixed_rank = 8
        self.grouping_seed = 2026
        self.diagnostics_enabled = False
        self.adaptation_enabled = True

        # Adaptive Experts
        self.experts = nn.ModuleList([
            AdaptiveExpert(max_rank) for _ in range(num_experts)
        ])

    def set_hyperparams(self, rank: int, alpha: float, top_k: int):
        """Updates layer hyperparameters dynamically based on ACO search results."""
        self.current_rank = min(rank, self.max_rank)
        self.current_alpha = alpha
        self.current_top_k = min(top_k, self.max_top_k)
        self.aco_gate.set_top_k(self.current_top_k)

    def set_epoch(self, epoch, total_epochs=None):
        if self.importance_reorder_interval > 0 and epoch % self.importance_reorder_interval == 0:
            self.update_importance_order()
        if total_epochs is not None:
            warmup_epochs = total_epochs * self.rank_modulator_warmup_ratio
            self.rank_modulator_active = epoch >= warmup_epochs
            for parameter in self.rank_modulator.parameters():
                parameter.requires_grad = self.rank_modulator_active
        self.orthogonality_warmup_scale = min(
            1.0, epoch / max(self.orthogonality_warmup_epochs, 1)
        )

    def update_importance_order(self, force=False):
        if not force and not self.training:
            return
        with torch.no_grad():
            flattened_a = self.lora_A.flatten(1).float()
            # diag(U S^2 U^T) equals the squared row norm of A.
            a_score = flattened_a.norm(dim=1)
            if self.importance_type == 'paper_svd':
                u, singular_values, _ = torch.linalg.svd(flattened_a, full_matrices=False)
                a_score = (u.square() * singular_values.square().unsqueeze(0)).sum(dim=1).sqrt()
            importance = a_score
            if self.importance_type in ('paper_svd', 'legacy_ab_norm'):
                importance = importance * self.lora_B.flatten(2).norm(dim=(0, 2))
            if self.grouping_mode == 'sequential_grouping':
                self.rank_order.copy_(torch.arange(self.max_rank, device=self.rank_order.device))
                return
            self.rank_order.copy_(torch.argsort(importance, descending=True))

    def get_expert_channel_masks(self, rank=None, device=None, dtype=None):
        rank = self.current_rank if rank is None else min(rank, self.max_rank)
        device = self.lora_A.device if device is None else device
        dtype = self.lora_A.dtype if dtype is None else dtype
        positions = torch.arange(rank, device=device)
        expert_ids = positions.remainder(self.num_experts)
        if self.grouping_mode == 'random_grouping':
            generator = torch.Generator(device='cpu').manual_seed(self.grouping_seed)
            permutation = torch.randperm(rank, generator=generator).to(device)
            expert_ids = expert_ids[permutation]
        return torch.stack(
            [(expert_ids == expert_idx).to(dtype=dtype) for expert_idx in range(self.num_experts)]
        )

    def build_sample_rank_mask(self, ranks, coefficients, rank_upper):
        del coefficients
        channel_positions = torch.arange(
            rank_upper, device=ranks.device, dtype=ranks.dtype
        ).unsqueeze(0)
        hard_mask = (channel_positions < ranks.detach().unsqueeze(1)).to(ranks.dtype)
        soft_mask = torch.sigmoid(
            4.0 * (ranks.unsqueeze(1) - channel_positions - 0.5)
        )
        mask = hard_mask + (soft_mask - soft_mask.detach())
        return mask.unsqueeze(-1).unsqueeze(-1)

    def _factor_decorrelation_loss(self, active_lora_A, expert_masks):
        """Cross-expert A-factor correlation under the current parameterization.

        This is not a basis-invariant orthogonality constraint on BA.
        """
        normalized_rows = F.normalize(active_lora_A.flatten(1), dim=1)
        loss = active_lora_A.new_tensor(0.0)
        pair_count = 0
        for first_idx in range(self.num_experts):
            first = normalized_rows[expert_masks[first_idx].bool()]
            if first.numel() == 0:
                continue
            for second_idx in range(first_idx + 1, self.num_experts):
                second = normalized_rows[expert_masks[second_idx].bool()]
                if second.numel() == 0:
                    continue
                loss = loss + (first @ second.transpose(0, 1)).pow(2).mean()
                pair_count += 1
        return loss / max(pair_count, 1)

    def _orthogonality_loss(self, active_lora_A, expert_masks):
        return self._factor_decorrelation_loss(active_lora_A, expert_masks)

    @staticmethod
    def _rank_smoothness_loss(x, coefficients):
        if x.shape[0] < 2:
            return x.new_tensor(0.0)
        features = F.normalize(F.adaptive_avg_pool2d(x, 1).flatten(1), dim=1)
        similarity = features @ features.transpose(0, 1)
        pair_mask = torch.triu(similarity > 0.8, diagonal=1)
        if not pair_mask.any():
            return x.new_tensor(0.0)
        differences = (coefficients[:, None] - coefficients[None, :]).pow(2)
        return differences[pair_mask].mean()

    def forward(self, x, iteration=0):
        # 1. Base Convolution (Static)
        base_output = self.conv(x)
        if not self.adaptation_enabled:
            return base_output, x.new_tensor(0.0)

        if self.current_rank > 0:
            # 2. Importance-ordered LoRA projection and sample-adaptive rank masking
            active_indices = self.rank_order[:self.current_rank]
            lora_A = self.lora_A.index_select(0, active_indices)
            lora_output = F.conv2d(x, lora_A, stride=self.conv.stride,
                                   padding=self.conv.padding, dilation=self.conv.dilation)

            # active_rank controls effective capacity; ceiling projection has
            # already run, so proportional FLOPs reduction is not guaranteed.

            batch_size, current_rank, H, W = lora_output.shape
            if self.rank_modulator_active and self.sarm_mode != 'fixed_mean_rank':
                sample_ranks, rank_coefficients, complexity = self.rank_modulator(
                    x, rank_upper=current_rank
                )
                if self.sarm_mode == 'random_rank':
                    # Permutation preserves the exact batch rank distribution.
                    permutation = torch.randperm(batch_size, device=x.device)
                    sample_ranks = sample_ranks[permutation]
                    rank_coefficients = rank_coefficients[permutation]
            else:
                rank = min(current_rank, max(min(self.min_rank, current_rank), self.fixed_rank)) if self.sarm_mode == 'fixed_mean_rank' and self.rank_modulator_active else current_rank
                sample_ranks = x.new_full((batch_size,), float(rank))
                rank_coefficients = sample_ranks / current_rank
                complexity = InstanceRankModulator.compute_complexity_feature(x, self.rank_modulator.complexity_type).detach()
            sample_rank_mask = self.build_sample_rank_mask(
                sample_ranks, rank_coefficients, rank_upper=current_rank
            )
            lora_output = lora_output * sample_rank_mask

            expert_masks = self.get_expert_channel_masks(
                current_rank, device=lora_output.device, dtype=lora_output.dtype
            )
            channel_energy = lora_output.abs().mean(dim=(2, 3))
            expert_energy = channel_energy @ expert_masks.transpose(0, 1) / current_rank

            # 3. ACO expert selection with subspace-energy alignment
            gates, aco_loss = self.aco_gate(x, iteration, expert_energy=expert_energy)

            # 4. Expert processing on disjoint latent subspaces
            expert_outputs = []
            active_experts = range(len(self.experts)) if self.training else self.aco_gate.last_selected_experts.unique(sorted=True).tolist()
            for i in active_experts:
                expert = self.experts[i]
                expert_input = lora_output * expert_masks[i].view(1, -1, 1, 1)
                if current_rank < self.max_rank:
                    expert_input = F.pad(
                        expert_input, (0, 0, 0, 0, 0, self.max_rank - current_rank)
                    )
                expert_out = expert(expert_input)

                if current_rank < self.max_rank:
                    expert_out = expert_out[:, :current_rank]
                expert_out = expert_out * expert_masks[i].view(1, -1, 1, 1)

                gate_weight = gates[:, i].view(batch_size, 1, 1, 1)
                weighted_out = expert_out * gate_weight
                expert_outputs.append(weighted_out)

            # Aggregate Expert Outputs
            if expert_outputs:
                moe_output = sum(expert_outputs)
            else:
                moe_output = lora_output

            # 5. Importance-matched reconstruction and per-sample scaling
            lora_B = self.lora_B.index_select(1, active_indices)
            moe_output = F.conv2d(moe_output, lora_B)

            scaling = (self.current_alpha / sample_ranks.clamp_min(1.0)).view(-1, 1, 1, 1)
            output = base_output + moe_output * scaling

            orthogonality_loss = self._orthogonality_loss(lora_A, expert_masks)
            rank_smoothness_loss = self._rank_smoothness_loss(x, rank_coefficients)
            rank_efficiency_loss = rank_coefficients.mean()
            adaptation_loss = (
                aco_loss
                + self.orthogonality_weight * self.orthogonality_warmup_scale
                * orthogonality_loss
                + self.rank_smoothness_weight * rank_smoothness_loss
                + self.rank_efficiency_weight * rank_efficiency_loss
            )
            if not self.training:
                adaptation_loss = x.new_tensor(0.0)
            self.last_adaptation_stats = {
                'sample_ranks': sample_ranks.detach(),
                'rank_coefficients': rank_coefficients.detach(),
                'complexity': complexity.detach(),
                'orthogonality_loss': orthogonality_loss.detach(),
                'rank_smoothness_loss': rank_smoothness_loss.detach(),
                'rank_efficiency_loss': rank_efficiency_loss.detach()
            }

            return output, adaptation_loss

        return base_output, torch.tensor(0.0, device=x.device)

    def get_expert_stats(self):
        return self.aco_gate.get_expert_stats()

    def get_adaptation_stats(self):
        return self.last_adaptation_stats

    def get_lora_fisher_score(self):
        scores = []
        for param in [self.lora_A, self.lora_B]:
            if param.grad is not None:
                scores.append(param.grad.detach().pow(2).mean().item())
        return float(np.mean(scores)) if scores else 0.0

    def get_current_hyperparams(self):
        return {
            'rank': self.current_rank,
            'alpha': self.current_alpha,
            'top_k': self.current_top_k
        }


class TriEvoMoELoRA(nn.Module):
    """
    Main Model Architecture: SAM2 Encoder + TriEvo-MoE-LoRA Decoder.

    Uses Ant Colony Optimization to dynamically adjust:
    1. LoRA Rank and Alpha for the Encoder.
    2. Rank, Alpha, and Top-K experts for the Decoder layers.
    """
    def __init__(self, checkpoint_path=None,
                 num_experts=8,
                 use_multi_start=True,
                 aco_optimizer_config: Dict = None,
                 model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml'):
        super(TriEvoMoELoRA, self).__init__()

        # --- ACO Configuration ---
        if aco_optimizer_config is None:
            aco_optimizer_config = {
                'rank_candidates': [8, 16, 24, 32],
                'alpha_candidates': [8.0, 16.0, 24.0, 32.0, 48.0],
                'top_k_candidates': [1, 2, 3],
                'num_ants': 24,
                'evaporation_rate': 0.22,
                'diversification_interval': 20,
                'mutation_prob': 0.12,
                'cost_penalty_weight': 0.003,
                'uncertainty_warmup_ratio': 0.2,
                'fisher_update_interval': 5
            }

        if use_multi_start:
            self.aco_optimizer = MultiStartACOOptimizer(**aco_optimizer_config)
        else:
            self.aco_optimizer = ACOHyperparamOptimizer(**aco_optimizer_config)

        # --- Backbone Initialization (SAM2) ---
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path, device="cpu")
        else:
            model = build_sam2(model_cfg, device="cpu")

        # Prune unused SAM2 components to reduce memory footprint
        components_to_remove = [
            'sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder',
            'memory_attention', 'mask_downsample', 'obj_ptr_tpos_proj',
            'obj_ptr_proj'
        ]

        for comp in components_to_remove:
            if hasattr(model, comp):
                delattr(model, comp)

        if hasattr(model.image_encoder, 'neck'):
            del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk

        # Freeze Encoder Backbone
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Inject Dynamic LoRA into Encoder Attention Layers
        self.apply_dynamic_lora_to_attention(self.encoder)

        # --- Decoder Construction ---
        # Stacked ACO-ConvLoRA layers acting as the decoding path
        self.aco_layers = nn.ModuleList([
            ACOConvLoRA(channels, 64, num_experts=num_experts)
            for channels in reversed(self.encoder.channel_list)
        ])

        # Upsampling Path (U-Net style)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)

        self.head = nn.Conv2d(64, 1, kernel_size=1)

        # Training State
        self.training_iteration = 0
        self.current_encoder_params = None
        self.current_decoder_params = None
        self.current_epoch = 0

        # Initialize with default hyperparameters
        self._update_hyperparams()

    def forward(self, x, update_hyperparams: bool = True):

        # Step 1: Optimize Hyperparameters via ACO (Training Only)
        if self.training and update_hyperparams:
            self._update_hyperparams()

        # Step 2: Encoder Forward Pass
        x1, x2, x3, x4 = self.encoder(x)

        # Step 3: Decoder Forward Pass (ACO Layers)
        aco_losses = []

        # Pass features through ACO-LoRA layers bottom-up
        x1, loss1 = self.aco_layers[0](x1, self.training_iteration)
        aco_losses.append(loss1)

        x2, loss2 = self.aco_layers[1](x2, self.training_iteration)
        aco_losses.append(loss2)

        x3, loss3 = self.aco_layers[2](x3, self.training_iteration)
        aco_losses.append(loss3)

        x4, loss4 = self.aco_layers[3](x4, self.training_iteration)
        aco_losses.append(loss4)

        total_aco_loss = sum(aco_losses) / len(aco_losses)

        if self.training:
            self.training_iteration += 1

        # Step 4: Upsampling and Head
        x = self.up4(x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear', align_corners=True)

        return out, total_aco_loss

    def apply_dynamic_lora_to_attention(self, module):
        """Recursively replaces linear layers in attention blocks with DynamicLoRAWrapper."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and ('qkv' in name or 'proj' in name):
                setattr(module, name, DynamicLoRAWrapper(child))
            else:
                self.apply_dynamic_lora_to_attention(child)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if hasattr(self.aco_optimizer, 'set_epoch'):
            self.aco_optimizer.set_epoch(epoch)
        for layer in self.aco_layers:
            layer.set_epoch(epoch, total_epochs=self.aco_optimizer.total_epochs)

    def _update_hyperparams(self):
        """Retrieves new hyperparameters from ACO and applies them to the model."""
        encoder_params, decoder_params = self.aco_optimizer.select_hyperparams(
            self.training_iteration
        )

        self.current_encoder_params = encoder_params
        self.current_decoder_params = decoder_params

        # Logging (Optional: consider using standard logging instead of print)
        print(f"  Encoder: rank={encoder_params['rank']}, alpha={encoder_params['alpha']}")
        if 'layers' in decoder_params:
            layer_summary = ", ".join(
                f"L{i}:r{p['rank']}/a{p['alpha']}/k{p['top_k']}"
                for i, p in enumerate(decoder_params['layers'])
            )
            print(f"  Decoder: {layer_summary}")
        else:
            print(f"  Decoder: rank={decoder_params['rank']}, alpha={decoder_params['alpha']}, top_k={decoder_params['top_k']}")

        # Apply to Encoder
        self._apply_encoder_hyperparams(encoder_params)
        
        # Apply to Decoder
        decoder_layers = decoder_params.get('layers') or [decoder_params for _ in self.aco_layers]
        for layer, layer_params in zip(self.aco_layers, decoder_layers):
            layer.set_hyperparams(
                rank=layer_params['rank'],
                alpha=layer_params['alpha'],
                top_k=layer_params['top_k']
            )

    def _apply_encoder_hyperparams(self, encoder_params):
        """Applies chosen rank/alpha to all Dynamic LoRA layers in the encoder."""
        if not isinstance(encoder_params, dict):
            print(f"Warning: encoder_params expected dict, got {type(encoder_params)}")
            encoder_params = {'rank': 16, 'alpha': 1.0}

        def apply_to_module(module):
            for child in module.children():
                if isinstance(child, DynamicLoRAWrapper):
                    child.set_hyperparams(
                        rank=encoder_params['rank'],
                        alpha=encoder_params['alpha']
                    )
                else:
                    apply_to_module(child)

        apply_to_module(self.encoder)

    def set_deterministic_mode(self):
        """Disables noisy gating without changing the caller's random seed."""
        for layer in self.aco_layers:
            layer.aco_gate.noisy_gating = False

    def set_test_hyperparams(self):
        """Locks the model to the best historical hyperparameters for inference/testing."""
        if not self.training:
            encoder_params, decoder_params = self.aco_optimizer.get_best_hyperparams()

            print(f"Setting test hyperparameters:")
            print(f"  Encoder: rank={encoder_params['rank']}, alpha={encoder_params['alpha']}")
            print(
                f"  Decoder: rank={decoder_params['rank']}, alpha={decoder_params['alpha']}, top_k={decoder_params['top_k']}")

            self._apply_encoder_hyperparams(encoder_params)

            decoder_layers = decoder_params.get('layers') or [decoder_params for _ in self.aco_layers]
            for layer, layer_params in zip(self.aco_layers, decoder_layers):
                layer.set_hyperparams(
                    rank=layer_params['rank'],
                    alpha=layer_params['alpha'],
                    top_k=layer_params['top_k']
                )

            self.set_deterministic_mode()
            self.current_encoder_params = encoder_params
            self.current_decoder_params = decoder_params

    def update_aco_with_performance(self, performance_metric: float,
                                    layer_scores: Dict[str, float] = None,
                                    prediction_uncertainty: float = None):
        """Feeds back the validation metric (e.g., IoU) to the ACO optimizer."""
        solution = (self.current_encoder_params, self.current_decoder_params)
        self.aco_optimizer.update_pheromone(
            [(solution, performance_metric)],
            layer_scores=layer_scores,
            prediction_uncertainty=prediction_uncertainty
        )

        if self.training_iteration % 100 == 0:
            stats = self.aco_optimizer.get_optimizer_stats()
            print(f"ACO Stats: Best Perf={stats['best_performance']:.4f}, "
                  f"Stagnation={stats['stagnation_count']}, Diversity={stats['current_diversity']:.3f}")

    def get_expert_statistics(self):
        """Aggregates usage statistics from all expert layers."""
        stats = {}
        for i, layer in enumerate(self.aco_layers):
            expert_stats = layer.get_expert_stats()
            if expert_stats:
                stats[f'layer_{i}'] = expert_stats
        return stats

    def get_rank_distribution(self):
        """Calculates average rank usage across encoder and decoder."""
        encoder_rank = self.current_encoder_params['rank'] if self.current_encoder_params else 16

        decoder_ranks = []
        for layer in self.aco_layers:
            hyperparams = layer.get_current_hyperparams()
            decoder_ranks.append(hyperparams['rank'])

        decoder_mean_rank = sum(decoder_ranks) / len(decoder_ranks) if decoder_ranks else 0

        return {
            'encoder_mean_rank': encoder_rank,
            'decoder_mean_rank': decoder_mean_rank
        }

    def collect_lora_fisher_scores(self):
        """Gradient-squared, Fisher-inspired local sensitivity proxy from L_total.

        This is not a causal contribution estimator.
        """
        encoder_scores = []
        for name, param in self.encoder.named_parameters():
            if 'lora' in name and param.grad is not None:
                encoder_scores.append(param.grad.detach().pow(2).mean().item())

        layer_scores = {
            'encoder': float(np.mean(encoder_scores)) if encoder_scores else 0.0
        }
        for layer_idx, layer in enumerate(self.aco_layers):
            layer_scores[f'decoder_{layer_idx}'] = layer.get_lora_fisher_score()
        return layer_scores

    def collect_adaptation_stats(self):
        """Collects SAOED/SARM statistics from decoder layers."""
        stats = {}
        for layer_idx, layer in enumerate(self.aco_layers):
            layer_stats = layer.get_adaptation_stats()
            if not layer_stats:
                continue
            stats[f'decoder_{layer_idx}'] = {
                key: float(value.float().mean().item())
                for key, value in layer_stats.items()
            }
        return stats

    def get_current_hyperparams(self):
        return {
            'encoder': self.current_encoder_params,
            'decoder': self.current_decoder_params
        }

    def get_best_hyperparams(self):
        return self.aco_optimizer.get_best_hyperparams()


class AdaptiveTrainingScheduler:
    """
    Adaptive scheduler for the ACO loss weight.

    Dynamically adjusts the importance of the auxiliary ACO loss based on performance history.
    It reduces the ACO weight if performance stagnates to allow the main task loss to dominate,
    and periodically resets it (cooldown) to re-encourage exploration.
    """

    def __init__(self, initial_aco_weight=1.0, cooldown_epochs=10):
        self.initial_aco_weight = initial_aco_weight
        self.cooldown_epochs = cooldown_epochs
        self.aco_weight = initial_aco_weight
        self.epoch_count = 0

    def step(self, current_performance, performance_history):
        self.epoch_count += 1

        # Check for stagnation (no significant improvement in last 10 epochs)
        if len(performance_history) > 10:
            recent_improvement = performance_history[-1] - performance_history[-10]
            if recent_improvement < 0.01:
                self.aco_weight = max(min(0.1, self.initial_aco_weight), self.aco_weight * 0.9)

        # Periodic Reset (Cooldown/Heating)
        if self.epoch_count % self.cooldown_epochs == 0:
            self.aco_weight = min(self.initial_aco_weight, self.aco_weight * 1.2)

        return self.aco_weight

if __name__ == "__main__":
    # Test Block: Verifies model instantiation and forward pass dimension integrity.
    # Note: Requires a CUDA environment for full execution.
    try:
        with torch.no_grad():
            print("Initializing TriEvoMoELoRA model...")
            model = TriEvoMoELoRA(num_experts=4)
            if torch.cuda.is_available():
                model = model.cuda()
                x = torch.randn(1, 3, 352, 352).cuda()
            else:
                x = torch.randn(1, 3, 352, 352)
                print("Warning: Running on CPU.")

            print("Executing forward pass...")
            out, aco_loss = model(x)
            print(f"Output Shape: {out.shape}, ACO Loss Shape: {aco_loss.shape}")
            print(f"ACO Loss Value: {aco_loss.item():.4f}")

            # Verify stats extraction
            stats = model.get_expert_statistics()
            print("\nExpert Statistics Sample:", list(stats.keys()))

            rank_info = model.get_rank_distribution()
            print("Rank Distribution:", rank_info)

            hyperparams = model.get_current_hyperparams()
            print("Current Hyperparameters:", hyperparams)

    except Exception as e:
        print(f"Test failed with error: {e}")
