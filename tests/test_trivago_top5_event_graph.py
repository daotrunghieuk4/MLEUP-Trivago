"""Regression checks for event graphs, candidate scoring, and loss reduction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.trivago_action_types import NUM_ACTIONS, EVENT_GRAPH_ENCODER, EVENT_GRAPH_FUSION
from src.trivago_top5_checkpoint import (
    METADATA_LOSS_REDUCTION, CANDIDATE_POLICY, checkpoint_model_arguments, validate_checkpoint,
)
from src.trivago_price_features import fit_price_scaler, flatten_candidate_records

try:
    import torch
    if not hasattr(torch, 'Tensor'):
        raise ImportError('Incomplete torch namespace')
    from src.model import MLEUP, forward
    from src.util import Data, collate_fn
    from src.train_trivago_top5 import evaluate
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class EventGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.options = SimpleNamespace(
            m=1, emb_size=4, batch_size=2, sample_order=2, num_heads=2,
            beta=.3, alpha=.2, si_dc=2., metadata_projection='fieldwise',
            valid_groups_only=True, action_mode='event_graph_candidate',
            num_actions=NUM_ACTIONS, action_encoder=EVENT_GRAPH_ENCODER,
            action_fusion=EVENT_GRAPH_FUSION,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLEUP(
            self.options, 5, [2], [np.array([[1, 2, 3], [4, 5, 0]])],
            np.array([[0, 0, 0, 1, 1]]),
        ).to(device)
        groups = [np.array([[3], [3], [0]]), np.array([[3], [3], [1]]),
                  np.zeros((3, 1))]
        self.dataset = Data(
            [[[1, 3], [4, 5], [1, 2, 1]], [2, 2, 3]], groups, 2, 2.,
            lazy_graphs=True, action_sequences=[[4, 5], [3, 6], [4, 2, 3]],
            candidate_lists=[[2, 3], [1, 2, 4, 5], [3, 1, 2]],
        )

    def batch(self, indices):
        return collate_fn([self.dataset[i] for i in indices])

    def test_candidate_forward_backward_masks_padding_and_trains_both_extensions(self):
        targets, scores, auxiliary = forward(self.model, self.batch([0, 1, 2]))
        self.assertEqual(targets.tolist(), [0, 1, 0])
        self.assertEqual(tuple(scores.shape), (3, 4))
        self.assertEqual(scores[0, 2].item(), torch.finfo(scores.dtype).min)
        loss = torch.nn.functional.cross_entropy(scores, targets) + auxiliary
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in ('action_embedding', 'event_input', 'event_temporal_value',
                       'event_repeat_value', 'candidate_query', 'candidate_event_key'):
            with self.subTest(module=module):
                grad = getattr(self.model, module).weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_relative_price_head_masks_padding_and_receives_finite_gradients(self):
        self.options.price_mode = 'relative_mlp_v1'
        self.model = MLEUP(
            self.options, 5, [2], [np.array([[1, 2, 3], [4, 5, 0]])],
            np.array([[0, 0, 0, 1, 1]]),
        ).to(next(self.model.parameters()).device)
        rows = [
            [(2, 0, 10.0), (3, 1, 30.0)],
            [(1, 0, 15.0), (2, 1, None), (4, 2, 25.0), (5, 3, 100.0)],
            [(3, 0, None), (1, 1, 8.0), (2, 2, 16.0)],
        ]
        records = flatten_candidate_records(rows)
        records['query_ids'] = np.asarray([100, 101, 102], dtype=np.int64)
        records['target_ids'] = self.dataset.targets.copy()
        scaler = fit_price_scaler([[record[2] for record in row] for row in rows])
        price_dataset = Data(
            [[[1, 3], [4, 5], [1, 2, 1]], [2, 2, 3]],
            [np.array([[3], [3], [0]]), np.array([[3], [3], [1]]),
             np.zeros((3, 1))],
            2, 2., lazy_graphs=True,
            action_sequences=[[4, 5], [3, 6], [4, 2, 3]],
            candidate_lists=self.dataset.candidate_lists,
            candidate_price_records=records, price_scaler=scaler,
        )
        self.model.eval()
        targets, scores, auxiliary = forward(
            self.model, collate_fn([price_dataset[0], price_dataset[1]]),
        )
        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertEqual(scores[0, 2].item(), torch.finfo(scores.dtype).min)
        self.assertTrue(torch.isfinite(scores[0, :2]).all())
        (torch.nn.functional.cross_entropy(scores, targets) + auxiliary).backward()
        for parameter in (self.model.price_encoder[0].weight,
                          self.model.price_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_validation_loss_uses_candidate_targets_and_weights_uneven_batches(self):
        self.model.eval()
        expected_ce = expected_metadata = 0.
        with torch.no_grad():
            for indices in ([0, 1], [2]):
                targets, scores, auxiliary = forward(self.model, self.batch(indices))
                expected_ce += float(torch.nn.functional.cross_entropy(scores, targets)) * len(indices) / 3
                expected_metadata += float(auxiliary) * len(indices) / 3
        result = evaluate(self.model, self.dataset, self.dataset.candidate_lists, 2, include_loss=True)
        self.assertAlmostEqual(result['ce_loss'], expected_ce)
        self.assertAlmostEqual(result['metadata_loss'], expected_metadata)
        self.assertAlmostEqual(result['total_loss'], expected_ce + expected_metadata)

    def test_metadata_mean_and_its_gradient_are_invariant_to_duplicate_batch_rows(self):
        self.model.eval()  # Isolate reduction from stochastic dropout.
        reference = None
        for batch_size in (1, 2, 8):
            self.model.zero_grad(set_to_none=True)
            targets, scores, auxiliary = forward(self.model, self.batch([2] * batch_size))
            ce = torch.nn.functional.cross_entropy(scores, targets)
            auxiliary.backward()
            grad = self.model.all_feature_model[0].embedding_f.weight.grad.clone()
            if reference is None:
                reference = (ce.detach(), auxiliary.detach(), grad)
                self.assertGreater(auxiliary.item(), 0)
            else:
                torch.testing.assert_close(ce, reference[0])
                torch.testing.assert_close(auxiliary, reference[1])
                torch.testing.assert_close(grad, reference[2], atol=1e-6, rtol=1e-5)

    def test_missing_pairs_do_not_dilute_the_valid_pair_mean(self):
        self.model.eval()
        with torch.no_grad():
            _, _, single = forward(self.model, self.batch([2]))
            _, _, mixed = forward(self.model, self.batch([0, 2]))
        torch.testing.assert_close(single, mixed)

    def test_batch_without_valid_pairs_has_zero_finite_metadata_gradient(self):
        self.model.eval()
        _, _, auxiliary = forward(self.model, self.batch([0, 1]))
        self.assertEqual(auxiliary.item(), 0.)
        auxiliary.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() and not g.any() for g in grads))

    def test_checkpoint_requires_the_normalized_loss_version(self):
        args = checkpoint_model_arguments(self.options)
        self.assertEqual(args['metadata_loss_reduction'], METADATA_LOSS_REDUCTION)
        self.assertEqual(args['candidate_policy'], CANDIDATE_POLICY)
        legacy = dict(args)
        del legacy['metadata_loss_reduction']
        with self.assertRaisesRegex(ValueError, 'metadata_loss_reduction'):
            validate_checkpoint({'signature': ('fixture',), 'model_arguments': legacy},
                                ('fixture',), args)

    def test_all_modes_rank_duplicate_candidates_once_and_preserve_first_position(self):
        # The former evaluator counted a duplicate negative twice, pushing
        # rank five to rank six despite identical per-hotel scores.
        cases = [([1, 1, 2, 3, 4, 5], 5, [0., 6., 5., 4., 3., 2.]),
                 ([3, 3, 1, 4, 5, 2], 2, [0., 1., 1., 1., 1., 1.])]
        for raw_candidates, target, item_scores in cases:
            candidates = [raw_candidates.copy()]
            dataset = Data(
                [[[1, 2]], [target]],
                [np.array([[1]]), np.array([[1]]), np.zeros((1, 1))],
                2, 2., lazy_graphs=True, action_sequences=[[4, 3]],
                candidate_lists=candidates,
            )
            global_scores = torch.tensor([item_scores])
            results = []
            for mode in ('none', 'item_event', 'all_event', 'event_graph_candidate'):
                if mode == 'event_graph_candidate':
                    scores = global_scores[:, dataset.candidate_lists[0]]
                    labels = torch.as_tensor(dataset.candidate_target_positions)
                else:
                    scores, labels = global_scores, torch.tensor([target])
                fake_model = SimpleNamespace(action_mode=mode, eval=lambda: None)
                with patch('src.train_trivago_top5.batch_at', return_value=None), \
                        patch('src.train_trivago_top5.forward', return_value=(labels, scores, 0)):
                    results.append(evaluate(fake_model, dataset, candidates, 1))
            for result in results:
                self.assertEqual(result['recall_at_5'], 1.)
                self.assertAlmostEqual(result['ndcg_at_5'], 1 / np.log2(6))
            self.assertEqual(candidates[0], raw_candidates)

    def test_duplicate_target_has_one_class_and_correct_local_position(self):
        dataset = Data(
            [[[1, 2]], [2]],
            [np.array([[1]]), np.array([[1]]), np.zeros((1, 1))],
            2, 2., lazy_graphs=True, action_sequences=[[4, 3]],
            candidate_lists=[[3, 2, 3, 2]],
        )
        self.assertEqual(dataset.candidate_lists, [[3, 2]])
        self.assertEqual(dataset.candidate_target_positions.tolist(), [1])
        batch = collate_fn([dataset[0]])
        self.assertEqual(batch[16].tolist(), [[3, 2]])
        self.assertEqual(batch[18].tolist(), [1])


if __name__ == '__main__':
    unittest.main()
