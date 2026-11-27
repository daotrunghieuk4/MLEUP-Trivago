"""Runs in Colab once PyTorch is available: model forward/backward checks."""

import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.trivago_top5_checkpoint import restore_checkpoint_rng

try:
    import torch
    if not hasattr(torch, "Tensor"):
        raise ImportError("Incomplete torch namespace")
    from src.model import MLEUP, forward
    from src.util import Data, collate_fn
    from src.train_trivago_top5 import evaluate
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable here")
class ModelTests(unittest.TestCase):
    def test_fieldwise_full_model_matches_concat_with_and_without_actions(self):
        """Compare outputs in train/eval and gradients in train mode."""
        from src.trivago_action_types import ACTION_ENCODER, ACTION_FUSION, NUM_ACTIONS
        cpu_rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, cpu_rng)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            self.addCleanup(torch.cuda.set_rng_state_all, torch.cuda.get_rng_state_all())
        cards = [6, 4, 5] + [2] * 50
        adjacency_fi = [np.array([
            [1 + (value * 2 + field) % 5, 1 + (value * 2 + field + 1) % 5, 0]
            for value in range(card)
        ], dtype=np.int32) for field, card in enumerate(cards)]
        adjacency_if = np.array([(np.arange(5) + field) % card
                                 for field, card in enumerate(cards)], dtype=np.uint8)
        data = [[[1, 3], [4, 5], [1, 2]], [2, 2, 3]]
        groups = [np.array([[3], [3], [0]]), np.array([[3], [3], [1]]),
                  np.zeros((3, 1), dtype=np.float32)]
        for action_mode in ('none', 'item_event'):
            opts = SimpleNamespace(m=1, emb_size=4, batch_size=1, sample_order=2,
                                   num_heads=2, beta=0.3, alpha=0.2,
                                   metadata_projection='concat', valid_groups_only=True,
                                   action_mode=action_mode,
                                   num_actions=NUM_ACTIONS if action_mode == 'item_event' else 0,
                                   action_encoder=ACTION_ENCODER if action_mode == 'item_event' else 'none',
                                   action_fusion=ACTION_FUSION if action_mode == 'item_event' else 'none')
            template = MLEUP(opts, 5, cards, adjacency_fi, adjacency_if).to(device)
            actions = [[4, 5], [3, 6], [4, 2]] if action_mode == 'item_event' else None
            dataset = Data(data, groups, 2, 2.0, lazy_graphs=True, action_sequences=actions)
            batch = collate_fn([dataset[2]])
            for training in (False, True):
                with self.subTest(action_mode=action_mode, training=training):
                    concat = copy.deepcopy(template).train(training)
                    fieldwise = copy.deepcopy(template).train(training)
                    fieldwise.metadata_projection = 'fieldwise'
                    cpu_state = torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
                    with torch.set_grad_enabled(training):
                        targets_a, scores_a, aux_a = forward(concat, batch)
                        torch.set_rng_state(cpu_state)
                        if cuda_state is not None:
                            torch.cuda.set_rng_state_all(cuda_state)
                        targets_b, scores_b, aux_b = forward(fieldwise, batch)
                    self.assertTrue(torch.equal(targets_a, targets_b))
                    torch.testing.assert_close(scores_a, scores_b, atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(aux_a, aux_b, atol=1e-5, rtol=1e-4)
                    if not training:
                        self.assertFalse(scores_a.requires_grad)
                        self.assertFalse(scores_b.requires_grad)
                        continue
                    loss_a = torch.nn.functional.cross_entropy(scores_a, targets_a) + aux_a
                    loss_b = torch.nn.functional.cross_entropy(scores_b, targets_b) + aux_b
                    self.assertTrue(torch.isfinite(loss_a).item())
                    self.assertTrue(torch.isfinite(loss_b).item())
                    loss_a.backward()
                    loss_b.backward()
                    for (name_a, a), (name_b, b) in zip(concat.named_parameters(),
                                                       fieldwise.named_parameters()):
                        self.assertEqual(name_a, name_b)
                        self.assertEqual(a.grad is None, b.grad is None, name_a)
                        if a.grad is not None:
                            self.assertTrue(torch.isfinite(a.grad).all().item(), name_a)
                            self.assertTrue(torch.isfinite(b.grad).all().item(), name_b)
                            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4,
                                                       msg=name_a)

    def test_top5_metrics_candidate_ranks_and_batch_averaging(self):
        # Scores outside impressions must not affect the candidate rank.
        scores = torch.tensor([[100., 6., 5., 4., 3., 2., 1.]] * 3)
        targets = torch.tensor([1, 5, 6])  # ranks 1, 5, 6
        candidates = [[1, 2, 3, 4, 5, 6]] * 3
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                with patch('src.train_trivago_top5.batch_at', side_effect=lambda data, ids: ids), \
                     patch('src.train_trivago_top5.forward',
                           side_effect=lambda model, ids: (targets[ids], scores[ids], None)):
                    result = evaluate(unittest.mock.Mock(), list(range(3)), candidates, batch_size)
                self.assertEqual(set(result), {'ndcg_at_5', 'recall_at_5', 'examples'})
                self.assertEqual(result['examples'], 3)
                self.assertAlmostEqual(result['recall_at_5'], 2 / 3)
                self.assertAlmostEqual(result['ndcg_at_5'], (1 + 1 / np.log2(6)) / 3)

    def test_top5_ties_keep_impression_order(self):
        targets = torch.tensor([5, 6])
        scores = torch.ones((2, 7))
        with patch('src.train_trivago_top5.batch_at', side_effect=lambda data, ids: ids), \
             patch('src.train_trivago_top5.forward',
                   side_effect=lambda model, ids: (targets[ids], scores[ids], None)):
            result = evaluate(unittest.mock.Mock(), [0, 1], [[1, 2, 3, 4, 5, 6]] * 2, 2)
        self.assertEqual(result['recall_at_5'], 0.5)
        self.assertAlmostEqual(result['ndcg_at_5'], 1 / np.log2(6) / 2)

    def test_top5_rejects_empty_and_misaligned_splits(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            evaluate(unittest.mock.Mock(), [], [], 1)
        with self.assertRaisesRegex(ValueError, 'align'):
            evaluate(unittest.mock.Mock(), [0], [], 1)

    def test_single_concat_matches_incremental_full_model_and_gradients(self):
        """Check all 53 fields, dropout RNG, both losses and every parameter."""
        cpu_rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, cpu_rng)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.addCleanup(torch.cuda.set_rng_state_all, torch.cuda.get_rng_state_all())
        cards = [6, 4, 5] + [2] * 50
        opts = SimpleNamespace(m=1, emb_size=4, batch_size=1, sample_order=2,
                               num_heads=2, beta=0.3, alpha=0.2,
                               metadata_projection="concat", valid_groups_only=True)
        adjacency_fi = [np.array([
            [1 + (value * 2 + field) % 5,
             1 + (value * 2 + field + 1) % 5, 0]
            for value in range(card)
        ], dtype=np.int32) for field, card in enumerate(cards)]
        adjacency_if = np.array([
            (np.arange(5) + field) % card
            for field, card in enumerate(cards)
        ], dtype=np.uint8)
        template = MLEUP(opts, 5, cards, adjacency_fi, adjacency_if).to(device)
        data = [[[1, 3], [4, 5], [1, 2]], [2, 2, 3]]
        groups = [np.array([[3], [3], [0]]), np.array([[3], [3], [1]]),
                  np.zeros((3, 1), dtype=np.float32)]
        batch = collate_fn([Data(data, groups, 2, 2.0, lazy_graphs=True)[2]])
        real_cat = torch.cat

        def incremental_cat(tensors, dim=0, **kwargs):
            if dim == -1 and len(tensors) == len(cards):
                result = tensors[0].new_empty((5, 0))
                for tensor in tensors:
                    result = real_cat([result, tensor], dim=dim)
                return result
            return real_cat(tensors, dim=dim, **kwargs)

        for training in (False, True):
            with self.subTest(training=training):
                old_model = copy.deepcopy(template).train(training)
                new_model = copy.deepcopy(template).train(training)
                cpu_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                with patch("src.model.torch.cat", side_effect=incremental_cat):
                    old_targets, old_scores, old_metadata_loss = forward(old_model, batch)
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
                with patch("src.model.torch.cat", wraps=real_cat) as cat_spy:
                    targets, scores, metadata_loss = forward(new_model, batch)
                metadata_cats = [call for call in cat_spy.call_args_list
                                 if call.kwargs.get("dim") == -1
                                 and len(call.args[0]) == len(cards)]
                self.assertEqual(len(metadata_cats), 1)
                self.assertTrue(torch.equal(old_targets, targets))
                torch.testing.assert_close(old_scores, scores, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(old_metadata_loss, metadata_loss,
                                           atol=1e-6, rtol=1e-5)
                old_loss = torch.nn.functional.cross_entropy(old_scores, old_targets) + old_metadata_loss
                new_loss = torch.nn.functional.cross_entropy(scores, targets) + metadata_loss
                old_loss.backward()
                new_loss.backward()
                for (name, old_parameter), (new_name, parameter) in zip(
                        old_model.named_parameters(), new_model.named_parameters()):
                    self.assertEqual(name, new_name)
                    self.assertEqual(old_parameter.grad is None, parameter.grad is None, name)
                    if parameter.grad is not None:
                        torch.testing.assert_close(old_parameter.grad, parameter.grad,
                                                   atol=1e-6, rtol=1e-5, msg=name)

    def test_checkpoint_rng_roundtrip_on_available_device(self):
        cpu_rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, cpu_rng)
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if cuda_rng is not None:
            self.addCleanup(torch.cuda.set_rng_state_all, cuda_rng)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        buffer = io.BytesIO()
        torch.save({"cpu_rng": cpu_rng, "cuda_rng": cuda_rng}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location=device, weights_only=False)
        torch.rand(3)
        if device.type == "cuda":
            torch.rand(3, device=device)
        restore_checkpoint_rng(checkpoint, device, torch)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        if cuda_rng is not None:
            for restored, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng):
                self.assertTrue(torch.equal(restored, expected))

    def test_fieldwise_matches_concat_output_and_gradients(self):
        opts = SimpleNamespace(m=2, emb_size=8, batch_size=1, sample_order=2,
                               num_heads=2, beta=0.3, alpha=0.2,
                               metadata_projection="fieldwise")
        adjacency_fi = [np.array([[0, 1], [2, 3]]),
                        np.array([[0, 2], [1, 3]])]
        adjacency_if = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        model = MLEUP(opts, 3, [2, 2], adjacency_fi, adjacency_if)
        model.eval()
        embeddings = torch.cat((model.embedding_i.weight.new_zeros((1, 8)),
                                model.embedding_i.weight), dim=0)
        field_values = [module(torch.as_tensor(neighbors), embeddings)
                        for module, neighbors in zip(model.all_feature_model,
                                                     adjacency_fi)]
        old = model.W1_feature(torch.cat([
            value[torch.as_tensor(adjacency_if[index], dtype=torch.long)]
            for index, value in enumerate(field_values)
        ], dim=-1))
        new = model.project_feature_values(embeddings)
        self.assertTrue(torch.allclose(old, new, atol=1e-5, rtol=1e-5))
        old.sum().backward(retain_graph=True)
        old_gradient = model.W1_feature.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        new.sum().backward()
        self.assertTrue(torch.allclose(old_gradient,
                                       model.W1_feature.weight.grad,
                                       atol=1e-5, rtol=1e-5))

    def test_real_groups_keep_four_metadata_paths_trainable(self):
        opts = SimpleNamespace(m=1, emb_size=8, batch_size=1, sample_order=2,
                               num_heads=2, beta=0.3, alpha=0.2,
                               metadata_projection="concat", valid_groups_only=True)
        adjacency_fi = [np.array([[1, 3, 5], [2, 4, 0]]),
                        np.array([[2, 4, 0], [1, 3, 5]])]
        adjacency_if = np.array([[0, 1, 0, 1, 0],
                                 [1, 0, 1, 0, 1]], dtype=np.uint8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLEUP(opts, 5, [2, 2], adjacency_fi, adjacency_if).to(device)
        self.assertEqual(len(model.all_feature_model), 2)
        optimizer = torch.optim.Adam(model.parameters())
        registered = {id(param) for group in optimizer.param_groups
                      for param in group["params"]}
        for field in model.all_feature_model:
            self.assertIn(id(field.embedding_f.weight), registered)
        data = [[[1, 3], [4, 5], [1, 2]], [2, 2, 3]]
        # Example 2 has a real overlapping positive (0) and disjoint negative (1).
        # Index 3 is padding for earlier examples and must not enter the loss.
        groups = [np.array([[3], [3], [0]]),
                  np.array([[3], [3], [1]]),
                  np.zeros((3, 1), dtype=np.float32)]
        dataset = Data(data, groups, 2, 2.0, lazy_graphs=True)
        targets, scores, metadata_loss = forward(model, collate_fn([dataset[2]]))
        self.assertEqual(tuple(scores.shape), (1, 6))
        self.assertTrue(torch.isfinite(metadata_loss))
        self.assertGreater(metadata_loss.item(), 0)
        metadata_loss.backward(retain_graph=True)
        self.assertIsNotNone(model.W1_feature.weight.grad)
        self.assertGreater(model.W1_feature.weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        (torch.nn.functional.cross_entropy(scores, targets) + metadata_loss).backward()
        self.assertIsNotNone(model.W1_feature.weight.grad)
        self.assertIsNotNone(model.W2_feature.weight.grad)
        self.assertIsNotNone(model.W7_co.weight.grad)
        self.assertIsNotNone(model.W6_seq.weight.grad)
        for field in model.all_feature_model:
            self.assertIsNotNone(field.embedding_f.weight.grad)

        _, _, padded_loss = forward(model, collate_fn([dataset[0]]))
        self.assertEqual(padded_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
