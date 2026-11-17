"""Action-v1 preprocessing contract and GRU integration tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trivago_action_types import (
    ACTION_ENCODER, ACTION_FUSION, ACTION_SCHEMA_VERSION, ACTION_TO_ID,
    ITEM_ACTIONS, NUM_ACTIONS, PAD_ACTION_ID, UNK_ACTION_ID,
)

try:
    import torch
    from model import MLEUP, forward
    from util import Data, collate_fn
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable here")
class ActionModelTests(unittest.TestCase):
    def setUp(self):
        self.n_node = 3
        self.cards = [2]
        self.adjacency_fi = [np.array([[1, 2], [3, 0]], dtype=np.int32)]
        self.adjacency_if = np.array([[0, 1, 0]], dtype=np.uint8)

    def make_model(self, action_mode="item_event"):
        options = SimpleNamespace(
            m=1, emb_size=4, batch_size=2, sample_order=2, num_heads=2,
            beta=0.3, alpha=0.2, metadata_projection="concat",
            valid_groups_only=True, action_mode=action_mode,
            num_actions=NUM_ACTIONS if action_mode == "item_event" else 0,
            action_encoder=ACTION_ENCODER if action_mode == "item_event" else "none",
            action_fusion=ACTION_FUSION if action_mode == "item_event" else "none",
        )
        return MLEUP(options, self.n_node, self.cards,
                     self.adjacency_fi, self.adjacency_if)

    def make_dataset(self, with_actions=True):
        item_sequences = [[1, 2, 1], [2, 3]]
        targets = [2, 3]
        groups = [np.full((2, 1), 2, dtype=np.int32),
                  np.full((2, 1), 2, dtype=np.int32),
                  np.zeros((2, 1), dtype=np.float32)]
        actions = [[4, 5, 3], [6, 2]] if with_actions else None
        return Data([item_sequences, targets], groups, 2, 2.0,
                    lazy_graphs=True, action_sequences=actions)

    def test_chronological_gru_output_is_restored_to_reverse_item_positions(self):
        model = self.make_model().eval()
        re_actions = torch.tensor([[3, 5, 4, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        actual = model.encode_action_sequence(re_actions, mask)
        chronological = torch.tensor([[4, 5, 3]], dtype=torch.long)
        with torch.no_grad():
            expected_chrono, _ = model.action_gru(model.action_embedding(chronological))
        expected = torch.cat((expected_chrono[:, [2, 1, 0]],
                              torch.zeros(1, 1, model.emb_size)), dim=1)
        torch.testing.assert_close(actual, expected)

    def test_item_and_action_histories_reverse_pad_and_collate_together(self):
        batch = collate_fn([self.make_dataset()[0], self.make_dataset()[1]])
        self.assertEqual(len(batch), 16)
        self.assertEqual(batch[15].dtype, torch.int64)
        self.assertEqual(tuple(batch[15].shape), (2, 3))
        self.assertEqual(batch[15].tolist(), [[3, 5, 4], [2, 6, 0]])
        self.assertEqual(batch[2].tolist(), [[1, 1, 1], [1, 1, 0]])
        baseline = collate_fn([self.make_dataset(with_actions=False)[0]])
        self.assertEqual(len(baseline), 15)

    def test_action_forward_backward_has_finite_scores_and_action_gradients(self):
        torch.manual_seed(17)
        model = self.make_model()
        batch = collate_fn([self.make_dataset()[0], self.make_dataset()[1]])
        targets, scores, metadata_loss = forward(model, batch)
        self.assertEqual(tuple(scores.shape), (2, self.n_node + 1))
        self.assertTrue(torch.isfinite(scores).all())
        loss = torch.nn.functional.cross_entropy(scores, targets) + metadata_loss
        loss.backward()
        self.assertGreater(model.action_embedding.weight.grad[1:].abs().sum().item(), 0)
        self.assertGreater(model.action_gru.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.action_gru.weight_hh_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.W5_co.weight.grad[:, -model.emb_size:].abs().sum().item(), 0)
        self.assertGreater(model.W4_seq.weight.grad[:, -model.emb_size:].abs().sum().item(), 0)
        self.assertTrue(torch.equal(model.action_embedding.weight[0],
                                    torch.zeros_like(model.action_embedding.weight[0])))
        self.assertTrue(torch.equal(model.action_embedding.weight.grad[0],
                                    torch.zeros_like(model.action_embedding.weight.grad[0])))

    def test_cpu_model_stays_on_cpu_when_cuda_is_available(self):
        model = self.make_model().cpu()
        batch = collate_fn([self.make_dataset()[0], self.make_dataset()[1]])
        with patch("model.torch.cuda.is_available", return_value=True):
            targets, scores, _ = forward(model, batch)
        self.assertEqual(targets.device.type, "cpu")
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.isfinite(scores).all())

    def test_action_mode_rejects_batches_without_the_action_field(self):
        model = self.make_model()
        batch = collate_fn([self.make_dataset(with_actions=False)[0]])
        with self.assertRaisesRegex(ValueError, "item_event mode requires its action-history batch field"):
            forward(model, batch)

    def test_baseline_mode_has_original_two_part_event_projections(self):
        model = self.make_model(action_mode="none")
        self.assertEqual(model.W5_co.in_features, 2 * model.emb_size)
        self.assertEqual(model.W4_seq.in_features, 2 * model.emb_size)
        self.assertFalse(any("action_" in name for name in model.state_dict()))


class ActionSchemaTests(unittest.TestCase):
    def test_shared_action_vocabulary_is_stable_and_disjoint_from_padding(self):
        self.assertEqual(ACTION_TO_ID, {
            "clickout item": 1,
            "interaction item rating": 2,
            "interaction item info": 3,
            "interaction item image": 4,
            "interaction item deals": 5,
            "search for item": 6,
            "change of sort order": 7,
            "filter selection": 8,
            "search for destination": 9,
            "search for poi": 10,
        })
        self.assertEqual(len(set(ACTION_TO_ID.values())), len(ACTION_TO_ID))
        self.assertEqual((PAD_ACTION_ID, UNK_ACTION_ID, NUM_ACTIONS), (0, 11, 12))
        self.assertEqual(len(ITEM_ACTIONS), 6)
        self.assertEqual(ACTION_SCHEMA_VERSION, "trivago_item_event_actions_v1")
        self.assertEqual(ACTION_FUSION, "concat_post_graph_gru_v1")


if __name__ == "__main__":
    unittest.main()
