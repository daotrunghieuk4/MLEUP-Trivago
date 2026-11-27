"""All-ten-action adapter: causal inputs, train vocabulary and unified GRU."""
import csv
import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from prepare_trivago_top5 import prepare, DEFAULT_FIELD_BUDGET
from trivago_action_types import (
    ACTION_TO_ID, ALL_EVENT_ENCODER, ALL_EVENT_FUSION, NUM_ACTIONS, event_token,
)
from trivago_top5_checkpoint import checkpoint_model_arguments, validate_checkpoint

try:
    import torch
    from model import MLEUP, forward
    from util import Data, collate_fn
    from train_trivago_top5 import make_data, evaluate
except (ImportError, OSError):
    torch = None


def fixture(root, max_events=64, full_metadata=False, *,
            field_budget=DEFAULT_FIELD_BUDGET, action_scope='all_event', with_prices=False):
    metadata, train, out = root / 'item_metadata.csv', root / 'train.csv', root / 'prepared'
    with metadata.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('item_id', 'properties'))
        extra_properties = [f'Property {i:02d}' for i in range(field_budget - 2)] if full_metadata else []
        writer.writerows([
            ('A', '1 Star|Satisfactory Rating|Car Park' + ''.join('|' + p for p in extra_properties)),
            ('B', '2 Star|From 2 Stars|Satisfactory Rating|Good Rating|Gym'),
            ('C', '3 Star|From 2 Stars|From 3 Stars|Satisfactory Rating|Good Rating|Very Good Rating'),
            ('D', '4 Star|From 2 Stars|From 3 Stars|From 4 Stars|Satisfactory Rating|Good Rating|Very Good Rating|Excellent Rating'),
            ('E', '5 Star|From 2 Stars|From 3 Stars|From 4 Stars'),
        ])
    with train.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        header = ('user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions')
        writer.writerow((*header, 'prices') if with_prices else header)
        for i in range(5):
            hotel = 'D' if i == 1 else 'A'
            events = [(action, hotel, '') for action in (
                'interaction item image', 'interaction item info', 'interaction item rating',
                'interaction item deals', 'search for item',
            )]
            events += [('change of sort order', 'price', ''),
                       ('filter selection', 'Car Park', ''),
                       ('search for destination', 'Train City' if i < 3 else 'Heldout City', ''),
                       ('search for poi', 'A', ''),  # Same reference as a hotel, different type.
                       ('clickout item', 'B', 'A|B|C|D|E'),
                       ('filter selection', 'Gym', ''),
                       ('clickout item', 'C', 'A|B|C|D|E')]
            if i == 0 and full_metadata:
                events = [('filter selection', p, '') for p in extra_properties] + events
            # Session names deliberately collide across users.
            for step, (action, reference, impressions) in enumerate(events, 1):
                row = (f'u{i}', 'shared_session_id', i * 100 + step, step, action, reference, impressions)
                prices = '10|20|30|40|50' if impressions else ''
                writer.writerow((*row, prices) if with_prices else row)
    config = prepare(Namespace(
        train_csv=train, metadata_csv=metadata, output_dir=out, scratch_dir=root,
        field_budget=field_budget, max_neighbors=3, max_session_items=20, m=2, frequency=2,
        seed=2023, max_rows=0, action_scope=action_scope, max_session_events=max_events,
    ))
    return out, config


def load(path):
    with path.open('rb') as file:
        return pickle.load(file)


class AllEventPreparationTests(unittest.TestCase):
    def test_all_event_keeps_15_binary_plus_three_categorical_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, config = fixture(Path(tmp), full_metadata=True)
            self.assertEqual(config['field_budget'], 15)
            self.assertEqual(config['selected_binary_count'], 15)
            self.assertEqual(len(config['field_names']), 18)
            self.assertEqual(config['field_names'][:3], ['star_rating', 'from_star_max', 'rating_highest'])
            self.assertEqual(config['all_feature'], [6, 4, 5] + [2] * 15)
            self.assertEqual(load(out / 'adjacency_if.txt').shape, (18, 5))
            self.assertEqual(config['data_schema_version'], 4)

    def test_joint_keys_train_only_vocabulary_and_causal_query_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, config = fixture(Path(tmp))
            vocabulary = load(out / 'event_vocabulary.pkl')
            train, val = load(out / 'train.txt'), load(out / 'val.txt')
            self.assertEqual(config['data_schema_version'], 4)
            self.assertEqual(config['example_counts']['train_examples'], 6)
            self.assertEqual(config['example_counts']['val_examples'], 2)
            self.assertEqual(config['example_counts']['test_examples'], 2)
            self.assertGreater(config['group_pair_counts']['train']['real_pairs'], 0)
            self.assertEqual(len(train), 4)
            self.assertNotIn((9, 'Heldout City'), vocabulary)
            self.assertNotEqual(vocabulary[(3, 'A')], vocabulary[(4, 'A')])
            self.assertNotEqual(vocabulary[(10, 'A')], vocabulary[(4, 'A')])
            # First query sees nine actions, but not the current target B.
            self.assertEqual(len(train[3][0]), 9)
            self.assertNotIn(vocabulary[(1, 'B')], train[3][0])
            self.assertIn(vocabulary[(1, 'B')], train[3][1])
            self.assertEqual(train[3][1][-1], vocabulary[(8, 'Gym')])
            # Global actions never become item nodes. Training prior clickout does.
            self.assertEqual(train[0][0], [1] * 5)
            self.assertEqual(train[0][1], [1] * 5 + [2])
            self.assertEqual(val[0][1], [1] * 5)
            self.assertEqual(val[3][1][-2:], [1, vocabulary[(8, 'Gym')]])
            self.assertNotIn(vocabulary[(1, 'B')], val[3][1])
            self.assertIn(9, val[3][0])  # Heldout destination keeps type, not new vocabulary.
            self.assertEqual(config['num_event_tokens'], len(vocabulary) + 11)

    def test_event_history_budget_keeps_latest_global_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = fixture(Path(tmp), max_events=3)
            train = load(out / 'train.txt')
            vocabulary = load(out / 'event_vocabulary.pkl')
            self.assertEqual(train[3][1], [vocabulary[(10, 'A')], vocabulary[(1, 'B')],
                                         vocabulary[(8, 'Gym')]])
            self.assertEqual(len(train[0][1]), 6)  # Independent item and event budgets.

    def test_unknown_reference_preserves_action_and_namespaces(self):
        self.assertEqual(event_token({}, 'search for poi', 'unknown'), 10)
        self.assertEqual(event_token({(1, 'A'): 11}, 'clickout item', 'A', masked=True), 1)
        self.assertEqual(event_token({(1, 'A'): 11}, 'interaction item info', 'A'), 3)


@unittest.skipIf(torch is None, 'PyTorch is unavailable here')
class AllEventModelTests(unittest.TestCase):
    def make_model(self):
        options = SimpleNamespace(
            m=1, emb_size=4, batch_size=2, sample_order=2, num_heads=2,
            beta=0.3, alpha=0.2, metadata_projection='fieldwise', valid_groups_only=True,
            action_mode='all_event', num_actions=NUM_ACTIONS, num_event_tokens=31,
            action_encoder=ALL_EVENT_ENCODER, action_fusion=ALL_EVENT_FUSION,
            event_token_action_ids=list(range(11)) + list(range(1, 11)) * 2,
        )
        return MLEUP(options, 3, [2], [np.array([[1, 2], [3, 0]])], np.array([[0, 1, 0]]))

    def make_dataset(self):
        groups = [np.full((2, 1), 2), np.full((2, 1), 2), np.zeros((2, 1))]
        return Data([[[1, 2], [3]], [2, 1]], groups, 2, 2.0, lazy_graphs=True,
                    event_sequences=[list(range(11, 21)), [21, 22, 23]])

    def test_chronological_gru_zero_init_and_padding_invariance(self):
        torch.manual_seed(17)
        model = self.make_model().eval()
        tokens = torch.tensor([[11, 12, 13]])
        actual = model.encode_all_events(tokens, torch.ones_like(tokens))
        embeddings = model.event_embedding(tokens) + model.event_type_embedding(model.event_token_action_ids[tokens])
        _, expected = model.event_gru(embeddings)
        torch.testing.assert_close(actual, expected[-1])
        padded = torch.tensor([[11, 12, 13, 0, 0]])
        torch.testing.assert_close(actual, model.encode_all_events(padded, padded.ne(0)))
        changed = torch.tensor([[11, 12, 14]])
        self.assertFalse(torch.allclose(actual, model.encode_all_events(changed, changed.ne(0))))
        with self.assertRaises(ValueError):
            model.encode_all_events(torch.tensor([[11, 0, 12]]), torch.tensor([[1, 0, 1]]))

    def test_independent_masks_and_gradients_for_all_ten_tokens(self):
        torch.manual_seed(29)
        model = self.make_model()
        data = self.make_dataset()
        batch = collate_fn([data[0], data[1]])
        self.assertEqual(len(batch), 17)
        self.assertEqual(tuple(batch[2].shape), (2, 2))
        self.assertEqual(tuple(batch[15].shape), (2, 10))
        self.assertEqual(batch[15][1].tolist(), [21, 22, 23] + [0] * 7)
        targets, scores, metadata_loss = forward(model, batch)
        self.assertTrue(torch.isfinite(scores).all())
        (torch.nn.functional.cross_entropy(scores, targets) + metadata_loss).backward()
        for token in range(11, 21):
            self.assertGreater(model.event_embedding.weight.grad[token].abs().sum().item(), 0)
        for action_id in range(1, 11):
            self.assertGreater(model.event_type_embedding.weight.grad[action_id].abs().sum().item(), 0)
        for parameter in (model.event_gru.weight_ih_l0, model.event_gru.weight_hh_l0,
                          model.event_gate.weight, model.embedding_i.weight, model.W1_feature.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(model.event_embedding.weight.grad[0].abs().sum().item(), 0)
        self.assertFalse(hasattr(model, 'action_gru'))  # Not two competing action encoders.
        self.assertEqual(model.W5_co.in_features, 2 * model.emb_size)
        with self.assertRaises(ValueError):
            forward(model, batch[:15])

    def test_unseen_reference_uses_learned_type_not_random_pair(self):
        model = self.make_model().eval()
        unknown = torch.tensor([[7, 8, 9, 10]])
        actual = model.encode_all_events(unknown, torch.ones_like(unknown))
        _, expected = model.event_gru(model.event_type_embedding(unknown))
        torch.testing.assert_close(actual, expected[-1])
        with torch.no_grad():
            model.event_embedding.weight[1:11].fill_(999)
        torch.testing.assert_close(actual, model.encode_all_events(unknown, torch.ones_like(unknown)))

    def test_checkpoint_rejects_changed_event_vocabulary_size(self):
        args = SimpleNamespace(emb_size=4, batch_size=2, sample_order=2,
                               metadata_projection='fieldwise', num_heads=2, beta=.3,
                               alpha=.2, si_dc=2., action_mode='all_event', num_actions=12,
                               num_event_tokens=31, action_encoder=ALL_EVENT_ENCODER,
                               action_fusion=ALL_EVENT_FUSION)
        expected = checkpoint_model_arguments(args)
        checkpoint = {'signature': 'sig', 'model_arguments': expected.copy()}
        validate_checkpoint(checkpoint, 'sig', expected)
        checkpoint['model_arguments']['num_event_tokens'] = 32
        with self.assertRaisesRegex(ValueError, 'num_event_tokens'):
            validate_checkpoint(checkpoint, 'sig', expected)

    def test_prepared_data_to_full_model_and_top5_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, config = fixture(Path(tmp), full_metadata=True)
            options = SimpleNamespace(
                m=config['m'], emb_size=4, batch_size=2, sample_order=2, num_heads=2,
                beta=.3, alpha=.2, si_dc=2., metadata_projection='fieldwise', valid_groups_only=True,
                action_mode='all_event', num_actions=12, num_event_tokens=config['num_event_tokens'],
                action_encoder=ALL_EVENT_ENCODER, action_fusion=ALL_EVENT_FUSION,
            )
            vocabulary = load(out / 'event_vocabulary.pkl')
            options.event_token_action_ids = list(range(11)) + [0] * len(vocabulary)
            for (action_id, _), token in vocabulary.items():
                options.event_token_action_ids[token] = action_id
            model = MLEUP(options, config['n_node'], config['all_feature'],
                          load(out / 'adjacency_fi.txt'), load(out / 'adjacency_if.txt'))
            self.assertEqual(model.n_feature, 18)
            train = make_data(out, 'train', config, options)
            targets, scores, s_loss = forward(model, collate_fn([train[4], train[5]]))
            (torch.nn.functional.cross_entropy(scores, targets) + s_loss).backward()
            self.assertTrue(torch.isfinite(model.event_gate.weight.grad).all())
            val = make_data(out, 'val', config, options)
            metrics = evaluate(model, val, load(out / 'val_candidates.txt'), 2)
            self.assertEqual(set(metrics), {'ndcg_at_5', 'recall_at_5', 'examples'})
            self.assertEqual(metrics['examples'], 2)
            self.assertTrue(0 <= metrics['ndcg_at_5'] <= metrics['recall_at_5'] <= 1)


if __name__ == '__main__':
    unittest.main()
