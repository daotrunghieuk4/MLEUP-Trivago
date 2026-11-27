"""Preparation and leakage checks for train-filter-selected Trivago metadata."""

import csv
import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, sentinel

from src.prepare_trivago_top5 import build_groups, fingerprint_file, prepare
from src.trivago_top5_checkpoint import (
    checkpoint_model_arguments, restore_checkpoint_rng, validate_checkpoint,
    best_validation_ndcg,
)


class CheckpointTests(unittest.TestCase):
    def test_best_score_tracks_ndcg_not_legacy_mrr(self):
        self.assertEqual(best_validation_ndcg({"best_mrr": 0.8}), -1.0)
        self.assertEqual(best_validation_ndcg({
            "selection_metric": "ndcg_at_20", "best_ndcg_at_5": 0.8,
        }), -1.0)
        self.assertEqual(best_validation_ndcg({
            "selection_metric": "ndcg_at_5", "best_ndcg_at_5": 0.4,
        }), 0.4)

    def setUp(self):
        self.signature = ("items", "schema", "data")
        self.arguments = checkpoint_model_arguments(Namespace(
            emb_size=16, batch_size=1, sample_order=3,
            metadata_projection="concat", num_heads=4, beta=0.3,
            alpha=0.2, si_dc=2.0, action_mode="none", num_actions=0,
            action_encoder="none", action_fusion="none",
        ))

    def test_matching_configuration_is_accepted(self):
        validate_checkpoint({"signature": self.signature,
                             "model_arguments": dict(self.arguments)},
                            self.signature, self.arguments)

    def test_changed_or_missing_arguments_and_data_are_rejected(self):
        for name, value in self.arguments.items():
            with self.subTest(argument=name):
                altered = dict(self.arguments)
                altered[name] = "fieldwise" if isinstance(value, str) else value + 1
                with self.assertRaisesRegex(ValueError, name):
                    validate_checkpoint({"signature": self.signature,
                                         "model_arguments": altered},
                                        self.signature, self.arguments)
                del altered[name]
                with self.assertRaisesRegex(ValueError, name):
                    validate_checkpoint({"signature": self.signature,
                                         "model_arguments": altered},
                                        self.signature, self.arguments)
        with self.assertRaisesRegex(ValueError, "prepared data"):
            validate_checkpoint({"signature": ("different data",),
                                 "model_arguments": self.arguments},
                                self.signature, self.arguments)

    def test_old_checkpoint_without_configuration_is_rejected(self):
        legacy = {"signature": self.signature,
                  **{name: self.arguments[name] for name in (
                      "emb_size", "batch_size", "sample_order", "metadata_projection",
                  )}}
        with self.assertRaisesRegex(ValueError, "older trainer"):
            validate_checkpoint(legacy, self.signature, self.arguments)

    def test_rng_states_are_moved_to_cpu_before_restoring(self):
        for device_type in ("cpu", "cuda"):
            with self.subTest(device=device_type):
                cpu_rng = Mock()
                cpu_rng.cpu.return_value = sentinel.cpu_rng
                cuda_rng = Mock()
                cuda_rng.cpu.return_value = sentinel.cuda_rng
                torch_api = Mock()
                restore_checkpoint_rng(
                    {"cpu_rng": cpu_rng, "cuda_rng": [cuda_rng]},
                    Namespace(type=device_type), torch_api,
                )
                cpu_rng.cpu.assert_called_once_with()
                torch_api.set_rng_state.assert_called_once_with(sentinel.cpu_rng)
                if device_type == "cuda":
                    cuda_rng.cpu.assert_called_once_with()
                    torch_api.cuda.set_rng_state_all.assert_called_once_with(
                        [sentinel.cuda_rng])
                else:
                    cuda_rng.cpu.assert_not_called()
                    torch_api.cuda.set_rng_state_all.assert_not_called()


class PreparationTests(unittest.TestCase):
    def test_train_only_field_selection_and_metadata_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "item_metadata.csv"
            train = root / "train.csv"
            out = root / "prepared"
            items = [
                ("A", "1 Star|Satisfactory Rating|Car Park"),
                ("B", "2 Star|From 2 Stars|Satisfactory Rating|Good Rating|Gym"),
                ("C", "3 Star|From 2 Stars|From 3 Stars|Satisfactory Rating|Good Rating|Very Good Rating|Pet Friendly"),
                ("D", "4 Star|From 2 Stars|From 3 Stars|From 4 Stars|Satisfactory Rating|Good Rating|Very Good Rating|Excellent Rating"),
                ("E", "5 Star|From 2 Stars|From 3 Stars|From 4 Stars|Car Park"),
            ]
            with metadata.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("item_id", "properties"))
                writer.writeheader()
                for item_id, properties in items:
                    writer.writerow({"item_id": item_id, "properties": properties})
            columns = ("user_id", "session_id", "timestamp", "step",
                       "action_type", "reference", "impressions")
            with train.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for i, filter_name in enumerate(("Car Park", "Car Park", "Gym",
                                                  "Pet Friendly", "Pet Friendly")):
                    common = {"user_id": f"u{i}", "session_id": f"s{i}"}
                    base = i * 10
                    writer.writerow({**common, "timestamp": base + 1, "step": 1,
                                     "action_type": "interaction item image",
                                     "reference": "A", "impressions": ""})
                    writer.writerow({**common, "timestamp": base + 2, "step": 2,
                                     "action_type": "interaction item deals",
                                     "reference": "B", "impressions": ""})
                    writer.writerow({**common, "timestamp": base + 3, "step": 3,
                                     "action_type": "filter selection",
                                     "reference": filter_name, "impressions": ""})
                    writer.writerow({**common, "timestamp": base + 4, "step": 4,
                                     "action_type": "clickout item",
                                     "reference": "B", "impressions": "A|B"})
                    writer.writerow({**common, "timestamp": base + 5, "step": 5,
                                     "action_type": "clickout item",
                                     "reference": "A", "impressions": "A|B"})
            config = prepare(Namespace(
                train_csv=train, metadata_csv=metadata, output_dir=out,
                scratch_dir=root, field_budget=50, max_neighbors=3,
                max_session_items=20, m=2, frequency=2, seed=2023,
                max_rows=0,
            ))
            self.assertEqual(config["selected_binary_properties"], ["Car Park", "Gym"])
            self.assertEqual(config["selected_binary_count"], 2)
            self.assertEqual(len(config["field_names"]), 5)
            self.assertEqual(config["session_counts"],
                             {"train": 3, "val": 1, "test": 1})
            self.assertEqual(config["example_counts"]["train_examples"], 6)
            self.assertEqual(config["example_counts"]["val_examples"], 2)
            self.assertEqual(config["example_counts"]["test_examples"], 2)
            self.assertEqual(config["converter_version"],
                             "top5_metadata_v3_item_event_actions")
            self.assertEqual(config["data_schema_version"], 3)
            self.assertEqual(config["num_actions"], 12)
            with (out / "train.txt").open("rb") as file:
                train_payload = pickle.load(file)
            self.assertEqual(len(train_payload), 3)
            self.assertEqual(train_payload[0][0], [1, 2])
            self.assertEqual(train_payload[2][0], [4, 5])
            self.assertEqual(train_payload[0][1], [1, 2, 2])
            self.assertEqual(train_payload[2][1], [4, 5, 1])
            with (out / "val.txt").open("rb") as file:
                val_payload = pickle.load(file)
            self.assertEqual(val_payload[0][1], [1, 2])
            self.assertEqual(val_payload[2][1], [4, 5])
            schema = json.loads((out / "metadata_schema.json").read_text())
            self.assertEqual(schema["all_feature"], [6, 4, 5, 2, 2])
            with (out / "adjacency_if.txt").open("rb") as file:
                adjacency = pickle.load(file)
            self.assertEqual(adjacency.shape, (5, 5))
            self.assertEqual(adjacency[:, 0].tolist(), [1, 0, 1, 1, 0])
            self.assertEqual(adjacency[:, 2].tolist(), [3, 2, 3, 0, 0])
            for name, expected in config["artifact_sha256"].items():
                self.assertEqual(fingerprint_file(out / name), expected)
            self.assertEqual(config["group_pair_counts"]["train"]["real_pairs"], 0)

    def test_groups_only_reference_real_prior_examples(self):
        examples = [
            ([1, 3], 2, 10, 0, [1, 2]),
            ([4, 5], 2, 20, 1, [2, 4]),
            ([1, 2], 3, 30, 2, [1, 3], [4, 5]),
        ]
        positive, negative, _ = build_groups(examples, m=2, frequency=2)
        self.assertEqual(positive[2, 0], 0)
        self.assertEqual(negative[2, 0], 1)
        self.assertEqual(positive[2, 1], len(examples))
        self.assertEqual(negative[2, 1], len(examples))
        for index in range(len(examples)):
            self.assertTrue(all(group < index for group in positive[index]
                                if group < len(examples)))
            self.assertTrue(all(group < index for group in negative[index]
                                if group < len(examples)))


if __name__ == "__main__":
    unittest.main()
