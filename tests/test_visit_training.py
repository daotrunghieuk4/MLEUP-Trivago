import unittest

import pandas as pd
import torch

from visit_training import prepare_visits, train_visits


def visit_fixture():
    rows = []
    for user in range(6):
        for day in range(9):
            rows.append((f"test-{user}", str((user + day) % 6),
                         pd.Timestamp("2025-01-01") + pd.Timedelta(days=day, minutes=user)))
    rows.append(("test-0", "unseen", pd.Timestamp("2025-01-09 12:00")))
    return pd.DataFrame(rows, columns=["sequence", "destination", "event_time"])


class VisitTrainingTest(unittest.TestCase):
    def test_chronological_split_and_train_catalog(self):
        splits, ids, skipped = prepare_visits(visit_fixture(), "2025-01-05", "2025-01-07")
        self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")], [18, 12, 18])
        self.assertNotIn("unseen", ids)
        self.assertEqual(skipped["test"]["unknown_target"], 1)
        self.assertEqual(splits["validation"][0][:2], ([ids[str(i)] for i in range(4)], ids["4"]))

    def test_train_and_evaluate_without_metadata(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            metrics, checkpoint = train_visits(visit_fixture(), "2025-01-05", "2025-01-07",
                                               epochs=2, batch_size=8)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertIn(metrics["best_epoch"], (1, 2))
        self.assertEqual(metrics["test"]["examples"], 18)
        self.assertTrue(0 <= metrics["test"]["ndcg_at_5"] <= 1)
        self.assertEqual(checkpoint["options"]["action_mode"], "none")
        self.assertEqual(checkpoint["options"]["price_mode"], "off")
        self.assertIn("embedding_i.weight", checkpoint["model_state"])


if __name__ == "__main__":
    unittest.main()
