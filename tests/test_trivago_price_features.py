"""Unit checks for the versioned relative price feature contract."""

import math
import unittest

import numpy as np

from trivago_price_features import (
    PRICE_FEATURE_ORDER, PRICE_FEATURE_SCHEMA, build_price_features,
    fit_price_scaler, flatten_candidate_records, normalize_candidate_records,
    parse_prices,
)


class PriceFeatureTests(unittest.TestCase):
    def test_parser_invalidates_whole_list_on_length_mismatch(self):
        prices, mismatch = parse_prices("12|invalid", 3)
        self.assertTrue(mismatch)
        self.assertEqual(prices, [None, None, None])

    def test_parser_distinguishes_absent_prices_from_length_mismatch(self):
        prices, mismatch = parse_prices("", 3)
        self.assertFalse(mismatch)
        self.assertEqual(prices, [None, None, None])

    def test_parser_accepts_only_finite_positive_prices(self):
        prices, mismatch = parse_prices("1|0|-2|NaN|Infinity|bad", 6)
        self.assertFalse(mismatch)
        self.assertEqual(prices, [1.0, None, None, None, None, None])

    def test_first_candidate_record_and_raw_position_are_preserved(self):
        rows = normalize_candidate_records(
            [4, 0, 2, 4], [None, 50.0, 25.0, 10.0], [0, 1, 2, 3], target=2,
        )
        self.assertEqual(rows, [(4, 0, None), (2, 2, 25.0)])
        packed = flatten_candidate_records([rows])
        self.assertEqual(packed["item_ids"].tolist(), [4, 2])
        self.assertEqual(packed["raw_positions"].tolist(), [0, 2])
        self.assertTrue(np.isnan(packed["prices"][0]))
        self.assertEqual(packed["prices"][1], 25.0)

    def test_rank_ratio_and_spread_are_scale_invariant(self):
        scaler = fit_price_scaler([[10.0, 20.0, 20.0, 80.0]])
        first = build_price_features([10.0, 20.0, 20.0, 80.0], scaler)
        scaled = build_price_features([100.0, 200.0, 200.0, 800.0], scaler)
        np.testing.assert_allclose(first[:, 1:5], scaled[:, 1:5], atol=1e-7)
        self.assertEqual(tuple(PRICE_FEATURE_ORDER), (
            "log1p_price_z", "price_percentile_rank", "log_price_over_median",
            "price_minus_min_over_median", "list_iqr_over_median", "valid_price_count",
            "valid_price_rate", "price_valid", "relative_price_available",
        ))
        self.assertEqual(PRICE_FEATURE_SCHEMA, "relative_price_v1")

    def test_average_rank_ties_and_single_price_neutrality(self):
        scaler = fit_price_scaler([[5.0, 5.0, 20.0]])
        tied = build_price_features([5.0, 5.0, 20.0], scaler)
        np.testing.assert_allclose(tied[:2, 1], [0.25, 0.25])
        single = build_price_features([5.0, None], scaler)
        np.testing.assert_allclose(single[:, 1:5], 0.0)
        np.testing.assert_allclose(single[:, 8], 0.0)
        self.assertTrue(np.isfinite(single).all())

    def test_empty_or_invalid_prices_produce_finite_neutral_features(self):
        scaler = fit_price_scaler([[10.0]])
        features = build_price_features([None, -1.0, math.inf], scaler)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features, 0.0)


if __name__ == "__main__":
    unittest.main()
