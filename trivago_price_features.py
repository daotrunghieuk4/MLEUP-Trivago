"""Versioned, train-only price features for Trivago impression ranking."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


PRICE_FEATURE_SCHEMA = "relative_price_v1"
PRICE_ARTIFACT_SCHEMA = "context_price_v1"  # Keep the persisted schema for existing prepared data.
PRICE_MODE = "relative_mlp_v1"
PRICE_FEATURE_ORDER = (
    "log1p_price_z", "price_percentile_rank", "log_price_over_median",
    "price_minus_min_over_median", "list_iqr_over_median",
    "valid_price_count", "valid_price_rate", "price_valid",
    "relative_price_available",
)
PRICE_EPS = 1e-8


def parse_prices(raw_prices: str | None, expected_length: int) -> tuple[list[float | None], bool]:
    """Parse impression prices, invalidating the whole vector on length mismatch."""
    if raw_prices is None or not raw_prices.strip():
        return [None] * expected_length, False
    tokens = (raw_prices or "").split("|") if raw_prices else []
    if len(tokens) != expected_length:
        return [None] * expected_length, bool(tokens or expected_length)
    result: list[float | None] = []
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError):
            result.append(None)
            continue
        result.append(value if math.isfinite(value) and value > 0 else None)
    return result, False


def normalize_candidate_records(
    item_ids: Sequence[int], prices: Sequence[float | None],
    raw_positions: Sequence[int], target: int | None = None,
) -> list[tuple[int, int, float | None]]:
    """Filter unknown catalog IDs and deduplicate by first raw impression."""
    if not (len(item_ids) == len(prices) == len(raw_positions)):
        raise ValueError("Candidate IDs, prices and raw positions must align")
    records: list[tuple[int, int, float | None]] = []
    seen: set[int] = set()
    for item_id, price, raw_position in zip(item_ids, prices, raw_positions):
        item_id = int(item_id)
        if item_id <= 0 or item_id in seen:
            continue
        seen.add(item_id)
        valid_price = (float(price) if price is not None and math.isfinite(float(price))
                       and float(price) > 0 else None)
        records.append((item_id, int(raw_position), valid_price))
    if not records:
        raise ValueError("Candidate records are empty after catalog filtering")
    if target is not None and sum(record[0] == int(target) for record in records) != 1:
        raise ValueError("Target must occur exactly once in normalized candidates")
    return records


def fit_price_scaler(price_rows: Iterable[Sequence[float | None]]) -> dict:
    """Fit mean/std of log1p prices on valid training candidates only."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for row in price_rows:
        for price in row:
            if price is None:
                continue
            value = float(price)
            if not math.isfinite(value) or value <= 0:
                continue
            transformed = math.log1p(value)
            count += 1
            delta = transformed - mean
            mean += delta / count
            m2 += delta * (transformed - mean)
    std = math.sqrt(m2 / count) if count else 1.0
    return {"schema": PRICE_FEATURE_SCHEMA, "count": count,
            "log1p_mean": mean, "log1p_std": std if std > 1e-12 else 1.0,
            "eps": PRICE_EPS, "invalid_policy": "nonfinite_or_nonpositive_missing_v1"}


def build_price_features(prices: Sequence[float | None], scaler: dict) -> np.ndarray:
    """Build candidate-local relative statistics plus train-standardized log price."""
    if scaler.get("schema") != PRICE_FEATURE_SCHEMA:
        raise ValueError("Unsupported price feature scaler schema")
    if scaler.get("eps") != PRICE_EPS:
        raise ValueError("Price feature epsilon differs from the versioned policy")
    values = [float(price) if price is not None and math.isfinite(float(price))
              and float(price) > 0 else None for price in prices]
    valid_indices = [index for index, value in enumerate(values) if value is not None]
    valid = np.asarray([values[index] for index in valid_indices], dtype=np.float64)
    candidate_count = len(values)
    valid_count = len(valid_indices)
    valid_rate = valid_count / candidate_count if candidate_count else 0.0
    relative_available = valid_count >= 2
    output = np.zeros((candidate_count, len(PRICE_FEATURE_ORDER)), dtype=np.float32)
    if not candidate_count:
        return output

    output[:, 5] = valid_count
    output[:, 6] = valid_rate
    if valid_count:
        mean = float(scaler["log1p_mean"])
        std = max(float(scaler["log1p_std"]), 1e-12)
        for index in valid_indices:
            output[index, 0] = (math.log1p(values[index]) - mean) / std
            output[index, 7] = 1.0
    if relative_available:
        median = float(np.median(valid))
        minimum = float(valid.min())
        q25, q75 = np.quantile(valid, (0.25, 0.75), method="linear")
        dispersion = (float(q75) - float(q25)) / (median + PRICE_EPS)
        ordered = sorted((values[index], index) for index in valid_indices)
        rank_start = 0
        while rank_start < valid_count:
            rank_end = rank_start + 1
            while (rank_end < valid_count
                   and ordered[rank_end][0] == ordered[rank_start][0]):
                rank_end += 1
            average_zero_based_rank = (rank_start + rank_end - 1) / 2.0
            percentile = average_zero_based_rank / (valid_count - 1)
            for _, index in ordered[rank_start:rank_end]:
                value = values[index]
                output[index, 1] = percentile
                output[index, 2] = math.log((value + PRICE_EPS) / (median + PRICE_EPS))
                output[index, 3] = (value - minimum) / (median + PRICE_EPS)
                output[index, 4] = dispersion
                output[index, 8] = 1.0
            rank_start = rank_end
    if not np.isfinite(output).all():
        raise FloatingPointError("Price feature builder produced non-finite values")
    return output


def flatten_candidate_records(rows: Sequence[Sequence[tuple[int, int, float | None]]]):
    """Compact query records into offsets and aligned ID/position/price arrays."""
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(row)
    total = int(offsets[-1])
    item_ids = np.empty(total, dtype=np.int32)
    raw_positions = np.empty(total, dtype=np.int32)
    prices = np.full(total, np.nan, dtype=np.float64)
    cursor = 0
    for row in rows:
        for item_id, position, price in row:
            item_ids[cursor] = item_id
            raw_positions[cursor] = position
            if price is not None:
                prices[cursor] = price
            cursor += 1
    return {"offsets": offsets, "item_ids": item_ids,
            "raw_positions": raw_positions, "prices": prices}


def candidate_price_row(sidecar: dict, index: int) -> list[float | None]:
    """Return a query's prices from the compact shared candidate record."""
    start, end = map(int, sidecar["offsets"][index:index + 2])
    if end <= start:
        raise ValueError(f"Query {index} has no candidate records")
    result = []
    for value in sidecar["prices"][start:end]:
        number = float(value)
        result.append(number if math.isfinite(number) and number > 0 else None)
    return result
