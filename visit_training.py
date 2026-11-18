"""Train the item-only MLEUP encoder on chronological visit events."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import MLEUP, forward
from prepare_trivago_top5 import build_groups
from train_trivago_top5 import batch_at, evaluate
from util import Data


def prepare_visits(events, validation_start, test_start):
    frame = events[["sequence", "destination", "event_time"]].copy()
    if frame.isna().any().any():
        raise ValueError("Sequence, destination and event time must be present on every row.")
    for column in ("sequence", "destination"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Blank {column} values are not supported.")
    if pd.api.types.is_numeric_dtype(frame["event_time"]):
        raise ValueError("Convert numeric times to UTC timestamps before training.")
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    if frame["event_time"].isna().any():
        raise ValueError("Event time must be a valid UTC timestamp.")
    frame = frame.drop_duplicates().sort_values("event_time", kind="stable")
    if frame.duplicated(["sequence", "event_time"]).any():
        raise ValueError("Multiple destinations share a timestamp within a sequence. Resolve event order first.")
    val_at, test_at = pd.to_datetime([validation_start, test_start], utc=True)
    if pd.isna(val_at) or pd.isna(test_at) or val_at >= test_at:
        raise ValueError("Supply validation_start and test_start in chronological order.")

    train_items = sorted(frame.loc[frame.event_time < val_at, "destination"].unique())
    item_ids = {item: i for i, item in enumerate(train_items, start=1)}
    if len(item_ids) < 2:
        raise ValueError("Train needs at least two distinct destinations for ranking.")

    splits = {"train": [], "validation": [], "test": []}
    skipped = {name: {"no_known_history": 0, "unknown_target": 0} for name in splits}
    for _, sequence in frame.groupby("sequence", sort=False):
        history = deque(maxlen=20)
        for row in sequence.itertuples(index=False):
            time = row.event_time
            split = "train" if time < val_at else "validation" if time < test_at else "test"
            target = item_ids.get(row.destination, 0)
            known_history = [item for item in history if item]
            if not target:
                skipped[split]["unknown_target"] += 1
            elif not known_history:
                skipped[split]["no_known_history"] += 1
            else:
                # build_groups consumes (prefix, target, unix seconds, query, candidates).
                splits[split].append((known_history, target, time.timestamp(), None, None))
            history.append(target)
    for name, examples in splits.items():
        if not examples:
            raise ValueError(f"No {name} examples with known destinations and past visits. Check IDs and cutoffs.")
        examples.sort(key=lambda example: example[2])
    return splits, item_ids, skipped


def train_visits(events, validation_start, test_start, epochs=2, batch_size=32):
    if min(epochs, batch_size) < 1:
        raise ValueError("Epochs and batch size must be positive.")
    splits, item_ids, skipped = prepare_visits(events, validation_start, test_start)
    options = SimpleNamespace(
        m=3, emb_size=16, batch_size=batch_size, sample_order=3, num_heads=4,
        beta=0.0, alpha=0.2, si_dc=2.0, action_mode="none", price_mode="off",
        valid_groups_only=True,
    )
    datasets = {
        split: Data(
            [[example[0] for example in examples], [example[1] for example in examples]],
            build_groups(examples, m=options.m, frequency=50),
            options.sample_order, options.si_dc, lazy_graphs=True,
        )
        for split, examples in splits.items()
    }
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLEUP(options, len(item_ids), [], [], []).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    catalog = list(range(1, len(item_ids) + 1))
    best_metrics = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train = datasets["train"]
        order = rng.permutation(len(train)).tolist()
        for start in range(0, len(train), batch_size):
            target, score, _ = forward(model, batch_at(train, order[start:start + batch_size]))
            loss = F.cross_entropy(score[:, 1:], target - 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = datasets["validation"]
        metrics = evaluate(model, val, [catalog] * len(val), batch_size)
        history.append({"epoch": epoch, **metrics})
        if best_metrics is None or metrics["ndcg_at_5"] > best_metrics["ndcg_at_5"]:
            best_metrics = metrics
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    test = datasets["test"]
    metrics = {
        "device": str(device),
        "catalog_size": len(item_ids),
        "examples": {name: len(dataset) for name, dataset in datasets.items()},
        "skipped": skipped,
        "history": history,
        "best_epoch": best_epoch,
        "test": evaluate(model, test, [catalog] * len(test), batch_size),
    }
    checkpoint = {"model_state": {name: value.cpu() for name, value in best_state.items()},
                  "options": vars(options), "item_ids": item_ids}
    return metrics, checkpoint
