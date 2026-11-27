"""Prepare Trivago for MLEUP with train-filter-selected metadata fields.

The selection rule follows HUONG_DAN_TOI_UU_BO_NHO_METADATA_MLEUP.md:
Copy 3 keeps three categorical fields plus at most 15 binary hotel-property fields. Only
``filter selection`` events in the temporal *train* sessions choose fields.
The original MLEUP graph/metadata inputs are then rebuilt for those fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from .trivago_action_types import (
    ACTION_HISTORY_POLICY, ACTION_INPUT_SCOPE, ACTION_SCHEMA_VERSION,
    ACTION_TO_ID, ITEM_ACTIONS, NUM_ACTIONS, PAD_ACTION_ID, UNK_ACTION_ID,
    all_event_schema, event_token,
)
from .trivago_price_features import (
    PRICE_ARTIFACT_SCHEMA, PRICE_FEATURE_ORDER, PRICE_FEATURE_SCHEMA, fit_price_scaler,
    flatten_candidate_records, normalize_candidate_records, parse_prices,
)


STAR = ("1 Star", "2 Star", "3 Star", "4 Star", "5 Star")
FROM_STAR = ("From 2 Stars", "From 3 Stars", "From 4 Stars")
RATING = ("Satisfactory Rating", "Good Rating", "Very Good Rating", "Excellent Rating")
SPECIAL = set(STAR + FROM_STAR + RATING)
CAT_FIELDS = ("star_rating", "from_star_max", "rating_highest")
DEFAULT_FIELD_BUDGET = 15
SPLITS = ("train", "val", "test")


def dump_pickle(value, path: Path) -> None:
    with path.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def property_set(raw: str | None) -> set[str]:
    return {part.strip() for part in (raw or "").split("|") if part.strip()}


def scan_catalog(path: Path) -> tuple[dict[str, int], list[str], str]:
    item_map = {}
    vocabulary = set()
    mapping_digest = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"item_id", "properties"}.issubset(reader.fieldnames or []):
            raise ValueError("item_metadata.csv needs item_id and properties")
        for row in reader:
            raw_id = row["item_id"].strip()
            if not raw_id or raw_id in item_map:
                raise ValueError(f"Empty or duplicate item_id: {raw_id!r}")
            item_map[raw_id] = len(item_map) + 1  # 0 is reserved for padding.
            mapping_digest.update(raw_id.encode("utf-8") + b"\n")
            vocabulary.update(property_set(row["properties"]))
    missing = SPECIAL - vocabulary
    if missing:
        raise ValueError(f"Missing expected categorical labels: {sorted(missing)}")
    binary_names = sorted(vocabulary - SPECIAL)
    if not binary_names:
        raise ValueError("No binary hotel properties in metadata")
    return item_map, binary_names, mapping_digest.hexdigest()


def read_train(path: Path, max_rows: int = 0):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"user_id", "session_id", "timestamp", "step",
                    "action_type", "reference", "impressions"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"train.csv missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 1):
            if max_rows and row_number > max_rows:
                break
            yield row_number, row


def split_sessions(train_csv: Path, output_dir: Path,
                   max_rows: int) -> tuple[dict[tuple[str, str], str], dict]:
    end_time = {}
    for row_number, row in read_train(train_csv, max_rows):
        key = (row["user_id"], row["session_id"])
        timestamp = int(row["timestamp"])
        if timestamp > end_time.get(key, -1):
            end_time[key] = timestamp
        if row_number % 1_000_000 == 0:
            print(f"Session scan: {row_number:,} rows", flush=True)
    ordered = sorted(end_time, key=lambda key: (end_time[key], key[0], key[1]))
    n = len(ordered)
    train_end, val_end = int(n * 0.60), int(n * 0.80)
    split_map = {}
    with (output_dir / "session_split.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("user_id", "session_id", "session_end_timestamp", "split"))
        for index, key in enumerate(ordered):
            split = "train" if index < train_end else (
                "val" if index < val_end else "test"
            )
            split_map[key] = split
            writer.writerow((key[0], key[1], end_time[key], split))
    return split_map, {
        "session_count": n,
        "session_counts": {"train": train_end, "val": val_end - train_end,
                           "test": n - val_end},
        "train_cutoff": end_time[ordered[train_end - 1]] if train_end else None,
        "val_cutoff": end_time[ordered[val_end - 1]] if val_end else None,
    }


def collect_events_and_filters(
    train_csv: Path, db_path: Path, split_map: dict,
    item_map: dict[str, int], binary_names: list[str], max_rows: int,
    action_scope: str = "item_event",
) -> tuple[sqlite3.Connection, Counter, Counter]:
    """Count train-only filters while externally sorting hotel events on disk."""
    valid_names = set(binary_names)
    counts, unmatched = Counter(), Counter()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE events (user_id TEXT, session_id TEXT, ts INTEGER, "
        "step INTEGER, row_number INTEGER, action TEXT, item_id INTEGER, "
        "impressions TEXT, reference TEXT, prices TEXT)"
    )
    connection.execute(
        "CREATE TABLE session_boundary_events (user_id TEXT, session_id TEXT, ts INTEGER)"
    )
    batch = []
    boundary_batch = []
    for row_number, row in read_train(train_csv, max_rows):
        boundary_batch.append((row["user_id"], row["session_id"], int(row["timestamp"])))
        if len(boundary_batch) >= 20_000:
            connection.executemany(
                "INSERT INTO session_boundary_events VALUES (?,?,?)", boundary_batch,
            )
            connection.commit()
            boundary_batch.clear()
        action = row["action_type"].strip().lower()
        if action not in ACTION_TO_ID:
            raise ValueError(
                f"Unknown action_type at train.csv row {row_number}: {action!r}. "
                "Add an explicit ID to trivago_action_types.py before preparing data."
            )
        split = split_map[(row["user_id"], row["session_id"])]
        if action == "filter selection" and split == "train":
            name = (row["reference"] or "").strip()
            if name in valid_names:
                counts[name] += 1
            else:
                unmatched[name] += 1
        is_clickout = action == "clickout item"
        is_item_action = action in ITEM_ACTIONS and action != "clickout item"
        if action_scope != "all_event" and not (is_clickout or is_item_action):
            continue
        reference = (row["reference"] or "").strip()
        item_id = item_map.get(reference, 0) if action in ITEM_ACTIONS else 0
        if action_scope != "all_event" and not is_clickout and not item_id:
            continue
        batch.append((row["user_id"], row["session_id"], int(row["timestamp"]),
                      int(row["step"]), row_number, action, item_id,
                      row["impressions"] if is_clickout else "", reference,
                      (row.get("prices") or "") if is_clickout else ""))
        if len(batch) >= 20_000:
            connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            connection.commit()
            batch.clear()
        if row_number % 1_000_000 == 0:
            print(f"Event/filter scan: {row_number:,} rows", flush=True)
    if batch:
        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        connection.commit()
    if boundary_batch:
        connection.executemany(
            "INSERT INTO session_boundary_events VALUES (?,?,?)", boundary_batch,
        )
        connection.commit()
    connection.execute(
        "CREATE INDEX chronological_events ON events(user_id, ts, step, row_number)"
    )
    connection.execute(
        "CREATE INDEX chronological_session_boundaries "
        "ON session_boundary_events(user_id, session_id, ts)"
    )
    connection.commit()
    return connection, counts, unmatched


def choose_fields(binary_names: list[str], counts: Counter, unmatched: Counter,
                  budget: int, output_dir: Path) -> list[str]:
    ranked = sorted((name for name in binary_names if counts[name] > 0),
                    key=lambda name: (-counts[name], name))
    selected = ranked[:budget]
    if not selected:
        raise ValueError("No train filter-selection reference matches a binary hotel property")
    with (output_dir / "metadata_filter_counts.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("property", "train_filter_selection_count", "rank", "selected"))
        rank = {name: position + 1 for position, name in enumerate(ranked)}
        for name in sorted(binary_names, key=lambda value: (-counts[value], value)):
            writer.writerow((name, counts[name], rank.get(name, ""), int(name in selected)))
    with (output_dir / "unmatched_filter_references.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("reference", "train_filter_selection_count"))
        writer.writerows(sorted(unmatched.items(), key=lambda pair: (-pair[1], pair[0])))
    return selected


def cumulative_code(props: set[str], labels: tuple[str, ...], item_id: str) -> int:
    present = tuple(label for label in labels if label in props)
    if present != labels[:len(present)]:
        raise ValueError(f"item_id={item_id}: non-cumulative labels {present}")
    return len(present)


def field_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return "has_" + slug


def build_metadata(metadata_csv: Path, output_dir: Path,
                   item_map: dict[str, int], selected: list[str],
                   max_neighbors: int, seed: int) -> tuple[list[int], list[str], str]:
    field_names = list(CAT_FIELDS) + [field_slug(name) for name in selected]
    if len(field_names) != len(set(field_names)):
        raise ValueError("Property-name slug collision in selected fields")
    cardinalities = [6, 4, 5] + [2] * len(selected)
    n_items = len(item_map)
    adjacency_if = np.zeros((len(field_names), n_items), dtype=np.uint8)
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as source, (
        output_dir / "item_metadata_selected.csv"
    ).open("w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(source)
        writer = csv.writer(output)
        writer.writerow(["item_id"] + field_names)
        for row_number, row in enumerate(reader, 1):
            raw_id = row["item_id"].strip()
            item_index = item_map[raw_id] - 1
            props = property_set(row["properties"])
            stars = [position + 1 for position, name in enumerate(STAR) if name in props]
            if len(stars) > 1:
                raise ValueError(f"item_id={raw_id}: multiple exact star labels {stars}")
            values = [
                stars[0] if stars else 0,
                cumulative_code(props, FROM_STAR, raw_id),
                cumulative_code(props, RATING, raw_id),
            ] + [int(name in props) for name in selected]
            adjacency_if[:, item_index] = values
            writer.writerow([raw_id] + values)
            if row_number % 100_000 == 0:
                print(f"Encoded metadata: {row_number:,}/{n_items:,}", flush=True)
    rng = np.random.default_rng(seed)
    adjacency_fi = []
    for field_index, cardinality in enumerate(cardinalities):
        neighbors = np.zeros((cardinality, max_neighbors), dtype=np.int32)
        for value in range(cardinality):
            matching = np.flatnonzero(adjacency_if[field_index] == value) + 1
            if len(matching) > max_neighbors:
                matching = rng.choice(matching, max_neighbors, replace=False)
            neighbors[value, :len(matching)] = matching
        adjacency_fi.append(neighbors)
    dump_pickle(adjacency_if, output_dir / "adjacency_if.txt")
    dump_pickle(adjacency_fi, output_dir / "adjacency_fi.txt")
    schema = {
        "field_names": field_names,
        "all_feature": cardinalities,
        "selected_binary_properties": selected,
        "categorical_codebooks": {
            "star_rating": {"0": "not recorded", **{str(i): f"{i} Star" for i in range(1, 6)}},
            "from_star_max": {"0": "not recorded", "1": "From 2 Stars", "2": "From 3 Stars", "3": "From 4 Stars"},
            "rating_highest": {"0": "not recorded", "1": "Satisfactory Rating", "2": "Good Rating", "3": "Very Good Rating", "4": "Excellent Rating"},
        },
        "binary_values": {"0": "not recorded", "1": "recorded"},
        "item_missing_metadata_policy": "exclude_from_examples; every embedding item belongs to catalog",
        "padding_item_id": 0,
        "max_neighbors_per_value": max_neighbors,
    }
    schema_bytes = json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8")
    (output_dir / "metadata_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cardinalities, field_names, hashlib.sha256(schema_bytes).hexdigest()


def build_event_vocabulary(connection, split_map):
    """Fit only on training sessions; validation/test cannot expand the vocabulary."""
    vocabulary = {}
    for user, session, action, reference in connection.execute(
        "SELECT user_id, session_id, action, reference FROM events ORDER BY row_number"
    ):
        if split_map[(user, session)] != "train" or not reference:
            continue
        key = (ACTION_TO_ID[action], reference)
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary) + len(ACTION_TO_ID) + 1
    return vocabulary


def collect_examples(connection: sqlite3.Connection, split_map: dict,
                     item_map: dict[str, int],
                     max_session_items: int, event_vocabulary=None,
                     max_session_events: int = 64) -> tuple[dict, dict]:
    examples = {split: [] for split in SPLITS}
    counters = Counter()
    current_user = None
    session_history = {}
    session_action_history = {}
    session_event_history = {}
    query_count = 0
    event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    price_column = "prices" if "prices" in event_columns else "''"
    cursor = connection.execute(
        "SELECT user_id, session_id, ts, step, row_number, action, item_id, impressions, reference, "
        f"{price_column} FROM events ORDER BY user_id, ts, step, row_number"
    )
    for user, session, timestamp, step, row_number, action, item_id, impressions, reference, prices_raw in cursor:
        if user != current_user:
            current_user = user
            session_history = {}
            session_action_history = {}
            session_event_history = {}
        history = session_history.setdefault(session, [])
        actions = session_action_history.setdefault(session, [])
        events = session_event_history.setdefault(session, deque(maxlen=max_session_events))
        if len(history) != len(actions):
            raise ValueError(f"Item/action history is misaligned for {(user, session)!r}")
        split = split_map[(user, session)]
        if action == "clickout item":
            raw_candidates = [name.strip() for name in (impressions or "").split("|")]
            raw_item_ids = [item_map.get(name, 0) for name in raw_candidates]
            raw_prices, price_length_mismatch = parse_prices(prices_raw, len(raw_candidates))
            if price_length_mismatch:
                counters[f"{split}_price_length_mismatch_queries"] += 1
            counters["unknown_impression_positions"] += sum(item <= 0 for item in raw_item_ids)
            try:
                candidate_records = normalize_candidate_records(
                    raw_item_ids, raw_prices, range(len(raw_item_ids)),
                    target=item_id if item_id else None,
                )
            except ValueError:
                candidate_records = []
            candidates = [record[0] for record in candidate_records]
            if not item_id:
                counters["clickout_target_not_in_catalog"] += 1
            elif not history:
                counters["clickout_without_item_history"] += 1
            elif item_id not in candidates:
                counters["clickout_target_not_in_known_impressions"] += 1
            else:
                sequence = history[-max_session_items:].copy()
                action_sequence = actions[-max_session_items:].copy()
                if len(sequence) != len(action_sequence):
                    raise ValueError("Truncated item/action histories are misaligned")
                example = (sequence, item_id, timestamp, row_number, candidates, action_sequence)
                if event_vocabulary is not None:
                    example += (list(events),)
                example += (candidate_records,)
                examples[split].append(example)
                counters[f"{split}_examples"] += 1
                query_count += 1
                if query_count % 100_000 == 0:
                    print(f"Built {query_count:,} examples", flush=True)
            # Mask the chosen item from val/test history after this query.
            if item_id and split == "train":
                history.append(item_id)
                actions.append(ACTION_TO_ID[action])
        elif action in ITEM_ACTIONS and item_id:
            history.append(item_id)
            actions.append(ACTION_TO_ID[action])
        if event_vocabulary is not None:
            # Snapshot first. This query's answer is never an input. Later
            # validation/test queries see only the clickout action, not its label.
            events.append(event_token(event_vocabulary, action, reference,
                                      masked=action == "clickout item" and split != "train"))
    for split in SPLITS:
        examples[split].sort(key=lambda item: (item[2], item[3]))
    return examples, dict(counters)


def build_groups(examples: list[tuple], m: int, frequency: int) -> list[np.ndarray]:
    """Causal same-split groups; n is a masked padding index, not a session."""
    n = len(examples)
    positive = np.full((n, m), n, dtype=np.int32)
    negative = np.full((n, m), n, dtype=np.int32)
    day_diff = np.zeros((n, m), dtype=np.float32)
    recent_by_item = defaultdict(list)
    recent_all = deque(maxlen=256)
    for index, example in enumerate(examples):
        sequence, target, timestamp, query_id, candidates = example[:5]
        current_items = set(sequence)
        overlap = Counter()
        for item in current_items:
            overlap.update(recent_by_item[item])
        positive_ids = sorted(overlap, key=lambda neighbor: (-overlap[neighbor], -neighbor))[:m]
        negative_ids = []
        for neighbor in reversed(recent_all):
            if current_items.isdisjoint(examples[neighbor][0]):
                negative_ids.append(neighbor)
                if len(negative_ids) == m:
                    break
        for slot, neighbor in enumerate(positive_ids):
            positive[index, slot] = neighbor
            day_diff[index, slot] = min(30, max(0, (timestamp - examples[neighbor][2]) // 86400))
        negative[index, :len(negative_ids)] = negative_ids
        for item in current_items:
            previous = recent_by_item[item]
            previous.append(index)
            if len(previous) > frequency:
                del previous[0]
        recent_all.append(index)
        if (index + 1) % 100_000 == 0:
            print(f"Groups: {index + 1:,}/{n:,}", flush=True)
    return [positive, negative, day_diff]


def prepare(args: argparse.Namespace) -> dict:
    action_scope = getattr(args, "action_scope", "item_event")
    max_session_events = getattr(args, "max_session_events", 64)
    if action_scope not in ("item_event", "all_event") or max_session_events < 1:
        raise ValueError("Invalid action scope or event history budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_map, binary_names, item_map_sha = scan_catalog(args.metadata_csv)
    print(f"Catalog: {len(item_map):,} hotels; {len(binary_names)} binary properties", flush=True)
    split_map, split_info = split_sessions(args.train_csv, args.output_dir, args.max_rows)
    scratch = args.scratch_dir or args.output_dir
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trivago_top5_sort_", dir=scratch) as temporary:
        connection, counts, unmatched = collect_events_and_filters(
            args.train_csv, Path(temporary) / "events.sqlite", split_map,
            item_map, binary_names, args.max_rows, action_scope,
        )
        try:
            selected = choose_fields(binary_names, counts, unmatched,
                                     args.field_budget, args.output_dir)
            print(f"Selected {len(selected)} train-filter properties", flush=True)
            all_feature, field_names, schema_sha = build_metadata(
                args.metadata_csv, args.output_dir, item_map, selected,
                args.max_neighbors, args.seed,
            )
            event_vocabulary = (build_event_vocabulary(connection, split_map)
                                if action_scope == "all_event" else None)
            examples, example_counts = collect_examples(
                connection, split_map, item_map, args.max_session_items,
                event_vocabulary, max_session_events,
            )
        finally:
            connection.close()
    dump_pickle(item_map, args.output_dir / "item_id_map.pkl")
    if event_vocabulary is not None:
        dump_pickle(event_vocabulary, args.output_dir / "event_vocabulary.pkl")
    source_train_sha = fingerprint_file(args.train_csv) if not args.max_rows else None
    source_query_scope = source_train_sha or f"smoke_rows:{args.max_rows}"
    train_price_rows = ([record[2] for record in example[-1]]
                        for example in examples["train"])
    price_scaler = fit_price_scaler(train_price_rows)
    query_manifest_sha256 = {}
    price_audit = {}
    group_pair_counts = {}
    for split in SPLITS:
        split_examples = examples[split]
        data = [[example[0] for example in split_examples],
                [example[1] for example in split_examples],
                [example[5] for example in split_examples]]
        if event_vocabulary is not None:
            data.append([example[6] for example in split_examples])
        dump_pickle(data, args.output_dir / f"{split}.txt")
        dump_pickle([example[4] for example in split_examples],
                    args.output_dir / f"{split}_candidates.txt")
        candidate_rows = [example[-1] for example in split_examples]
        compact_records = flatten_candidate_records(candidate_rows)
        query_ids = np.asarray([example[3] for example in split_examples], dtype=np.int64)
        target_ids = np.asarray([example[1] for example in split_examples], dtype=np.int32)
        query_digest = hashlib.sha256()
        query_digest.update(source_query_scope.encode("utf-8") + b"\n")
        query_digest.update(query_ids.astype("<i8", copy=False).tobytes())
        query_manifest_sha256[split] = query_digest.hexdigest()
        sidecar = {
            "schema": PRICE_FEATURE_SCHEMA,
            "source_train_sha256": source_train_sha,
            "query_ids": query_ids,
            "query_manifest_sha256": query_manifest_sha256[split],
            "target_ids": target_ids,
            **compact_records,
        }
        dump_pickle(sidecar, args.output_dir / f"{split}_candidate_prices.pkl")
        split_prices = compact_records["prices"]
        price_audit[split] = {
            "queries": len(split_examples),
            "candidates": int(len(split_prices)),
            "valid_prices": int(np.isfinite(split_prices).sum()),
            "price_length_mismatch_queries": int(
                example_counts.get(f"{split}_price_length_mismatch_queries", 0)
            ),
        }
        groups = build_groups(split_examples, args.m, args.frequency)
        dump_pickle(groups, args.output_dir / f"{split}_groups_{args.m}_{args.frequency}.txt")
        positive, negative, _ = groups
        n = len(examples[split])
        group_pair_counts[split] = {
            "real_pairs": int(np.count_nonzero((positive < n) & (negative < n))),
            "queries_with_real_pair": int(np.count_nonzero(
                np.any((positive < n) & (negative < n), axis=1))),
            "queries": n,
        }
    artifact_names = [
        "session_split.csv", "metadata_filter_counts.csv",
        "unmatched_filter_references.csv", "metadata_schema.json",
        "item_metadata_selected.csv", "item_id_map.pkl",
        "adjacency_if.txt", "adjacency_fi.txt",
    ]
    for split in SPLITS:
        artifact_names.extend((f"{split}.txt", f"{split}_candidates.txt",
                               f"{split}_groups_{args.m}_{args.frequency}.txt",
                               f"{split}_candidate_prices.pkl"))
    if event_vocabulary is not None:
        artifact_names.append("event_vocabulary.pkl")
    artifact_sha256 = {name: fingerprint_file(args.output_dir / name)
                       for name in artifact_names}
    action_schema = {
        "action_to_id": ACTION_TO_ID,
        "pad_action_id": PAD_ACTION_ID,
        "unk_action_id": UNK_ACTION_ID,
        "num_actions": NUM_ACTIONS,
        "action_input_scope": ACTION_INPUT_SCOPE,
        "action_history_policy": ACTION_HISTORY_POLICY,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    if event_vocabulary is not None:
        action_schema = all_event_schema(len(event_vocabulary) + 11, max_session_events)
    action_schema_bytes = json.dumps(
        action_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    action_schema_sha = hashlib.sha256(action_schema_bytes).hexdigest()
    config = {
        "converter_version": "top5_metadata_v3_item_event_actions",
        "data_schema_version": 3,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_to_id": ACTION_TO_ID,
        "pad_action_id": PAD_ACTION_ID,
        "unk_action_id": UNK_ACTION_ID,
        "num_actions": NUM_ACTIONS,
        "action_input_scope": ACTION_INPUT_SCOPE,
        "action_history_policy": ACTION_HISTORY_POLICY,
        "action_schema_sha256": action_schema_sha,
        "dataset": "TrivagoTop5", "split_scheme": "session_end_time_60_20_20",
        "metadata_mode": "train_filter_top_k", "field_budget": args.field_budget,
        "selected_binary_count": len(selected), "selected_binary_properties": selected,
        "n_node": len(item_map), "all_feature": all_feature,
        "field_names": field_names, "m": args.m, "frequency": args.frequency,
        "max_session_items": args.max_session_items,
        "max_neighbors": args.max_neighbors, "seed": args.seed,
        "max_rows_smoke_only": args.max_rows,
        "group_policy": "causal_same_split_recent_item_overlap; not author preprocessing",
        "group_padding_policy": "missing positive/negative pair masked from contrastive loss and graph",
        "group_pair_counts": group_pair_counts,
        "training_protocol": "full_catalog_cross_entropy",
        "evaluation_protocol": "NDCG@5 and Recall@5 over known-catalog impressions; local labels from raw train.csv",
        "selection_rule": "train filter selection exact reference; count>0; descending count then name",
        "selected_filter_counts": {name: counts[name] for name in selected},
        "unmatched_filter_events": sum(unmatched.values()),
        "item_id_map_sha256": item_map_sha,
        "metadata_schema_sha256": schema_sha,
        "source_train_sha256": source_train_sha,
        "source_metadata_sha256": fingerprint_file(args.metadata_csv),
        "query_manifest_sha256": query_manifest_sha256,
        "price_features": {
            "schema": PRICE_ARTIFACT_SCHEMA,
            "price_schema": PRICE_FEATURE_SCHEMA,
            "feature_order": list(PRICE_FEATURE_ORDER),
            "scaler": price_scaler,
            "candidate_policy": "unique_item_first_raw_impression_v1",
            "query_id_policy": "source_train_sha256_plus_raw_clickout_row_number_v1",
            "invalid_policy": "nonfinite_or_nonpositive_missing_v1",
            "length_mismatch_policy": "invalidate_all_prices_for_query_v1",
            "audit": price_audit,
        },
        "artifact_sha256": artifact_sha256,
        **split_info, "example_counts": example_counts,
    }
    if event_vocabulary is not None:
        config.update(action_schema)
        config.update(converter_version="top5_metadata_v4_all_event_pairs",
                      data_schema_version=4)
    dump_pickle(config, args.output_dir / "config.pkl")
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"fields": len(field_names), "selected": selected,
                      "examples": example_counts}, ensure_ascii=False, indent=2), flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--field-budget", type=int, default=DEFAULT_FIELD_BUDGET,
                        help="Copy 3 default: 15 train-filter binary fields + 3 categorical fields (18 total)")
    parser.add_argument("--max-neighbors", type=int, default=400)
    parser.add_argument("--max-session-items", type=int, default=20)
    parser.add_argument("--action-scope", choices=("item_event", "all_event"), default="item_event")
    parser.add_argument("--max-session-events", type=int, default=64)
    parser.add_argument("--m", type=int, default=14)
    parser.add_argument("--frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Smoke test only; 0 scans the full train.csv")
    args = parser.parse_args()
    if min(args.field_budget, args.max_neighbors, args.max_session_items,
           args.m, args.frequency, args.max_session_events) < 1 or args.max_rows < 0:
        parser.error("All budgets/group sizes must be positive; max-rows >= 0")
    if args.max_session_items > 200:
        parser.error("MLEUP position embedding supports at most 200 items")
    prepare(args)


if __name__ == "__main__":
    main()
