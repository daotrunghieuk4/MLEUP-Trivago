"""Lightweight checkpoint references and metric validation shared with reporting."""

import json
import math
import os
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


EPOCH_TIMING_FIELDS = (
    "train_time_seconds", "val_time_seconds", "epoch_time_seconds",
    "cumulative_train_time_seconds", "cumulative_val_time_seconds", "val_retry_seconds",
)
TEST_TIMING_FIELDS = (
    "test_time_seconds", "inference_total_seconds",
    "inference_ms_per_sample", "inference_samples_per_second",
)
PAUSED_STATUSES = ("PAUSED", "PAUSED_DURING_VAL", "PAUSED_DURING_VAL_MIGRATION", "SMOKE_STOP")
FINAL_STATUSES = ("COMPLETE", "EARLY_STOPPED", "EVALUATED")
STATUS_NAMES = ("RUNNING", *PAUSED_STATUSES, *FINAL_STATUSES)


def as_dict(value):
    return value if isinstance(value, dict) else {}


def replace_file(source, target):
    for delay in (0.05, 0.1, 0.2, None):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            # Windows readers can briefly hold the destination without delete sharing.
            if delay is None or getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(delay)


@contextmanager
def lock_output(output_dir, *, wait=False):
    """Keep the lock file in place; its OS lock ends when the writer exits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".writer.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Cannot lock {output_dir}. Stop the other writer or use "
                               "storage that supports exclusive file locks.") from exc
        yield


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    replace_file(temporary, path)


def read_json(path, expected_type):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        value = None
    if not isinstance(value, expected_type):
        print(f"Ignoring invalid JSON in {path}", flush=True)
        return None
    return value


def numeric(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def nonnegative(value):
    return numeric(value) and value >= 0


def completed_epochs(history, completed=None):
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("Invalid metric history")
    rows = [row for row in history if "epoch" in row]
    if (any(type(row["epoch"]) is not int or row["epoch"] != number
            for number, row in enumerate(rows, 1))
            or any("epoch" not in row and row.get("phase") != "resume_baseline" for row in history)
            or (completed is not None and len(rows) != completed)):
        raise ValueError("Metric history must contain consecutive completed epochs starting at 1")
    return rows


def timed_epochs(history, completed=None):
    rows = completed_epochs(history, completed)
    train = val = 0.0
    for row in rows:
        if any(not nonnegative(row.get(key)) for key in EPOCH_TIMING_FIELDS):
            raise ValueError(f"Missing or invalid timing in epoch {row['epoch']}")
        train += row["train_time_seconds"]
        val += row["val_time_seconds"]
        if not all(math.isclose(row[key], expected) for key, expected in (
                ("cumulative_train_time_seconds", train), ("cumulative_val_time_seconds", val),
                ("epoch_time_seconds", row["train_time_seconds"] + row["val_time_seconds"]))):
            raise ValueError(f"Inconsistent timing totals in epoch {row['epoch']}")
    return rows


def valid_test_status(status, track_time):
    metrics = status.get("metrics")
    return (
        status.get("status") == "EVALUATED" and status.get("split") == "test"
        and status.get("selection_metric") == "ndcg_at_5"
        and status.get("training_finished") is True and isinstance(metrics, dict)
        and all(nonnegative(metrics.get(key)) and metrics[key] <= 1
                for key in ("ndcg_at_5", "recall_at_5"))
        and type(metrics.get("examples")) is int and metrics["examples"] > 0
        and (not track_time or (
            all(nonnegative(metrics.get(key)) for key in TEST_TIMING_FIELDS)
            and metrics["inference_total_seconds"] > 0
            and metrics["test_time_seconds"] >= metrics["inference_total_seconds"]
            and math.isclose(metrics["inference_ms_per_sample"],
                             metrics["inference_total_seconds"] * 1000 / metrics["examples"])
            and math.isclose(metrics["inference_samples_per_second"],
                             metrics["examples"] / metrics["inference_total_seconds"])
        ))
    )


def file_reference(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if stat.S_ISREG(info.st_mode):
        return {"size_bytes": info.st_size, "modified_ns": info.st_mtime_ns}
    return None


def source_git_sha():
    repo = Path(__file__).resolve().parents[1]
    try:
        options = dict(cwd=repo, check=True, capture_output=True, text=True, timeout=5)
        head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], **options).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal", "--", "*.py"],
                                 **options).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return None if changed else head


HYPERPARAMETER_NAMES = {
    "max_epochs": "epochs", "batch_size": "batch_size", "embedding_size": "emb_size",
    "num_heads": "num_heads", "sample_order": "sample_order", "si_dc": "si_dc",
    "alpha": "alpha", "beta": "beta", "initial_learning_rate": "lr", "lr_factor": "lr_factor",
    "lr_patience": "lr_patience", "min_lr": "min_lr", "patience": "patience",
    "min_delta": "min_delta", "loss_patience": "loss_patience", "loss_min_delta": "loss_min_delta",
    "checkpoint_every": "checkpoint_every", "max_hours": "max_hours", "track_time": "track_time",
}


def new_run_metadata(args, config, model_arguments, signature, policy, *,
                     fresh, weight_decay, checkpoint=None):
    checkpoint = checkpoint or {}
    if fresh:
        parameters = {field: getattr(args, option) for field, option in HYPERPARAMETER_NAMES.items()}
        parameters["weight_decay"] = weight_decay
        commit = source_git_sha()
    else:
        recorded = {**(checkpoint.get("training_policy") or {}), **model_arguments,
                    "track_time": args.track_time}
        parameters = {field: recorded.get(option) for field, option in HYPERPARAMETER_NAMES.items()}
        groups = (checkpoint.get("optimizer") or {}).get("param_groups", [])
        decays = {group.get("weight_decay") for group in groups}
        parameters["weight_decay"] = next(iter(decays)) if len(decays) == 1 else None
        commit = None
    return {
        "hyperparameters": {"seed": config["seed"], **parameters},
        "model_configuration": {
            "metadata_fields": len(config["field_names"]),
            **{field: model_arguments.get(field) for field in
               ("metadata_projection", "action_mode", "price_mode")},
        },
        "reproducibility": {
            "git_commit_sha": commit,
            "data_signature": checkpoint.get("signature", signature),
            "training_policy": policy if fresh else checkpoint.get("training_policy"),
            "model_arguments": checkpoint.get("model_arguments", model_arguments),
        },
        "training_sessions": [],
        "starts_from_legacy_checkpoint": not fresh,
    }
