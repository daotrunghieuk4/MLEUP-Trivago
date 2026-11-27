"""Export experiment results from lightweight JSON and checkpoint file metadata."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from .run_metadata import (
    EPOCH_TIMING_FIELDS, FINAL_STATUSES, HYPERPARAMETER_NAMES, STATUS_NAMES, TEST_TIMING_FIELDS,
    as_dict, completed_epochs, file_reference, lock_output, nonnegative, numeric, read_json, replace_file,
    timed_epochs, valid_test_status,
)

ACCURACY = ("ndcg_at_5", "recall_at_5")
LOSSES = ("ce_loss", "metadata_loss", "total_loss")
MEAN_SECONDS = dict(average_train_time_per_epoch_seconds="train_time_seconds",
                    average_val_time_per_epoch_seconds="val_time_seconds",
                    average_epoch_time_seconds="epoch_time_seconds")
TOTAL_SECONDS = dict(total_train_time_seconds="train_time_seconds", total_val_time_seconds="val_time_seconds",
                     total_val_retry_seconds="val_retry_seconds", total_epoch_time_seconds="epoch_time_seconds")
MINUTES = dict(
    avg_train_time_min="average_train_time_per_epoch_seconds", avg_val_time_min="average_val_time_per_epoch_seconds",
    avg_epoch_time_min="average_epoch_time_seconds", total_train_time_min="total_train_time_seconds",
    total_val_time_min="total_val_time_seconds", total_val_retry_min="total_val_retry_seconds",
    total_epoch_time_min="total_epoch_time_seconds", train_time_to_best_min="train_time_to_best_seconds",
    time_to_best_min="time_to_best_seconds", test_time_min="test_time_seconds",
)
INFERENCE = ("inference_ms_per_sample", "inference_samples_per_second")
SUMMARY_COLUMNS = (
    "model", "display_name", "epochs_completed", "best_epoch", "best_val_ndcg_at_5",
    "best_val_recall_at_5", *ACCURACY, *MINUTES, *INFERENCE,
)
EPOCH_COLUMNS = (
    "model", "epoch", *EPOCH_TIMING_FIELDS, "val_ndcg_at_5", "val_recall_at_5",
    "train_total_loss", "val_total_loss", "train_ce_loss", "train_metadata_loss",
    "val_ce_loss", "val_metadata_loss", "learning_rate", "next_learning_rate",
)
ABLATION_COLUMNS = (
    "comparison", "from_model", "to_model",
    "ndcg_before", "ndcg_after", "ndcg_absolute_change", "ndcg_relative_change_pct",
    "recall_before", "recall_after", "recall_absolute_change", "recall_relative_change_pct",
    "train_time_before_min", "train_time_after_min", "train_time_change_pct",
    "time_to_best_before_min", "time_to_best_after_min", "time_to_best_change_pct",
    "inference_before_ms", "inference_after_ms", "inference_change_pct",
)
ENVIRONMENT_KEYS = ("device", "pytorch", "cuda", "databricks_runtime", "batch_size")


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and numeric(float(value)):
        return float(value)
    return None


def load_history(path):
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return [], None, None
    try:
        history = json.loads(content)
        rows = completed_epochs(history)
    except ValueError as exc:
        return [], None, str(exc)
    return rows, hashlib.sha256(content).hexdigest(), None


def accuracy_metrics(value):
    split = as_dict(value)
    return {key: split[key] if nonnegative(split.get(key)) and split[key] <= 1 else None for key in ACCURACY}


def loss_metrics(value):
    split = as_dict(value)
    return {key: split[key] if numeric(split.get(key)) else None for key in LOSSES}


def epoch_details(name, rows):
    return [{
        "model": name, "epoch": row["epoch"],
        **{key: row[key] if nonnegative(row.get(key)) else None for key in EPOCH_TIMING_FIELDS},
        **{f"train_{key}": value for key, value in loss_metrics(row.get("train")).items()},
        **{f"val_{key}": value for key, value in
           (loss_metrics(row.get("val")) | accuracy_metrics(row.get("val"))).items()},
        **{key: row[key] if nonnegative(row.get(key)) else None for key in ("learning_rate", "next_learning_rate")},
    } for row in rows]


def epoch_efficiency(rows):
    efficiency = dict.fromkeys((*MEAN_SECONDS, *TOTAL_SECONDS, *TEST_TIMING_FIELDS))
    if rows:
        efficiency.update({key: sum(row[field] for row in rows) / len(rows) for key, field in MEAN_SECONDS.items()})
        efficiency.update({key: sum(row[field] for row in rows) for key, field in TOTAL_SECONDS.items()})
    return efficiency


def environment_details(metadata, status):
    timing_sessions = status.get("timing_environment")
    sessions = metadata.get("training_sessions")
    observed = list(timing_sessions) if isinstance(timing_sessions, list) else []
    if isinstance(sessions, list):
        observed.extend(as_dict(session).get("environment") for session in sessions)
    observed = [entry for entry in observed if isinstance(entry, dict)]
    first = observed[0] if observed else {}
    test = as_dict(status.get("test_environment")) or None
    device = first.get("device")
    return dict(device=("cpu" if device == "cpu" else "cuda") if device else None,
                gpu_name=device if device != "cpu" else None, pytorch_version=first.get("pytorch"),
                cuda_version=first.get("cuda"), databricks_runtime_version=first.get("databricks_runtime"),
                environment_changed_during_resume=any(entry != first for entry in observed) if observed else None,
                training=observed, test=test)


def collect_experiment(folder, experiment, warnings):
    status_path = folder / "run_status.json"
    status = read_json(status_path, dict) or {}
    rows, history_sha, history_error = load_history(folder / "metric_history.json")
    latest_path, best_path = (folder / f"result_{kind}.pth" for kind in ("latest", "best"))
    latest, best, status_ref = map(file_reference, (latest_path, best_path, status_path))
    label = status.get("status")
    if label not in STATUS_NAMES:
        label = "CHECK_STATUS" if any((latest, best, status_ref, rows, history_error)) else "NOT_STARTED"
    if history_error:
        warnings.append(f"{folder.name}: {history_error}")

    refs = as_dict(status.get("checkpoint_files"))
    version = status.get("report_metadata_version")
    best_matches = latest_matches = history_matches = False
    if version == 1:
        best_matches = best is not None and best == refs.get("best")
        latest_matches = latest is not None and latest == refs.get("latest")
        history_matches = history_sha == status.get("metric_history_sha256") and not history_error
        if (latest and not latest_matches) or (best and not best_matches) or not history_matches:
            warnings.append(f"{folder.name}: checkpoint files or epoch history differ from the last status snapshot.")
    elif version is None and status_ref:
        best_matches = best is not None and best["modified_ns"] <= status_ref["modified_ns"]
        latest_matches = latest is not None and latest["modified_ns"] <= status_ref["modified_ns"]
        history_matches = history_sha is not None and (
            status.get("epochs") is None or status["epochs"] == len(rows)
        )
        warnings.append(f"{folder.name}: legacy status has limited checkpoint references and run metadata.")
    elif version is not None:
        warnings.append(f"{folder.name}: unsupported run status metadata version {version}.")

    if history_error or (best and latest is None) or (
            label in FINAL_STATUSES and not (latest_matches and best_matches and history_matches)):
        label = "CHECK_STATUS"
    metadata = as_dict(status.get("run_metadata")) if version == 1 and (latest_matches or best_matches or (
        label == "RUNNING" and not any((latest, best, refs.get("latest"), refs.get("best")))
    )) else {}
    configuration = as_dict(metadata.get("model_configuration"))
    declared = {key: experiment[option] for key, option in
                (("metadata_fields", "fields"), ("action_mode", "action_mode"), ("price_mode", "price_mode"))}
    if any(configuration.get(key) not in (None, expected) for key, expected in declared.items()):
        warnings.append(f"{folder.name}: recorded model configuration differs from the notebook definition.")
    parameters = as_dict(metadata.get("hyperparameters"))
    reproducibility = as_dict(metadata.get("reproducibility"))
    if metadata.get("starts_from_legacy_checkpoint"):
        warnings.append(f"{folder.name}: original settings unavailable in the legacy checkpoint are null.")
    environment = environment_details(metadata, status if version is None or metadata else {})
    if environment["environment_changed_during_resume"]:
        warnings.append(f"{folder.name}: training environments changed across resume sessions.")

    timed = []
    if rows and (parameters.get("track_time") is True or any(
            any(key in row for key in EPOCH_TIMING_FIELDS) for row in rows)):
        try:
            timed = timed_epochs(rows)
        except ValueError as exc:
            warnings.append(f"{folder.name}: {exc}. Timing totals are unavailable.")
    efficiency = epoch_efficiency(timed)

    best_epoch = status.get("best_epoch")
    best_record = None
    best_scores = [accuracy_metrics(row.get("val"))["ndcg_at_5"] for row in rows]
    if (best_matches and history_matches and type(best_epoch) is int and 1 <= best_epoch <= len(rows)
            and all(score is not None for score in best_scores)
            and best_epoch == best_scores.index(max(best_scores)) + 1
            and status.get("best_val_ndcg_at_5") == best_scores[best_epoch - 1]
            and status.get("selection_metric") in (None, "ndcg_at_5")):
        best_record = rows[best_epoch - 1]
    elif best_epoch is not None or label in FINAL_STATUSES:
        warnings.append(f"{folder.name}: saved best epoch could not be reconciled with validation history and checkpoint.")
        if label in FINAL_STATUSES:
            label = "CHECK_STATUS"

    best_info = None
    if best_record:
        before_best = timed[:best_epoch]
        if not before_best and parameters.get("track_time") is not False:
            try:
                before_best = timed_epochs(rows[:best_epoch])
            except ValueError:
                pass
        best_timing = dict(train_time_seconds=None, val_time_seconds=None, epoch_time_seconds=None,
                           val_retry_seconds=None, train_time_to_best_seconds=None, time_to_best_seconds=None)
        if before_best:
            for key in ("train_time_seconds", "val_time_seconds", "epoch_time_seconds", "val_retry_seconds"):
                best_timing[key] = before_best[-1][key]
            best_timing["train_time_to_best_seconds"] = sum(row["train_time_seconds"] for row in before_best)
            best_timing["time_to_best_seconds"] = sum(
                row["epoch_time_seconds"] + row["val_retry_seconds"] for row in before_best
            )
        best_info = dict(epoch=best_epoch, selection_metric="ndcg_at_5",
                         learning_rate=epoch_details(folder.name, [best_record])[0]["learning_rate"],
                         train=loss_metrics(best_record.get("train")),
                         validation=loss_metrics(best_record.get("val")) | accuracy_metrics(best_record.get("val")),
                         timing=best_timing)

    test = None
    best_id = refs.get("best_identifier") if version == 1 else status.get("best_checkpoint_id")
    evaluation_path = status.get("evaluated_checkpoint")
    if (label == "EVALUATED" and best_info and latest_matches and valid_test_status(status, False)
            and isinstance(evaluation_path, str) and Path(evaluation_path).resolve() == best_path.resolve()
            and (version is None or status.get("best_checkpoint_id") == best_id)
            and (version is None or status.get("epochs") == len(rows))
            and (version is None or parameters.get("track_time") is not True or bool(best_id))):
        metrics = status["metrics"]
        test = {key: metrics[key] for key in (*ACCURACY, "examples")}
        if valid_test_status(status, True):
            efficiency.update({key: metrics[key] for key in TEST_TIMING_FIELDS})
        elif any(key in metrics for key in TEST_TIMING_FIELDS):
            warnings.append(f"{folder.name}: test timing is incomplete or invalid.")
    elif status.get("status") == "EVALUATED":
        label = "CHECK_STATUS"
        warnings.append(f"{folder.name}: test results could not be matched to the completed run and best checkpoint.")
    early_stopping = as_dict(status.get("early_stopping"))
    return {
        "display_name": experiment["display_name"], "status": label,
        "verification": "checkpoint_file_metadata" if version == 1 and latest_matches else
                        "legacy_status" if version is None and latest_matches else None,
        "configured_model": declared,
        "model_configuration": {key: configuration.get(key) for key in (*declared, "metadata_projection")},
        "hyperparameters": {key: parameters.get(key) for key in ("seed", *HYPERPARAMETER_NAMES, "weight_decay")},
        "training_summary": {
            "epochs_completed": len(rows) if not history_error and (history_sha is not None or label == "NOT_STARTED") else None,
            "global_step": status.get("global_step") if latest_matches else None,
            "training_finished": status.get("training_finished") if label in STATUS_NAMES else
                                 False if label == "NOT_STARTED" else None,
            "early_stopped": early_stopping.get("stopped") if latest_matches else None,
            "early_stopping_reason": early_stopping.get("reason") if latest_matches else None,
        },
        "best_epoch": best_info, "test_results": test, "efficiency": efficiency,
        "checkpoint": dict(best_checkpoint_path=best_path if best else None,
                           latest_checkpoint_path=latest_path if latest else None,
                           best_checkpoint_size_bytes=best["size_bytes"] if best else None,
                           best_checkpoint_identifier=(best_id or
                               f"file-stat:{best['size_bytes']}:{best['modified_ns']}") if best_matches else None),
        "environment": environment,
        "reproducibility": {key: reproducibility.get(key) for key in
                            ("git_commit_sha", "data_signature", "training_policy", "model_arguments")},
        "training_sessions": metadata.get("training_sessions") or [],
        "starts_from_legacy_checkpoint": metadata.get("starts_from_legacy_checkpoint"),
    }, epoch_details(folder.name, rows)


def summary_row(name, result):
    best = result["best_epoch"]
    time = result["efficiency"] | best["timing"]
    return dict(
        model=name, display_name=result["display_name"],
        epochs_completed=result["training_summary"]["epochs_completed"], best_epoch=best["epoch"],
        **{f"best_val_{key}": best["validation"][key] for key in ACCURACY},
        **{key: result["test_results"][key] for key in ACCURACY},
        **{name: time[key] / 60 if time[key] is not None else None for name, key in MINUTES.items()},
        **{key: result["efficiency"][key] for key in INFERENCE},
    )


def timing_scope(result, test=False):
    env = result["environment"]
    if test:
        observed = env["test"]
    else:
        observed = env["training"][-1] if env["training"] and not env["environment_changed_during_resume"] else None
    return json.dumps([observed.get(key) for key in ENVIRONMENT_KEYS], sort_keys=True) if observed else None


def data_scope(result):
    signature = result["reproducibility"]["data_signature"]
    if not isinstance(signature, (tuple, list)) or len(signature) < 9:
        return None
    try:
        artifacts = dict(signature[8])
    except (TypeError, ValueError):
        return None
    return signature[4], signature[6], signature[7], *(artifacts.get(f"{part}_candidates.txt")
                                                       for part in ("train", "val", "test"))


def ablation_rows(experiments, results, summaries, warnings):
    comparisons = []
    names = list(experiments)
    for before, after in zip(names, names[1:]):
        if before not in summaries or after not in summaries:
            continue
        if any(value is not None and value != results[name]["configured_model"][key]
               for name in (before, after) for key, value in results[name]["model_configuration"].items()
               if key in results[name]["configured_model"]):
            continue
        old_scope, new_scope = (data_scope(results[key]) for key in (before, after))
        if old_scope and new_scope and old_scope != new_scope:
            warnings.append(f"{before} → {after}: data or evaluation candidates differ; comparison omitted.")
            continue
        old, new = summaries[before], summaries[after]
        comparison = dict(comparison=experiments[after].get("comparison") or f"{before} → {after}",
                          from_model=before, to_model=after)
        for key, prefix in (("ndcg_at_5", "ndcg"), ("recall_at_5", "recall")):
            first, second = old[key], new[key]
            delta = second - first
            comparison.update({f"{prefix}_before": first, f"{prefix}_after": second,
                               f"{prefix}_absolute_change": delta,
                               f"{prefix}_relative_change_pct": delta / first * 100 if first else None})
        for key, prefix, unit in (("avg_train_time_min", "train_time", "min"),
                                  ("time_to_best_min", "time_to_best", "min"),
                                  ("inference_ms_per_sample", "inference", "ms")):
            first, second = old[key], new[key]
            scopes = [timing_scope(results[name], test=prefix == "inference") for name in (before, after)]
            comparable = scopes[0] is not None and scopes[0] == scopes[1] and not any(
                results[name]["environment"]["environment_changed_during_resume"] for name in (before, after)
            )
            comparison.update({f"{prefix}_before_{unit}": first, f"{prefix}_after_{unit}": second,
                               f"{prefix}_change_pct": (second - first) / first * 100
                               if comparable and first and second is not None else None})
        comparisons.append(comparison)
    return comparisons


def comparisons(rows):
    fields = {
        "highest_test_ndcg_at_5": ("ndcg_at_5", max), "highest_test_recall_at_5": ("recall_at_5", max),
        "lowest_average_train_time": ("avg_train_time_min", min), "lowest_time_to_best": ("time_to_best_min", min),
        "lowest_inference_time": ("inference_ms_per_sample", min),
        "highest_inference_throughput": ("inference_samples_per_second", max),
    }
    result = {}
    for label, (key, choose) in fields.items():
        values = {name: row[key] for name, row in rows.items() if row[key] is not None}
        value = choose(values.values()) if values else None
        result[label] = (dict(experiment_ids=[name for name, entry in values.items() if entry == value], value=value)
                         if values else None)
    return result


def write_atomic(path, write):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            write(output)
        replace_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_results(runs_dir, experiments):
    with lock_output(runs_dir, wait=True):
        return _export_results(runs_dir, experiments)


def _export_results(runs_dir, experiments):
    results, summaries, epochs, warnings = {}, {}, [], []
    for name, experiment in experiments.items():
        result, details = collect_experiment(runs_dir / name, experiment, warnings)
        results[name] = result
        epochs.extend(details)
        if result["test_results"] is not None:
            summaries[name] = summary_row(name, result)
    for test, key in ((False, "avg_train_time_min"), (True, "inference_ms_per_sample")):
        scopes = [timing_scope(results[name], test) for name, row in summaries.items() if row[key] is not None]
        if len(scopes) > 1 and (len(set(scopes)) > 1 or None in scopes):
            warnings.append(f"{'Test' if test else 'Training'} timing environments differ or were not fully recorded.")
    ablation = ablation_rows(experiments, results, summaries, warnings)
    report = json_ready(dict(schema_version=1, experiment_name="MLEUP-Trivago Ablation", run_id=runs_dir.name,
                             updated_at=datetime.now(timezone.utc).isoformat(), experiments=results,
                             comparison=comparisons(summaries), ablation_comparison=ablation, warnings=warnings))
    summary = pd.DataFrame(summaries.values(), columns=SUMMARY_COLUMNS)
    timeline = pd.DataFrame(epochs, columns=EPOCH_COLUMNS)
    ablation_frame = pd.DataFrame(ablation, columns=ABLATION_COLUMNS)
    for frame, name in ((summary, "experiment_summary.csv"), (timeline, "epoch_timing.csv"),
                        (ablation_frame, "ablation_comparison.csv")):
        write_atomic(runs_dir / name, lambda file: frame.to_csv(file, index=False))
    write_atomic(runs_dir / "best_experiments.json",
                 lambda file: json.dump(report, file, ensure_ascii=False, allow_nan=False, indent=2))
    return report, summary, timeline, ablation_frame
