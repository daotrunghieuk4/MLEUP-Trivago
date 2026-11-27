"""Prepare shared datasets and invoke the existing checkpoint-aware trainer."""

import hashlib
import os
import pickle
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from .trivago_data_parts import CSV_NAMES, unpack
from .trivago_price_features import PRICE_ARTIFACT_SCHEMA, PRICE_FEATURE_ORDER, PRICE_FEATURE_SCHEMA
from .trivago_top5_checkpoint import prepared_data_signature

from .run_metadata import (
    FINAL_STATUSES, PAUSED_STATUSES, STATUS_NAMES, as_dict, atomic_json, file_reference,
    nonnegative, numeric, read_json, valid_test_status,
)

TRAIN_OPTIONS = (
    "epochs", "batch_size", "emb_size", "num_heads", "sample_order", "si_dc", "beta", "alpha",
    "lr", "lr_factor", "lr_patience", "min_lr", "patience", "min_delta", "loss_patience",
    "loss_min_delta", "metadata_projection", "checkpoint_every", "max_hours",
)
SPLITS = ("train", "val", "test")


def validate_configuration(experiments, training, runs_dir):
    if not experiments:
        raise ValueError("Configure at least one experiment")
    paths = set()
    for name, experiment in experiments.items():
        if not name or Path(name).name != name or name in (".", "..") or any(c in name for c in "/\\:"):
            raise ValueError(f"Invalid experiment ID: {name!r}")
        path = (runs_dir / name).resolve()
        key = os.path.normcase(str(path))
        if key in paths or path.parent != runs_dir.resolve():
            raise ValueError(f"Experiment {name} has a duplicate or redirected output directory: {path}")
        paths.add(key)
        if type(experiment["fields"]) is not int or experiment["fields"] < 4:
            raise ValueError(f"Invalid metadata field count for {name}")
        if experiment["action_mode"] not in ("none", "item_event", "event_graph_candidate"):
            raise ValueError(f"Unsupported action mode for the item-event pipeline: {name}")
        price = experiment["price_mode"]
        if price not in ("off", "relative_mlp_v1") or (
                price != "off" and experiment["action_mode"] != "event_graph_candidate"):
            raise ValueError(f"Incompatible action and price mode for {name}")
    positive = ("epochs", "batch_size", "emb_size", "num_heads", "sample_order", "checkpoint_every",
                "patience", "loss_patience")
    if any(type(training[key]) is not int or training[key] < 1 for key in positive):
        raise ValueError(f"Expected positive integers for {', '.join(positive)}")
    if training["emb_size"] % training["num_heads"]:
        raise ValueError("Embedding size must be divisible by number of attention heads")
    if (any(not numeric(training[key]) for key in ("lr", "min_lr", "lr_factor"))
            or not 0 < training["min_lr"] <= training["lr"] or not 0 < training["lr_factor"] < 1
            or type(training["lr_patience"]) is not int or training["lr_patience"] < 0):
        raise ValueError("Invalid initial learning rate, minimum learning rate or scheduler settings")
    if (not numeric(training["beta"]) or training["beta"] <= 0
            or any(not nonnegative(training[key]) for key in ("min_delta", "loss_min_delta", "max_hours"))
            or any(not numeric(training[key]) for key in ("alpha", "si_dc"))
            or training["metadata_projection"] not in ("concat", "fieldwise")
            or type(training["track_time"]) is not bool):
        raise ValueError("Invalid model, early stopping, session duration or timing configuration")


def check_repo(repo):
    if not (repo / "src" / "train_trivago_top5.py").is_file():
        raise FileNotFoundError(f"Đặt notebook tại thư mục gốc MLEUP-Trivago. CWD hiện tại: {repo}")


def check_environment(repo, runs_dir, *, custom_storage=False):
    import torch

    check_repo(repo)
    print(f"Repository: {repo}\nRun directory: {runs_dir}")
    if os.name == "posix":
        temporary_roots = ("/tmp", "/var/tmp", "/local_disk0", "/dev/shm", "/databricks/driver")
        if any(runs_dir.is_relative_to(path) for path in temporary_roots):
            raise ValueError(f"{runs_dir} nằm trên đĩa tạm. Đặt MLEUP_CHECKPOINT_ROOT sang vị trí bền vững.")
        if not custom_storage and not runs_dir.is_relative_to("/Workspace"):
            print("Cảnh báo: checkpoint nằm ngoài /Workspace. Hãy kiểm tra lại sau restart compute.")
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=runs_dir, prefix="mleup_write_probe_", suffix=".tmp") as probe:
            probe.write(b"mleup")
            probe.flush()
    except OSError as exc:
        raise RuntimeError(f"Không ghi được vào {runs_dir}. Kiểm tra quyền ghi và dung lượng "
                           "hoặc cấu hình MLEUP_CHECKPOINT_ROOT.") from exc
    cuda = torch.cuda.is_available()
    return dict(python=platform.python_version(), pytorch=str(torch.__version__), cuda_available=cuda,
                cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0) if cuda else None,
                databricks_runtime=os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def inspect_run_status(runs_dir, experiments):
    rows = []
    for name, experiment in experiments.items():
        output = runs_dir / name
        path = output / "run_status.json"
        status = read_json(path, dict) or {}
        latest, best = (output / f"result_{role}.pth" for role in ("latest", "best"))
        label = status.get("status")
        if label not in STATUS_NAMES:
            label = "CHECK_STATUS" if any(p.exists() for p in (path, latest, best)) else "NOT_STARTED"
        if not latest.is_file() and (best.exists() or label in (*FINAL_STATUSES, *PAUSED_STATUSES)):
            label = "CHECK_STATUS"
        rows.append(dict(model=name, display_name=experiment["display_name"], status=label,
                         global_step=status.get("global_step"), best_epoch=status.get("best_epoch"),
                         latest_exists=latest.is_file(), best_exists=best.is_file(),
                         verification="JSON and file presence only"))
    return rows


def run_script(repo, *args):
    command = [sys.executable, "-X", "utf8", "-u", *args]
    with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def fingerprint_file(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_checksums(raw_dir):
    return {name: fingerprint_file(raw_dir / name) for name in CSV_NAMES}


def restore_raw(parts, scratch):
    manifest_sha = fingerprint_file(parts / "manifest.json")
    cache_path = scratch / "raw_cache.json"
    cache = read_json(cache_path, dict) or {}
    directory = cache.get("directory")
    if cache.get("manifest") == manifest_sha and isinstance(directory, str) and Path(directory).name == directory:
        raw = scratch / directory / "raw"
        if all((raw / name).is_file() for name in CSV_NAMES):
            checksums = raw_checksums(raw)
            if checksums == cache.get("checksums"):
                return raw, checksums
    # A failed extraction is never reused as a complete raw dataset.
    temporary = Path(tempfile.mkdtemp(prefix="dataset_", dir=scratch))
    raw = temporary / "raw"
    print(f"Đang khôi phục CSV từ {parts}", flush=True)
    unpack(parts, raw)
    checksums = raw_checksums(raw)
    atomic_json(cache_path, dict(manifest=manifest_sha, directory=temporary.name, checksums=checksums))
    return raw, checksums


def verify_prepared(path, fields, seed, source, use_price):
    config = read_json(path / "config.json", dict)
    try:
        with (path / "config.pkl").open("rb") as file:
            binary_config = pickle.load(file)
        if config != binary_config:
            raise ValueError("config.json và config.pkl không khớp")
        if (len(config["field_names"]) != fields or config["seed"] != seed
                or config["metadata_mode"] != "train_filter_top_k"
                or config["data_schema_version"] != 3):
            raise ValueError(f"metadata, schema hoặc seed không khớp {fields} fields và seed {seed}")
        if (config["source_train_sha256"] != source["train.csv"]
                or config["source_metadata_sha256"] != source["item_metadata.csv"]):
            raise ValueError("SHA-256 của CSV nguồn không khớp")
        prepared_data_signature(config)
        if set(config["query_manifest_sha256"]) != set(SPLITS):
            raise ValueError("thiếu query manifest")
        required = {"item_id_map.pkl", "metadata_schema.json", "adjacency_if.txt", "adjacency_fi.txt"}
        for split in SPLITS:
            required.update((f"{split}.txt", f"{split}_candidates.txt",
                             f"{split}_groups_{config['m']}_{config['frequency']}.txt"))
            if use_price:
                required.add(f"{split}_candidate_prices.pkl")
        artifacts = config["artifact_sha256"]
        if not required.issubset(artifacts):
            raise ValueError("thiếu fingerprint của prepared artifacts")
        if use_price:
            price = config["price_features"]
            if (price["schema"] != PRICE_ARTIFACT_SCHEMA or price["price_schema"] != PRICE_FEATURE_SCHEMA
                    or price["feature_order"] != list(PRICE_FEATURE_ORDER) or price["scaler"]["count"] < 1):
                raise ValueError("schema hoặc scaler giá không tương thích")
        for name, checksum in artifacts.items():
            if Path(name).name != name or fingerprint_file(path / name) != checksum:
                raise ValueError(f"artifact không khớp SHA-256: {name}")
    except (KeyError, TypeError, ValueError, OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Prepared không hợp lệ ở {path}: {exc}. Dùng thư mục prepared mới hoặc khôi phục bộ cũ.") from exc
    return config


def prepare_datasets(repo, raw_dir, experiment_dir, scratch, experiments, seed):
    check_repo(repo)
    parts = repo / "data_parts"
    if parts.is_dir():
        raw_dir, source = restore_raw(parts, scratch)
        experiment_dir = scratch / "ablation_v1"
    else:
        source = raw_checksums(raw_dir)
    configs, prepared = {}, {}
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for fields in dict.fromkeys(item["fields"] for item in experiments.values()):
        path = experiment_dir / f"prepared_{fields}"
        use_price = any(item["fields"] == fields and item["price_mode"] != "off" for item in experiments.values())
        if path.exists():
            config = verify_prepared(path, fields, seed, source, use_price)
        else:
            with tempfile.TemporaryDirectory(prefix="prepare_", dir=experiment_dir) as temporary:
                staged = Path(temporary) / path.name
                run_script(repo, "-m", "src.prepare_trivago_top5", "--train-csv", str(raw_dir / "train.csv"),
                           "--metadata-csv", str(raw_dir / "item_metadata.csv"), "--output-dir", str(staged),
                           "--scratch-dir", str(scratch), "--field-budget", str(fields - 3),
                           "--action-scope", "item_event", "--seed", str(seed))
                config = verify_prepared(staged, fields, seed, source, use_price)
                staged.rename(path)
        prepared[fields], configs[fields] = path, config
    reference = next(iter(configs.values()))
    for fields, config in configs.items():
        for key in ("source_train_sha256", "source_metadata_sha256", "item_id_map_sha256", "query_manifest_sha256"):
            if config[key] != reference[key]:
                raise ValueError(f"Prepared {fields} fields khác nhau ở {key}.")
        for split in SPLITS:
            key = f"{split}_candidates.txt"
            if config["artifact_sha256"][key] != reference["artifact_sha256"][key]:
                raise ValueError(f"Prepared {fields} fields khác danh sách ứng viên của {split}.")
    return prepared


def build_train_command(experiment, training, prepared_dir, output_dir):
    command = ["-m", "src.train_trivago_top5", "--prepared-dir", str(prepared_dir), "--output-dir", str(output_dir)]
    for name in TRAIN_OPTIONS:
        command.extend(("--" + name.replace("_", "-"), str(training[name])))
    command.extend(("--action-mode", experiment["action_mode"], "--price-mode", experiment["price_mode"]))
    if training["track_time"]:
        command.append("--track-time")
    return command


def run_experiment(name, *, repo, runs_dir, experiments, training, prepared, seed):
    validate_configuration(experiments, training, runs_dir)
    experiment = experiments[name]
    output = runs_dir / name
    latest, best = output / "result_latest.pth", output / "result_best.pth"
    if not latest.is_file() and best.exists():
        raise RuntimeError(f"Thiếu {latest} trong khi best checkpoint tồn tại. Kiểm tra các file đã lưu.")
    check_repo(repo)
    prior = read_json(output / "run_status.json", dict) or {}
    parameters = as_dict(as_dict(prior.get("run_metadata")).get("hyperparameters"))
    if (latest.is_file() and as_dict(prior.get("checkpoint_files")).get("latest") == file_reference(latest)
            and parameters.get("initial_learning_rate") not in (None, training["lr"])):
        raise ValueError(f"{name}: learning rate khác cấu hình đã lưu. Dùng run directory mới.")
    fields = experiment["fields"]
    config = read_json(prepared[fields] / "config.json", dict) or {}
    if len(config.get("field_names", ())) != fields or config.get("seed") != seed:
        raise ValueError(f"{name}: prepared dataset chưa khớp {fields} fields hoặc seed {seed}.")
    print(f"{name} | {experiment['display_name']} | {fields} fields | "
          f"action={experiment['action_mode']} | price={experiment['price_mode']}", flush=True)
    command = build_train_command(experiment, training, prepared[fields], output)
    run_script(repo, *command, *(["--resume"] if latest.is_file() else []))
    status_path = output / "run_status.json"
    status = read_json(status_path, dict) or {}
    if status.get("status") in PAUSED_STATUSES:
        print(f"{name} tạm dừng tại bước {status.get('global_step')}. Chạy lại cell này để resume.")
        return status
    if status.get("training_finished") is not True:
        raise RuntimeError(f"Không tìm thấy trạng thái hoàn thành hợp lệ trong {status_path}")
    if not valid_test_status(status, training["track_time"]):
        run_script(repo, *command, "--eval-only", "test")
        status = read_json(status_path, dict) or {}
    if not valid_test_status(status, training["track_time"]):
        raise RuntimeError(f"Không tìm thấy kết quả test hợp lệ trong {status_path}")
    metrics = status["metrics"]
    print(f"Test | NDCG@5={metrics['ndcg_at_5']:.4f} | Recall@5={metrics['recall_at_5']:.4f} "
          f"| best epoch={status['best_epoch']} | {metrics['examples']} mẫu")
    if training["track_time"]:
        print(f"Test: {metrics['test_time_seconds']:.2f} s | "
              f"Forward: {metrics['inference_ms_per_sample']:.3f} ms/mẫu")
    print(f"Kết quả lưu tại {output}")
    return status
