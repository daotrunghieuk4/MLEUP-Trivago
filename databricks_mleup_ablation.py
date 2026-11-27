# Databricks notebook source
# MAGIC %md
# MAGIC # Thực nghiệm MLEUP trên Trivago
# MAGIC Đặt notebook tại thư mục gốc Git Folder `MLEUP-Trivago`, trên compute có `numpy`, `torch` và `pandas`.
# MAGIC Checkpoint được ghi trong `runs/databricks_checkpoints`. Dùng cùng GPU, seed, batch size,
# MAGIC số epoch tối đa và cách chọn checkpoint theo NDCG@5 trên validation cho cả bốn lượt.
# MAGIC Nếu có `data_parts/manifest.json` cạnh notebook, cell cấu hình tự ghép các phần,
# MAGIC kiểm tra SHA-256 và giải nén dữ liệu vào đĩa tạm. Kết quả ở đó có thể mất khi compute dừng.
# MAGIC Sau cell chuẩn bị dữ liệu, chạy từng cell train hoặc chọn **Run All**. Mỗi cell
# MAGIC in tiến độ theo epoch và tự tiếp tục từ checkpoint khi chạy lại. Cell cuối xuất hai CSV.
# MAGIC
# MAGIC Hai baseline là bản MLEUP thích nghi cho Trivago. Cải tiến 1 thêm đồ thị sự kiện và
# MAGIC attention theo ứng viên, đồng thời chuyển loss từ toàn bộ danh mục sang danh sách
# MAGIC impressions. Chênh lệch giữa hai lượt này phản ánh cả hai thay đổi. Cải tiến 2
# MAGIC thêm giá tương đối trên cấu hình của cải tiến 1. Cả bốn đánh giá trên impressions.

# COMMAND ----------

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

from train_trivago_top5 import EPOCH_TIMING_FIELDS, numeric, read_json, timed_epochs, valid_test_status
from trivago_data_parts import CSV_NAMES, unpack

RAW_DIR = Path("/Volumes/<catalog>/<schema>/<volume>/raw")
EXPERIMENT_DIR = RAW_DIR.parent / "ablation_v1"
EPOCHS = 50
BATCH_SIZE = 128
SEED = 2023
CHECKPOINT_EVERY = 1000
MAX_HOURS = 8.5
TRACK_TIME = True

PATIENCE = 5
MIN_DELTA = 0.0001
LOSS_PATIENCE = 50

repo = Path.cwd()
scratch = Path(tempfile.mkdtemp(prefix="mleup_ablation_"))
if not (repo / "train_trivago_top5.py").is_file():
    raise FileNotFoundError("Đặt notebook tại thư mục gốc MLEUP-Trivago.")
parts_dir = repo / "data_parts"
if parts_dir.is_dir():
    RAW_DIR = scratch / "raw"
    EXPERIMENT_DIR = scratch / "ablation_v1"
    print(f"Đang khôi phục CSV từ {parts_dir}", flush=True)
    unpack(parts_dir, RAW_DIR)
CHECKPOINT_ROOT = Path(os.environ.get(
    "MLEUP_CHECKPOINT_ROOT", repo / "runs" / "databricks_checkpoints"
)).expanduser().resolve()
run_tag = "_timed_v1" if TRACK_TIME else ""
RUNS_DIR = (CHECKPOINT_ROOT / f"seed{SEED}_e{EPOCHS}_b{BATCH_SIZE}{run_tag}").resolve()
print(f"Repository: {repo}\nCheckpoint root: {CHECKPOINT_ROOT}\nRun directory: {RUNS_DIR}")
if os.name == "posix":
    temporary_roots = ("/tmp", "/var/tmp", "/local_disk0", "/dev/shm", "/databricks/driver")
    if any(RUNS_DIR.is_relative_to(root) for root in temporary_roots):
        raise ValueError(f"{RUNS_DIR} nằm trên đĩa tạm. Đặt MLEUP_CHECKPOINT_ROOT "
                         "thành đường dẫn lưu trữ bền vững đã được cấp quyền.")
    if "MLEUP_CHECKPOINT_ROOT" not in os.environ and not RUNS_DIR.is_relative_to("/Workspace"):
        print("Cảnh báo: vị trí checkpoint nằm ngoài /Workspace. Hãy xác minh checkout "
              "còn tồn tại sau khi restart compute.", flush=True)
try:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=RUNS_DIR, prefix="mleup_write_probe_", suffix=".tmp") as probe:
        probe.write(b"mleup")
        probe.flush()
except OSError as exc:
    raise RuntimeError(f"Không ghi được vào {RUNS_DIR}. Kiểm tra quyền ghi và dung lượng "
                       "hoặc đặt MLEUP_CHECKPOINT_ROOT sang vị trí bền vững khác.") from exc

for name in CSV_NAMES:
    if not (RAW_DIR / name).is_file():
        raise FileNotFoundError(RAW_DIR / name)
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")


def run_script(*args):
    command = [sys.executable, "-X", "utf8", "-u", *args]
    with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding="utf-8", bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Chuẩn bị 53 và 18 trường
# MAGIC Dùng cùng hai CSV. Chọn trường nhị phân từ tập train. Giữ nguyên thư mục
# MAGIC checkpoint khi tiếp tục lượt đang tạm dừng. Dùng root riêng khi thay dữ liệu
# MAGIC đầu vào hoặc hyperparameter ngoài seed, số epoch và batch size.

# COMMAND ----------

prepared = {}
configs = {}
for fields in (53, 18):
    path = EXPERIMENT_DIR / f"prepared_{fields}"
    config_path = path / "config.json"
    if not config_path.is_file():
        run_script(
            "prepare_trivago_top5.py",
            "--train-csv", str(RAW_DIR / "train.csv"),
            "--metadata-csv", str(RAW_DIR / "item_metadata.csv"),
            "--output-dir", str(path), "--scratch-dir", str(scratch),
            "--field-budget", str(fields - 3), "--action-scope", "item_event",
            "--seed", str(SEED),
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if len(config["field_names"]) != fields or config["seed"] != SEED:
        raise ValueError(f"Prepared {path} không khớp {fields} trường hoặc seed {SEED}.")
    prepared[fields], configs[fields] = path, config

for key in ("source_train_sha256", "source_metadata_sha256", "item_id_map_sha256",
            "query_manifest_sha256"):
    if configs[53][key] != configs[18][key]:
        raise ValueError(f"Hai bộ prepared khác nhau ở {key}.")
for split in ("train", "val", "test"):
    key = f"{split}_candidates.txt"
    if configs[53]["artifact_sha256"][key] != configs[18]["artifact_sha256"][key]:
        raise ValueError(f"Hai bộ prepared khác danh sách ứng viên của {split}.")

# COMMAND ----------

def training_command(output, fields, action_mode, price_mode):
    command = [
        "train_trivago_top5.py",
        "--prepared-dir", str(prepared[fields]),
        "--output-dir", str(output),

        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),

        "--patience", str(PATIENCE),
        "--min-delta", str(MIN_DELTA),
        "--loss-patience", str(LOSS_PATIENCE),

        "--lr", "0.001",
        "--lr-factor", "0.5",
        "--lr-patience", "1",
        "--min-lr", "0.00001",

        "--metadata-projection", "fieldwise",
        "--action-mode", action_mode,
        "--price-mode", price_mode,
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--max-hours", str(MAX_HOURS),
    ]
    if TRACK_TIME:
        command.append("--track-time")
    return command


def train_and_evaluate(name, fields, action_mode, price_mode):
    print(f"{name} | {fields} trường | action={action_mode} | price={price_mode}", flush=True)
    output = RUNS_DIR / name
    status_path = output / "run_status.json"
    latest_path = output / "result_latest.pth"
    if not latest_path.is_file() and (output / "result_best.pth").exists():
        raise RuntimeError(f"Thiếu {latest_path} trong khi best checkpoint tồn tại. Kiểm tra các file đã lưu.")
    command = training_command(output, fields, action_mode, price_mode)
    # Only the latest checkpoint can establish whether training is complete.
    run_script(*command, *(["--resume"] if latest_path.is_file() else []))
    status = read_json(status_path, dict) or {}
    if status.get("status") in ("PAUSED", "PAUSED_DURING_VAL", "PAUSED_DURING_VAL_MIGRATION", "SMOKE_STOP"):
        print(f"{name} tạm dừng. Chạy lại cell để tiếp tục từ checkpoint.")
        return
    if status.get("training_finished") is not True:
        raise RuntimeError(f"Không tìm thấy trạng thái hoàn thành hợp lệ trong {status_path}")
    if not valid_test_status(status, TRACK_TIME):
        run_script(*command, "--eval-only", "test")
        status = read_json(status_path, dict) or {}
    if not valid_test_status(status, TRACK_TIME):
        raise RuntimeError(f"Không tìm thấy kết quả test hợp lệ trong {status_path}")
    metrics = status["metrics"]
    print(f"Test | NDCG@5={metrics['ndcg_at_5']:.4f} | Recall@5={metrics['recall_at_5']:.4f} "
          f"| epoch tốt nhất={status['best_epoch']} | {metrics['examples']} mẫu")
    if TRACK_TIME:
        print(f"Test: {metrics['test_time_seconds']:.2f} s | "
              f"Forward: {metrics['inference_ms_per_sample']:.3f} ms/mẫu")
    print(f"Kết quả lưu tại {output}")

# COMMAND ----------

train_and_evaluate("01_baseline_53", 53, "none", "off")

# COMMAND ----------

train_and_evaluate("02_baseline_18", 18, "none", "off")

# COMMAND ----------

train_and_evaluate("03_event_graph_18", 18, "event_graph_candidate", "off")

# COMMAND ----------

train_and_evaluate("04_event_graph_price_18", 18, "event_graph_candidate", "relative_mlp_v1")

# COMMAND ----------

def summarize_runs(runs_dir, track_time):
    model_names = ("01_baseline_53", "02_baseline_18", "03_event_graph_18", "04_event_graph_price_18")
    summary_columns = ["model", "epochs_completed", "best_epoch", "ndcg_at_5", "recall_at_5"]
    if track_time:
        summary_columns += [
            "avg_train_time_min", "avg_val_time_min", "avg_epoch_time_min", "total_train_time_min",
            "total_val_time_min", "total_val_retry_min", "total_epoch_time_min", "train_time_to_best_min",
            "time_to_best_min", "test_time_min", "inference_ms_per_sample", "inference_samples_per_second",
        ]
    epoch_columns = [
        "model", "epoch", *EPOCH_TIMING_FIELDS, "val_ndcg_at_5", "val_recall_at_5",
        "train_total_loss", "val_total_loss",
    ]
    results, epoch_rows = [], []
    for name in model_names:
        folder = runs_dir / name
        status = read_json(folder / "run_status.json", dict) or {}
        history = read_json(folder / "metric_history.json", list)
        if track_time:
            if history is None:
                print(f"{name}: chưa có lịch sử epoch hợp lệ.")
                continue
            try:
                rows = timed_epochs(history)
                details = [{
                    "model": name, "epoch": row["epoch"],
                    **{key: row[key] for key in EPOCH_TIMING_FIELDS},
                    "val_ndcg_at_5": row["val"]["ndcg_at_5"], "val_recall_at_5": row["val"]["recall_at_5"],
                    "train_total_loss": row["train"]["total_loss"], "val_total_loss": row["val"]["total_loss"],
                } for row in rows]
                if any(not numeric(row[key]) for row in details for key in epoch_columns[-4:]):
                    raise ValueError("Metric hoặc loss theo epoch không hợp lệ")
            except (KeyError, TypeError, ValueError) as exc:
                print(f"{name}: lịch sử epoch không hợp lệ ({exc}).")
                continue
            epoch_rows.extend(details)
        if not valid_test_status(status, track_time):
            print(f"{name}: chưa có kết quả test hợp lệ.")
            continue
        best_epoch = status.get("best_epoch")
        if track_time and (type(best_epoch) is not int or not 1 <= best_epoch <= len(rows)
                           or status.get("epochs") != len(rows)
                           or status.get("best_val_ndcg_at_5") != rows[best_epoch - 1]["val"]["ndcg_at_5"]):
            print(f"{name}: trạng thái test không khớp lịch sử epoch.")
            continue
        metrics = status["metrics"]
        result = dict(model=name, best_epoch=best_epoch,
                      ndcg_at_5=metrics["ndcg_at_5"], recall_at_5=metrics["recall_at_5"])
        if track_time:
            df = pd.DataFrame(details)
            best_history = df[df["epoch"] <= best_epoch]
            result.update(
                epochs_completed=len(df),
                avg_train_time_min=df["train_time_seconds"].mean() / 60,
                avg_val_time_min=df["val_time_seconds"].mean() / 60,
                avg_epoch_time_min=df["epoch_time_seconds"].mean() / 60,
                total_train_time_min=df["train_time_seconds"].sum() / 60,
                total_val_time_min=df["val_time_seconds"].sum() / 60,
                total_val_retry_min=df["val_retry_seconds"].sum() / 60,
                total_epoch_time_min=df["epoch_time_seconds"].sum() / 60,
                train_time_to_best_min=best_history["train_time_seconds"].sum() / 60,
                time_to_best_min=(best_history["epoch_time_seconds"].sum()
                                  + best_history["val_retry_seconds"].sum()) / 60,
                test_time_min=metrics["test_time_seconds"] / 60,
                inference_ms_per_sample=metrics["inference_ms_per_sample"],
                inference_samples_per_second=metrics["inference_samples_per_second"],
            )
        elif history is not None:
            result["epochs_completed"] = sum(isinstance(row, dict) and "epoch" in row for row in history)
        results.append(result)

    summary = pd.DataFrame(results, columns=summary_columns)
    timeline = pd.DataFrame(epoch_rows, columns=epoch_columns)
    for frame, filename in ((summary, "experiment_summary.csv"), (timeline, "epoch_timing.csv")):
        path = runs_dir / filename
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    return summary


display(summarize_runs(RUNS_DIR, TRACK_TIME).round(4))
