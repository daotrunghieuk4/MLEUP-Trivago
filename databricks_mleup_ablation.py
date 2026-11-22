# Databricks notebook source
# MAGIC %md
# MAGIC # Thực nghiệm MLEUP trên Trivago
# MAGIC Đặt notebook tại thư mục gốc Git Folder `MLEUP-Trivago`, trên compute có `numpy` và `torch`.
# MAGIC Chỉnh đường dẫn CSV và Volume lưu kết quả ở cell dưới. Dùng cùng seed, batch size,
# MAGIC số epoch tối đa và cách chọn checkpoint theo NDCG@5 trên validation cho cả bốn lượt.
# MAGIC Nếu có `data_parts/manifest.json` cạnh notebook, cell cấu hình tự ghép các phần,
# MAGIC kiểm tra SHA-256 và giải nén dữ liệu vào đĩa tạm. Kết quả ở đó có thể mất khi compute dừng.
# MAGIC Sau cell chuẩn bị dữ liệu, chạy từng cell train hoặc chọn **Run All**. Mỗi cell
# MAGIC in tiến độ theo epoch và kết quả test khi hoàn tất. Cell cuối tổng hợp kết quả.
# MAGIC
# MAGIC Hai baseline là bản MLEUP thích nghi cho Trivago. Cải tiến 1 thêm đồ thị sự kiện và
# MAGIC attention theo ứng viên, đồng thời chuyển loss từ toàn bộ danh mục sang danh sách
# MAGIC impressions. Chênh lệch giữa hai lượt này phản ánh cả hai thay đổi. Cải tiến 2
# MAGIC thêm giá tương đối trên cấu hình của cải tiến 1. Cả bốn đánh giá trên impressions.

# COMMAND ----------

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from trivago_data_parts import CSV_NAMES, unpack

RAW_DIR = Path("/Volumes/<catalog>/<schema>/<volume>/raw")
EXPERIMENT_DIR = RAW_DIR.parent / "ablation_v1"
EPOCHS = 50
BATCH_SIZE = 128
SEED = 2023

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
RUNS_DIR = EXPERIMENT_DIR / f"runs_e{EPOCHS}_b{BATCH_SIZE}"

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
# MAGIC thí nghiệm khi tiếp tục lượt đang tạm dừng. Đổi `EXPERIMENT_DIR` khi thay
# MAGIC dữ liệu đầu vào hoặc seed. Epoch và batch size tạo thư mục kết quả riêng.

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

def train_and_evaluate(name, fields, action_mode, price_mode):
    print(f"{name} | {fields} trường | action={action_mode} | price={price_mode}", flush=True)
    saved = RUNS_DIR / name
    output = scratch / name
    if saved.is_dir() and not output.exists():
        shutil.copytree(saved, output)
    status_path = output / "run_status.json"
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
    ]
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if not status.get("training_finished"):
        resume = ["--resume"] if (output / "result_latest.pth").is_file() else []
        run_script(*command, *resume)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        shutil.copytree(output, saved, dirs_exist_ok=True, copy_function=shutil.copyfile)
    if not status.get("training_finished"):
        print(f"{name} tạm dừng. Chạy lại cell để tiếp tục từ checkpoint.")
        return
    if status.get("status") != "EVALUATED":
        run_script(*command, "--eval-only", "test")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        shutil.copyfile(status_path, saved / status_path.name)
    metrics = status["metrics"]
    print(f"Test | NDCG@5={metrics['ndcg_at_5']:.4f} | Recall@5={metrics['recall_at_5']:.4f} "
          f"| epoch tốt nhất={status['best_epoch']} | {metrics['examples']} mẫu")
    print(f"Kết quả lưu tại {saved}")

# COMMAND ----------

train_and_evaluate("01_baseline_53", 53, "none", "off")

# COMMAND ----------

train_and_evaluate("02_baseline_18", 18, "none", "off")

# COMMAND ----------

train_and_evaluate("03_event_graph_18", 18, "event_graph_candidate", "off")

# COMMAND ----------

train_and_evaluate("04_event_graph_price_18", 18, "event_graph_candidate", "relative_mlp_v1")

# COMMAND ----------

print(f"{'Cấu hình':<28} {'NDCG@5':>9} {'Recall@5':>9}")
for name in ("01_baseline_53", "02_baseline_18", "03_event_graph_18", "04_event_graph_price_18"):
    path = RUNS_DIR / name / "run_status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if status.get("status") == "EVALUATED":
        metrics = status["metrics"]
        print(f"{name:<28} {metrics['ndcg_at_5']:>9.4f} {metrics['recall_at_5']:>9.4f}")
    else:
        print(f"{name:<28} {'Chưa có kết quả test'}")
