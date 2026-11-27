# Databricks notebook source
# MAGIC %md
# MAGIC # Thực nghiệm MLEUP trên Trivago
# MAGIC Mở notebook ở thư mục gốc Git Folder. Chạy từng cell mô hình để train hoặc resume.
# MAGIC Để xem lại kết quả đã lưu, chỉ cần chạy Configuration và Experiment Summary.

# COMMAND ----------

# 1. Configuration
import os
from pathlib import Path

REPO = Path.cwd().resolve()
if not (REPO / "src" / "train_trivago_top5.py").is_file():
    raise FileNotFoundError(f"Đặt notebook tại thư mục gốc MLEUP-Trivago. CWD hiện tại: {REPO}")
RAW_DIR = Path("/Volumes/<catalog>/<schema>/<volume>/raw")
EXPERIMENT_DIR = RAW_DIR.parent / "ablation_v1"
SEED = 2023
TRAINING = {
    "epochs": 50,
    "batch_size": 128,
    "emb_size": 16,
    "num_heads": 4,
    "sample_order": 3,
    "si_dc": 2.0,
    "beta": 0.3,
    "alpha": 0.2,
    "lr": 0.001,
    "lr_factor": 0.5,
    "lr_patience": 1,
    "min_lr": 0.00001,
    "patience": 5,
    "min_delta": 0.0001,
    "loss_patience": 50,
    "loss_min_delta": 0.001,
    "metadata_projection": "fieldwise",
    "checkpoint_every": 1000,
    "max_hours": 8.5,
    "track_time": True,
}
EXPERIMENTS = {
    "01_baseline_53": dict(display_name="Baseline 53", fields=53, action_mode="none", price_mode="off"),
    "02_baseline_18": dict(display_name="Baseline 18", fields=18, action_mode="none", price_mode="off",
                           comparison="Giảm metadata từ 53 xuống 18 trường"),
    "03_event_graph_18": dict(display_name="Improvement 1", fields=18, action_mode="event_graph_candidate",
                              price_mode="off", comparison="Event Graph, attention theo ứng viên và loss trên impressions"),
    "04_event_graph_price_18": dict(display_name="Improvement 2", fields=18, action_mode="event_graph_candidate",
                                    price_mode="relative_mlp_v1", comparison="Thêm Relative Price Features"),
}
CHECKPOINT_ROOT = Path(os.environ.get("MLEUP_CHECKPOINT_ROOT", REPO / "runs" / "databricks_checkpoints"))
run_tag = "_timed_v1" if TRAINING["track_time"] else ""
RUNS_DIR = (CHECKPOINT_ROOT.expanduser() /
            f"seed{SEED}_e{TRAINING['epochs']}_b{TRAINING['batch_size']}{run_tag}").resolve()

# COMMAND ----------

# 2. Environment and Checkpoint Status
import pandas as pd

from src.experiment_runner import check_environment, inspect_run_status, validate_configuration

validate_configuration(EXPERIMENTS, TRAINING, RUNS_DIR)
environment = check_environment(REPO, RUNS_DIR, custom_storage="MLEUP_CHECKPOINT_ROOT" in os.environ)
display(pd.DataFrame([environment]))
display(pd.DataFrame(inspect_run_status(RUNS_DIR, EXPERIMENTS)))

# COMMAND ----------

# 3. Data Preparation
import tempfile

from src.experiment_runner import prepare_datasets

if "scratch" not in globals() or not scratch.is_dir():
    scratch = Path(tempfile.mkdtemp(prefix="mleup_ablation_"))
prepared = prepare_datasets(REPO, RAW_DIR, EXPERIMENT_DIR, scratch, EXPERIMENTS, SEED)

# COMMAND ----------

# 4. Experiment Pipeline
from src import experiment_runner
from src.experiment_reporting import export_results


def run_experiment(experiment_id):
    experiment_runner.run_experiment(experiment_id, repo=REPO, runs_dir=RUNS_DIR, experiments=EXPERIMENTS,
                                     training=TRAINING, prepared=prepared, seed=SEED)
    export_results(RUNS_DIR, EXPERIMENTS)

# COMMAND ----------

# 5. Baseline 53
run_experiment("01_baseline_53")

# COMMAND ----------

# 6. Baseline 18
run_experiment("02_baseline_18")

# COMMAND ----------

# 7. Improvement 1
run_experiment("03_event_graph_18")

# COMMAND ----------

# 8. Improvement 2
run_experiment("04_event_graph_price_18")

# COMMAND ----------

# 9. Experiment Summary
import pandas as pd

from src.experiment_reporting import export_results

report, summary, _, ablation = export_results(RUNS_DIR, EXPERIMENTS)
display(pd.DataFrame([dict(model=name, status=result["status"],
                           best_epoch=(result["best_epoch"] or {}).get("epoch"),
                           **result["model_configuration"], **result["hyperparameters"])
                      for name, result in report["experiments"].items()]))
display(summary.round(4))
display(ablation.round(4))
for warning in report["warnings"]:
    print(warning)
for filename in ("best_experiments.json", "experiment_summary.csv", "epoch_timing.csv", "ablation_comparison.csv"):
    print(RUNS_DIR / filename)
