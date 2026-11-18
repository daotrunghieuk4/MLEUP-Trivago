# Databricks notebook source
# MAGIC %md
# MAGIC # MLEUP · Thử nghiệm dự đoán địa điểm ghé tiếp theo
# MAGIC Điền tên bảng, ba cột và hai mốc chia tập rồi chạy Run all.

# COMMAND ----------

import json
import tempfile
from pathlib import Path

import torch
from pyspark.sql import functions as F

from visit_training import train_visits

for name, default in {
    "table": "", "sequence_col": "user_id", "destination_col": "destination_id",
    "time_col": "event_time", "validation_start": "", "test_start": "",
    "max_rows": "10000", "epochs": "2", "batch_size": "32", "output_dir": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

def get(name):
    return dbutils.widgets.get(name).strip()


table, validation_start, test_start = (get(name) for name in ("table", "validation_start", "test_start"))
if not all((table, validation_start, test_start)):
    raise ValueError("Điền table, validation_start và test_start rồi chạy lại.")
max_rows = int(get("max_rows"))
if max_rows < 1:
    raise ValueError("max_rows phải lớn hơn 0.")

spark.conf.set("spark.sql.session.timeZone", "UTC")
source = spark.table(table)
column_names = [get(name) for name in ("sequence_col", "destination_col", "time_col")]
if len(set(column_names)) != 3 or any(name not in source.columns for name in column_names):
    raise ValueError("Chọn ba tên cột khác nhau, có sẵn trong bảng.")
events = source.select(
    F.col("`" + column_names[0].replace("`", "``") + "`").alias("sequence"),
    F.col("`" + column_names[1].replace("`", "``") + "`").alias("destination"),
    F.col("`" + column_names[2].replace("`", "``") + "`").alias("event_time"),
).limit(max_rows + 1).toPandas()
if len(events) > max_rows:
    raise ValueError("Bảng vượt giới hạn thử nghiệm. Chọn view nhỏ hơn, giữ đầy đủ lịch sử các ID được chọn.")

# COMMAND ----------

output_dir = get("output_dir")
if output_dir and not output_dir.startswith("/Volumes/"):
    raise ValueError("output_dir cần trỏ tới Unity Catalog Volume qua /Volumes/...")
output = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mleup-visits-"))
output.mkdir(parents=True, exist_ok=True)
if (output / "mleup_visits.pt").exists() or (output / "metrics.json").exists():
    raise FileExistsError("Chọn thư mục đầu ra mới cho lần chạy này.")
metrics, checkpoint = train_visits(
    events, validation_start, test_start, epochs=int(get("epochs")), batch_size=int(get("batch_size")),
)
torch.save(checkpoint, output / "mleup_visits.pt")
(output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(f"Artifacts: {output}")
