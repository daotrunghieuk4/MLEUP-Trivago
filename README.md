# MLEUP Trivago

Hệ thống gợi ý khách sạn theo phiên tương tác trên dữ liệu Trivago RecSys Challenge 2019. Mô hình xếp hạng các khách sạn trong danh sách impressions để dự đoán clickout tiếp theo và trả về Top 5.

Phiên bản này phát triển từ MLEUP, kết hợp 18 trường metadata, đồ thị sự kiện với attention phụ thuộc ứng viên và 9 đặc trưng giá tương đối.

Kiến trúc nền được trình bày trong bài báo *Multi-perspective learning for enhanced user preferences for session-based recommendation*, Lan Zhang, Derong Shen, Yue Kou và Tiezheng Nie, Knowledge-Based Systems 298 (2024). [Paper](https://doi.org/10.1016/j.knosys.2024.111997) · [Mã nguồn MLEUP](https://github.com/lean030/MLEUP)

## Kết quả đã ghi nhận

| Tập đánh giá | Số mẫu | NDCG@5 | Recall@5 |
|---|---:|---:|---:|
| Test nội bộ | 158.517 | 0.6231508934 | 0.7027385075 |

Checkpoint được chọn ở epoch 4 theo NDCG@5 trên validation, đạt 0.6236397602. Kết quả test và trạng thái chọn checkpoint được lưu trong [test.log](results/test.log) và [run_status.json](results/run_status.json).

Các số liệu thuộc lần chạy đã lưu. Tập train, validation và test được chia theo thời điểm kết thúc phiên theo tỷ lệ 60/20/20 từ `train.csv` có nhãn. Đây là kết quả đánh giá nội bộ. So sánh với mô hình khác cần thống nhất dữ liệu, cách chia tập, tập ứng viên và ngân sách huấn luyện.

## Cấu trúc

```text
MLEUP-Trivago/
├── databricks_mleup_ablation.py   # Notebook chính
├── src/
│   ├── model.py, util.py          # Model, dataset, batch và đồ thị
│   ├── prepare_trivago_top5.py    # Tiền xử lý
│   ├── train_trivago_top5.py      # Train, resume, evaluation và timing
│   ├── trivago_action_types.py
│   ├── trivago_price_features.py
│   ├── trivago_top5_checkpoint.py
│   ├── trivago_data_parts.py
│   ├── experiment_runner.py      # Điều phối notebook và subprocess
│   ├── experiment_reporting.py   # Tổng hợp JSON và CSV
│   └── run_metadata.py           # Metadata và kiểm tra báo cáo
├── data_parts/                   # Manifest và các phần nén của hai CSV
├── tests/                        # Kiểm thử và fixture nhỏ
├── results/                      # Kết quả nghiên cứu đã lưu trong Git
├── runs/                         # Checkpoint và báo cáo phát sinh, bỏ qua bởi Git
├── requirements.txt
└── .gitignore, .gitattributes
```

Các file `train_trivago_top5.py`, `prepare_trivago_top5.py` và `trivago_data_parts.py` ở thư mục gốc là wrapper cho CLI và import cũ. Implementation nằm trong `src/`. Thư mục `data/` và `runs/` được tạo khi chuẩn bị dữ liệu và huấn luyện.

## Cài đặt

Chạy các lệnh từ thư mục gốc repository. Dùng Python 3.11 và môi trường ảo riêng.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên Linux/macOS, kích hoạt bằng `source .venv/bin/activate`. Huấn luyện toàn bộ dữ liệu cần tài nguyên đáng kể, nên dùng GPU với bản PyTorch phù hợp CUDA của máy. Bộ dependencies liệt kê các thư viện trực tiếp sử dụng, chưa khóa môi trường Kaggle của lần chạy gốc.

## Tải dữ liệu

Sử dụng bộ **Trivago RecSys Challenge 2019**. Trang cuộc thi dẫn tới cổng dữ liệu của Trivago và yêu cầu tạo tài khoản, chấp nhận điều khoản trước khi tải. [Trang cuộc thi](https://www.recsyschallenge.com/2019/) · [Cổng dữ liệu](https://recsys.trivago.cloud/challenge/dataset/)

1. Mở trang cuộc thi và tìm mục **Participation and Data**.
2. Truy cập cổng dữ liệu, đăng nhập hoặc đăng ký theo hướng dẫn của nhà cung cấp.
3. Tải gói dữ liệu và giải nén bằng công cụ phù hợp với định dạng tệp nhận được.
4. Tìm `train.csv` và `item_metadata.csv` trong thư mục đã giải nén, sau đó sao chép vào `data/raw/`.

```text
data/
└── raw/
    ├── train.csv
    └── item_metadata.csv
```

Cổng tải cũ chưa truy cập được trong lần kiểm tra ngày 17/11/2026. Khi gặp lỗi, kiểm tra thông báo hoặc liên hệ ban tổ chức từ trang cuộc thi để lấy địa chỉ hiện hành. Nếu đã có bản dữ liệu tải trước đó, có thể đặt trực tiếp hai tệp vào thư mục trên và kiểm tra checksum.

| Tệp | Nội dung | Vai trò |
|---|---|---|
| `train.csv` | Sự kiện trong phiên, impressions và prices | Tạo ba tập train/validation/test nội bộ |
| `item_metadata.csv` | `item_id` và `properties` | Tạo metadata khách sạn |
| `test.csv` trong gói cuộc thi | Phiên có nhãn clickout được che | Dành cho cuộc thi, pipeline hiện tại chỉ đọc hai tệp phía trên |

Định nghĩa các cột và loại hành động được mô tả trong tài liệu của ban tổ chức. [Data specification](https://github.com/recsyschallenge/2019/blob/master/docs/problem_definition.md)

### Kiểm tra phiên bản dữ liệu

Hai checksum SHA-256 sau được lưu trong cấu hình thí nghiệm đã báo cáo.

| Tệp | SHA-256 |
|---|---|
| `train.csv` | `285f07abbc87ea22e608ce194ece7bcc1047926339e269bbf5b5fd477a120ee0` |
| `item_metadata.csv` | `b1757ea4eded7475c84763391ef8cff00ba707590886b61f6a6c4f2f76979d2e` |

Trên Windows PowerShell, chạy lệnh sau và đối chiếu cột `Hash`.

```powershell
Get-FileHash data/raw/train.csv, data/raw/item_metadata.csv -Algorithm SHA256
```

Trên Linux dùng `sha256sum data/raw/train.csv data/raw/item_metadata.csv`. Trên macOS dùng `shasum -a 256 data/raw/train.csv data/raw/item_metadata.csv`. Checksum khác có thể phản ánh khác biệt phiên bản hoặc nội dung tệp, cần kiểm tra trước khi đối chiếu kết quả thí nghiệm.

## Thiết lập Databricks

1. Tạo Databricks Git Folder từ repository, hoặc sao lưu thay đổi và checkpoint rồi **Pull** nếu đã có Git Folder.
2. Mở `databricks_mleup_ablation.py` ở thư mục gốc. Chỉnh dữ liệu và hyperparameter trong **Configuration**. Dùng cùng loại compute và GPU cho cả bốn cấu hình.
3. Chạy **Environment and Checkpoint Status**. Đối chiếu GPU, runtime, quyền ghi và `RUNS_DIR` với lượt cũ. Bảng trạng thái đọc JSON và kiểm tra sự hiện diện của tệp, còn trainer kiểm tra payload khi resume.
4. Chạy **Data Preparation** và **Experiment Pipeline**. Notebook xác minh SHA-256 của các phần nén khi khôi phục CSV. Nó kiểm tra fingerprint của artifact, metadata, mapping, query manifest và candidate lists trước khi tái sử dụng hai bộ prepared. Ba mô hình 18 trường dùng chung một bộ. Nếu bộ prepared đang có bị hỏng, dùng vị trí prepared sạch hoặc khởi động lại phiên để tạo scratch mới.

Mặc định checkpoint được ghi trực tiếp vào `runs/databricks_checkpoints/seed2023_e50_b128_timed_v1/`. Có thể đặt biến môi trường `MLEUP_CHECKPOINT_ROOT` sang vị trí bền vững đã được cấp quyền. Khi thay dữ liệu hoặc hyperparameter làm checkpoint cũ không tương thích, dùng nhóm run mới. Nếu Git Folder không có `data_parts/`, notebook dùng `RAW_DIR` và `EXPERIMENT_DIR` trong Configuration.

### Chạy và resume thực nghiệm

| Cell | Experiment ID | Thay đổi nghiên cứu |
|---|---|---|
| Baseline 53 | `01_baseline_53` | Baseline MLEUP với 53 trường metadata |
| Baseline 18 | `02_baseline_18` | Giảm metadata còn 18 trường |
| Improvement 1 | `03_event_graph_18` | Event Graph, attention theo ứng viên và loss trên impressions |
| Improvement 2 | `04_event_graph_price_18` | Bổ sung Relative Price Features |

Chạy riêng cell của mô hình cần train hoặc resume, hoặc dùng **Run All**. Mỗi mô hình có `result_latest.pth` để tiếp tục huấn luyện và `result_best.pth` để test theo NDCG@5 validation. Trainer lưu latest mỗi 1.000 bước optimizer, trước và sau validation, khi tạm dừng có kiểm soát và khi kết thúc. Khi có best mới, trainer ghi latest trước rồi xuất bản best từ cùng payload. Chạy mỗi thư mục mô hình bằng một trainer tại một thời điểm.

Sau khi mất phiên, chạy lại các cell 1–4 để khôi phục môi trường và prepared nếu scratch đã mất. Chạy lại đúng cell mô hình. Các batch sau lần lưu latest cuối cùng có thể phải chạy lại. Nếu latest đã lưu hết batch của epoch, trainer chỉ chạy lại validation. Mô hình đã train xong sẽ test best nếu thiếu kết quả hợp lệ, hoặc đọc kết quả đã lưu.

### Timing và báo cáo thực nghiệm

Train time tính phần chuẩn bị và xử lý batch huấn luyện. Validation time tính lần validation thành công, còn lượt validation dừng có kiểm soát được ghi riêng dưới dạng retry. Time to best cộng train, validation thành công và retry đã ghi nhận đến best epoch. Test time tính toàn bộ evaluation sau lúc chuẩn bị tập test. Inference ms/mẫu và throughput tính từ forward theo batch và số mẫu thực tế. So sánh timing cần GPU, runtime, batch size và cách đo tương đương. Đặt `TRAINING["track_time"] = False` để chạy nhóm không đo thời gian.

**Experiment Summary** có thể chạy ngay sau **Configuration** để đọc một `RUNS_DIR` hiện có. Các file trong `RUNS_DIR` cũng được cập nhật tự động khi một mô hình tạm dừng có kiểm soát, hoàn thành hoặc được mở lại.

| File | Nội dung |
|---|---|
| `best_experiments.json` | File tra cứu chính. Ghi run ID, trạng thái cả bốn mô hình, hyperparameter thực tế, best epoch và loss/metric tại epoch đó, test, timing, môi trường và đường dẫn checkpoint |
| `experiment_summary.csv` | Accuracy và chi phí tính toán của các lượt đã train và test hợp lệ |
| `epoch_timing.csv` | Timing, train/validation loss và validation metric theo epoch, kể cả lượt chưa hoàn thành |
| `ablation_comparison.csv` | Ba cặp ablation theo thứ tự nghiên cứu. `train_time_before_min` và `train_time_after_min` là thời gian train trung bình mỗi epoch |

`run_status.json` ghi hyperparameter thực tế và thông tin phiên đã dùng. Report đối chiếu trạng thái với lịch sử đầy đủ và metadata của tệp checkpoint, còn các tensor nằm trong `result_best.pth`. Các trường cũ chưa từng ghi nhận được để `null`. Best epoch thuộc một run cụ thể và được chọn theo NDCG@5 validation. Các hạng theo test metric chỉ mô tả kết quả của nhóm đang xem. Chênh lệch tương đối để trống khi mẫu số bằng 0 hoặc thiếu dữ liệu. Khi môi trường không tương đương, báo cáo cảnh báo và để trống tỷ lệ thay đổi thời gian của cặp tương ứng.

### Thêm cấu hình thực nghiệm

Thêm một ID duy nhất vào `EXPERIMENTS` trong Configuration, khai báo số trường, action/price mode, tên hiển thị và mô tả so sánh với cấu hình liền trước. Thêm một cell gọi `run_experiment(experiment_id)`. Pipeline tự gom các cấu hình cùng số trường vào một bộ prepared chung và báo cáo theo thứ tự trong `EXPERIMENTS`. Giữ ID cũ cho các lượt cần resume.

#### Kiểm tra checkpoint qua lần restart compute

Sau các cell 1–4, chạy smoke test trong một cell riêng trên compute có thể restart an toàn.

```python
smoke = CHECKPOINT_ROOT.expanduser().resolve() / "_smoke_timing_v1"
if smoke.exists():
    raise FileExistsError(smoke)
spec = EXPERIMENTS["02_baseline_18"]
smoke_training = {**TRAINING, "epochs": 2, "checkpoint_every": 1, "track_time": True}
command = experiment_runner.build_train_command(spec, smoke_training, prepared[spec["fields"]], smoke)
experiment_runner.run_script(REPO, *command, "--max-batches", "2")
```

Ghi lại đường dẫn, kích thước và trạng thái bằng cell sau. Chỉ đọc checkpoint do chính lượt chạy tạo.

```python
import torch

smoke = CHECKPOINT_ROOT.expanduser().resolve() / "_smoke_timing_v1"
path = smoke / "result_latest.pth"
assert path.is_file(), path
cp = torch.load(path, map_location="cpu", weights_only=False)
print(path.resolve(), path.stat().st_size, "bytes")
print(cp["epoch"], cp["global_step"], cp["next_offset"], cp["timing_state"])
del cp
```

Restart compute. Chạy lại Configuration rồi **chỉ chạy cell đọc** checkpoint và so sánh với giá trị đã ghi. Chạy các cell 2–4 để tạo lại prepared nếu cần. Resume smoke bằng lệnh sau rồi đọc lại checkpoint. Epoch vẫn là `0`, `global_step` tăng từ `2` lên `4`, `next_offset` từ `2 * TRAINING["batch_size"]` lên `4 * TRAINING["batch_size"]`, và `train_time_seconds` tăng.

```python
spec = EXPERIMENTS["02_baseline_18"]
smoke_training = {**TRAINING, "epochs": 2, "checkpoint_every": 1, "track_time": True}
command = experiment_runner.build_train_command(spec, smoke_training, prepared[spec["fields"]], smoke)
experiment_runner.run_script(REPO, *command, "--resume", "--max-batches", "4")
```

Kiểm tra dung lượng thực tế đủ cho latest, best và tệp tạm khi ghi, tính trên cả bốn cấu hình. Workspace Files có giới hạn 500 MB mỗi tệp. Nếu smoke không tồn tại sau restart, vượt giới hạn hoặc storage không hỗ trợ khóa ghi và thay thế file, đặt `MLEUP_CHECKPOINT_ROOT` sang đường dẫn bền vững khác đã được cấp quyền và thử lại. [Giới hạn Workspace Files của Databricks](https://docs.databricks.com/aws/en/files/workspace#limitations)

Để đóng gói lại từ hai CSV đã giải nén, chạy ở thư mục gốc repo. Mỗi phần mặc định tối đa 90 MiB để dùng Git thường trên GitHub.

```bash
python -m src.trivago_data_parts data/raw artifacts/trivago_git_parts
```

## Chuẩn bị dữ liệu trên local

Đặt hai tệp dữ liệu Trivago tại `data/raw/train.csv` và `data/raw/item_metadata.csv`. Các cột sự kiện được sử dụng gồm user/session, timestamp, action, reference, impressions và prices. Preprocessor kiểm tra schema khi đọc dữ liệu.

```powershell
python -m src.prepare_trivago_top5 --train-csv data/raw/train.csv --metadata-csv data/raw/item_metadata.csv --output-dir data/prepared --field-budget 15 --action-scope item_event --max-session-items 20 --max-session-events 64 --max-neighbors 400 --m 14 --frequency 10 --seed 2023
```

`--field-budget 15` chọn 15 thuộc tính nhị phân theo các bộ lọc xuất hiện trong tập train. Kết hợp ba trường `star_rating`, `from_star_max`, `rating_highest`, mô hình có tổng cộng 18 trường metadata. Cấu hình và checksum của dữ liệu trong lần chạy đã báo cáo nằm tại [prepared_config.json](results/prepared_config.json).

Preprocessor tạo mapping khách sạn, metadata, đồ thị, các tập ứng viên và giá tương ứng. Cần giữ toàn bộ thư mục prepared đi cùng checkpoint để bảo toàn mapping, schema và scaler. Chỉ nạp checkpoint và pickle từ nguồn tin cậy.

## Huấn luyện

```powershell
python -m src.train_trivago_top5 --prepared-dir data/prepared --output-dir runs/relative_price_v1 --epochs 5 --batch-size 256 --emb-size 16 --num-heads 4 --metadata-projection fieldwise --action-mode event_graph_candidate --price-mode relative_mlp_v1 --lr 0.001 --lr-factor 0.5 --lr-patience 1 --min-lr 0.00001 --min-delta 0.0001 --patience 3 --loss-patience 3 --loss-min-delta 0.001
```

Mô hình học biểu diễn metadata và lịch sử hành động, tổng hợp thông tin sự kiện theo từng ứng viên rồi kết hợp đặc trưng giá để xếp hạng. Chính sách lịch sử ghi nhận clickout trong train và che clickout trong validation/test. Trainer chọn checkpoint bằng NDCG@5 validation và áp dụng điều kiện dừng theo metric/loss.

Để tiếp tục lần chạy bị gián đoạn, chạy lại lệnh với `--resume`, giữ cùng prepared, output và cấu hình mô hình. Thêm `--track-time` từ lúc bắt đầu một lượt mới nếu cần ghi thời gian từng epoch và giữ cờ này khi resume hoặc đánh giá. Các checkpoint và báo cáo được lưu trong thư mục output.

## Đánh giá checkpoint

```powershell
python -m src.train_trivago_top5 --prepared-dir data/prepared --output-dir runs/relative_price_v1 --batch-size 256 --emb-size 16 --num-heads 4 --metadata-projection fieldwise --action-mode event_graph_candidate --price-mode relative_mlp_v1 --eval-only test
```

Lệnh đánh giá xác nhận lượt chạy đã hoàn tất từ `result_latest.pth` rồi nạp `result_best.pth`. Để đánh giá lại kết quả đã báo cáo từ bản lưu cũ, có thể dùng checkpoint best epoch 4 với `run_status.json` hoàn thành và toàn bộ prepared tương ứng của bộ `context_price_v1` trong chế độ không đo thời gian. Với `--track-time`, test time bao gồm toàn bộ evaluation sau khi load checkpoint và chuẩn bị tập test. Inference time chỉ tính forward hiện tại. Việc huấn luyện lại có thể cho số liệu khác do môi trường và tính ngẫu nhiên.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m py_compile databricks_mleup_ablation.py
python -m src.prepare_trivago_top5 --help
python -m src.train_trivago_top5 --help
```

Các kiểm thử bao phủ prepared integrity, cấu hình bốn mô hình, hành động, đồ thị sự kiện, đặc trưng giá, checkpoint, timing và báo cáo độc lập.

## Trích dẫn

Khi sử dụng kiến trúc MLEUP trong nghiên cứu, trích dẫn bài báo gốc. [Thông tin xuất bản](https://doi.org/10.1016/j.knosys.2024.111997)

```bibtex
@article{zhang2024mleup,
  title   = {Multi-perspective learning for enhanced user preferences for session-based recommendation},
  author  = {Zhang, Lan and Shen, Derong and Kou, Yue and Nie, Tiezheng},
  journal = {Knowledge-Based Systems},
  volume  = {298},
  pages   = {111997},
  year    = {2024},
  doi     = {10.1016/j.knosys.2024.111997}
}
```

Với dữ liệu Trivago, tham khảo công bố của ban tổ chức. [RecSys Challenge 2019 — Session-based Hotel Recommendations](https://doi.org/10.1145/3298689.3346974)

## Acknowledgements

Dự án kế thừa kiến trúc và mã nguồn MLEUP, đồng thời bổ sung pipeline Trivago, đồ thị sự kiện theo ứng viên và giá tương đối. Cảm ơn các tác giả MLEUP và ban tổ chức RecSys Challenge 2019 đã cung cấp tài liệu nghiên cứu và dữ liệu.

## License

Giấy phép phân phối của phiên bản này đang chờ xác nhận đối với phần mã kế thừa. Việc sử dụng dữ liệu tuân theo điều khoản do nhà cung cấp đưa ra.
