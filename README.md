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
├── model.py                      # Kiến trúc MLEUP và tính điểm
├── util.py                       # Dataset, batch và đồ thị
├── prepare_trivago_top5.py       # Chuyển dữ liệu thô thành dữ liệu mô hình
├── train_trivago_top5.py         # Huấn luyện, resume và đánh giá
├── trivago_action_types.py      # Schema hành động
├── trivago_price_features.py    # Đặc trưng và chuẩn hóa giá
├── trivago_top5_checkpoint.py   # Kiểm tra checkpoint, early stopping
├── tests/                       # Kiểm thử hiện có
├── results/                     # Bằng chứng kết quả và cấu hình
├── requirements.txt
└── .gitignore
```

Các module nằm cùng cấp để giữ nguyên các import và lệnh chạy của bộ model. Thư mục `data/` và `runs/` được tạo khi chuẩn bị dữ liệu và huấn luyện.

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

### Dùng bộ CSV chia phần trên Databricks

Sao lưu thay đổi và checkpoint hiện có trước khi **Pull** trong Databricks Git Folder. Mở `databricks_mleup_ablation.py` ở thư mục gốc trên cùng loại compute/GPU cho cả bốn cấu hình. Chạy cell cấu hình rồi cell chuẩn bị dữ liệu. Notebook kiểm tra SHA-256 của `data_parts/`, khôi phục hai CSV vào đĩa tạm và dựng lại hai bộ prepared khi cần.

Các cell training ghi trực tiếp vào `runs/databricks_checkpoints/seed2023_e50_b128_timed_v1/`. Mỗi cấu hình có `result_latest.pth` để tiếp tục huấn luyện và `result_best.pth` để test theo NDCG@5 validation. Trainer lưu latest mỗi 1.000 bước optimizer, trước và sau validation, khi tạm dừng có kiểm soát, và khi kết thúc hoặc early stop. Khi epoch lập best mới, trainer lưu latest trước rồi xuất bản best từ cùng payload. Nếu bị ngắt giữa hai lần ghi, lần resume có thể khôi phục best từ latest của chính epoch được chọn. Chạy mỗi thư mục output bằng một trainer tại một thời điểm.

Nếu phiên bị ngắt, chạy lại cell cấu hình và preprocessing nếu scratch đã mất. Chạy lại cell của mô hình tương ứng để resume. Lần chạy đã huấn luyện xong sẽ test checkpoint best nếu thiếu kết quả test hợp lệ. Các batch sau lần lưu latest cuối cùng có thể phải chạy lại. Validation bị ngắt sẽ chạy lại từ đầu mà không lặp lại các batch huấn luyện đã lưu.

Chạy cell cuối để lấy `experiment_summary.csv` và `epoch_timing.csv` trong thư mục run. Bảng tổng hợp chỉ chứa các cấu hình đã train và test xong. Bảng theo epoch có cả epoch hoàn thành của các lượt đang tạm dừng. `time_to_best_min` cộng thời gian train, validation thành công và retry validation đã ghi nhận qua best epoch. Đối chiếu số epoch và môi trường GPU/runtime trước khi so sánh. Inference ms/mẫu là thời gian forward bình quân theo batch size đã dùng.

Giữ nguyên đường dẫn khi resume. Khi thay dataset hoặc hyperparameter ngoài seed, epoch budget và batch size, chọn root mới. Có thể đặt biến môi trường `MLEUP_CHECKPOINT_ROOT` thành đường dẫn bền vững đã được cấp quyền trước khi chạy cell cấu hình. Giữ checkpoint cũ ở thư mục riêng, vì lượt `timed_v1` cần số liệu từ đầu. Khi đặt `TRACK_TIME=False`, notebook vẫn tổng hợp NDCG@5 và Recall@5 trong thư mục run riêng.

#### Kiểm tra checkpoint qua lần restart compute

Sau cell preprocessing và cell định nghĩa `training_command`, chạy smoke test trong một cell riêng trên compute có thể restart an toàn:

```python
smoke = CHECKPOINT_ROOT / "_smoke_timing_v1"
if smoke.exists():
    raise FileExistsError(smoke)
command = training_command(smoke, 18, "none", "off")
assert "--track-time" in command
run_script(*command, "--epochs", "2", "--checkpoint-every", "1", "--max-batches", "2")
```

Ghi lại đường dẫn, kích thước và trạng thái bằng cell sau. Chỉ đọc checkpoint do chính lượt chạy tạo.

```python
smoke = CHECKPOINT_ROOT / "_smoke_timing_v1"
path = smoke / "result_latest.pth"
assert path.is_file(), path
cp = torch.load(path, map_location="cpu", weights_only=False)
print(path.resolve(), path.stat().st_size, "bytes")
print(cp["epoch"], cp["global_step"], cp["next_offset"], cp["timing_state"])
del cp
```

Restart compute. Chạy lại cấu hình và định nghĩa hàm cần thiết, rồi **chỉ chạy cell đọc** và so sánh với giá trị đã ghi. Dựng lại prepared nếu scratch mất. Resume smoke bằng lệnh sau rồi đọc lại checkpoint. Kỳ vọng epoch vẫn là `0`, `global_step` tăng từ `2` lên `4`, `next_offset` từ `2 * BATCH_SIZE` lên `4 * BATCH_SIZE`, và `train_time_seconds` tăng.

```python
command = training_command(smoke, 18, "none", "off")
run_script(*command, "--epochs", "2", "--checkpoint-every", "1",
           "--resume", "--max-batches", "4")
```

Kiểm tra dung lượng thực tế đủ cho latest, best và tệp tạm khi ghi, tính trên cả bốn cấu hình. Workspace Files có giới hạn 500 MB mỗi tệp. Nếu smoke không tồn tại sau restart, vượt giới hạn hoặc storage không hỗ trợ khóa ghi và thay thế file, đặt `MLEUP_CHECKPOINT_ROOT` sang đường dẫn bền vững khác đã được cấp quyền và thử lại. [Giới hạn Workspace Files của Databricks](https://docs.databricks.com/aws/en/files/workspace#limitations)

Để đóng gói lại từ hai CSV đã giải nén, chạy ở thư mục gốc repo. Mỗi phần mặc định tối đa 90 MiB để dùng Git thường trên GitHub.

```bash
python trivago_data_parts.py data/raw artifacts/trivago_git_parts
```

Nếu Git Folder không có `data_parts/`, notebook dùng `RAW_DIR` và `EXPERIMENT_DIR` như trước.

## Chuẩn bị dữ liệu

Đặt hai tệp dữ liệu Trivago tại `data/raw/train.csv` và `data/raw/item_metadata.csv`. Các cột sự kiện được sử dụng gồm user/session, timestamp, action, reference, impressions và prices. Preprocessor kiểm tra schema khi đọc dữ liệu.

```powershell
python prepare_trivago_top5.py --train-csv data/raw/train.csv --metadata-csv data/raw/item_metadata.csv --output-dir data/prepared --field-budget 15 --action-scope item_event --max-session-items 20 --max-session-events 64 --max-neighbors 400 --m 14 --frequency 10 --seed 2023
```

`--field-budget 15` chọn 15 thuộc tính nhị phân theo các bộ lọc xuất hiện trong tập train. Kết hợp ba trường `star_rating`, `from_star_max`, `rating_highest`, mô hình có tổng cộng 18 trường metadata. Cấu hình và checksum của dữ liệu trong lần chạy đã báo cáo nằm tại [prepared_config.json](results/prepared_config.json).

Preprocessor tạo mapping khách sạn, metadata, đồ thị, các tập ứng viên và giá tương ứng. Cần giữ toàn bộ thư mục prepared đi cùng checkpoint để bảo toàn mapping, schema và scaler. Chỉ nạp checkpoint và pickle từ nguồn tin cậy.

## Huấn luyện

```powershell
python train_trivago_top5.py --prepared-dir data/prepared --output-dir runs/relative_price_v1 --epochs 5 --batch-size 256 --emb-size 16 --num-heads 4 --metadata-projection fieldwise --action-mode event_graph_candidate --price-mode relative_mlp_v1 --lr 0.001 --lr-factor 0.5 --lr-patience 1 --min-lr 0.00001 --min-delta 0.0001 --patience 3 --loss-patience 3 --loss-min-delta 0.001
```

Mô hình học biểu diễn metadata và lịch sử hành động, tổng hợp thông tin sự kiện theo từng ứng viên rồi kết hợp đặc trưng giá để xếp hạng. Chính sách lịch sử ghi nhận clickout trong train và che clickout trong validation/test. Trainer chọn checkpoint bằng NDCG@5 validation và áp dụng điều kiện dừng theo metric/loss.

Để tiếp tục lần chạy bị gián đoạn, chạy lại lệnh với `--resume`, giữ cùng prepared, output và cấu hình mô hình. Thêm `--track-time` từ lúc bắt đầu một lượt mới nếu cần ghi thời gian từng epoch và giữ cờ này khi resume hoặc đánh giá. Các checkpoint và báo cáo được lưu trong thư mục output.

## Đánh giá checkpoint

```powershell
python train_trivago_top5.py --prepared-dir data/prepared --output-dir runs/relative_price_v1 --batch-size 256 --emb-size 16 --num-heads 4 --metadata-projection fieldwise --action-mode event_graph_candidate --price-mode relative_mlp_v1 --eval-only test
```

Lệnh đánh giá xác nhận lượt chạy đã hoàn tất từ `result_latest.pth` rồi nạp `result_best.pth`. Để đánh giá lại kết quả đã báo cáo từ bản lưu cũ, có thể dùng checkpoint best epoch 4 với `run_status.json` hoàn thành và toàn bộ prepared tương ứng của bộ `context_price_v1` trong chế độ không đo thời gian. Với `--track-time`, test time bao gồm toàn bộ evaluation sau khi load checkpoint và chuẩn bị tập test. Inference time chỉ tính forward hiện tại. Việc huấn luyện lại có thể cho số liệu khác do môi trường và tính ngẫu nhiên.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
python prepare_trivago_top5.py --help
python train_trivago_top5.py --help
```

Các kiểm thử bao phủ chuẩn bị dữ liệu, hành động, đồ thị sự kiện, đặc trưng giá, checkpoint và báo cáo metric.

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
