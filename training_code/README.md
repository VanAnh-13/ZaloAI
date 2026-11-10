# Training code

**LLM**: không fine-tune, không sinh dữ liệu. `Qwen3-14B` dùng nguyên bản, weights trong `/code/models/Qwen3-14B`.

**Mô hình abstain** (`/code/src/datamate/agent/calibration.json`, nếu có): hồi quy logistic khoảng 10 hệ số, fit bằng
`eval/calibrate.py` (copy ở `/code/training_code/eval/`) trên log của một lượt chạy bộ **train** (shop holdout):
```bash
python -m datamate.run --data <train> --out runs/holdout --ids-file eval/splits/holdout_ids.txt
python -m eval.calibrate --logs runs/holdout/logs/logs.jsonl --gold <train>/answers.jsonl
```
Khi nộp bản có `calibration.json`:
- copy `runs/holdout/logs/logs.jsonl` vào `training_code/data/`;
- ghi đường dẫn, SHA-256, số câu vào `data_manifest.json`;
- đặt `"training_performed": true` và mô tả ngắn ở trên.
Không dùng dữ liệu public/private test.

Nếu sau này fine-tune LLM, thư mục này phải có đủ:
- `train.sh`, `configs/`, `src/`, `data_generation/`, `data/`, `logs/`;
- `checkpoints/` (nếu cần);
- `data_manifest.json`: đường dẫn, nguồn, kích thước, SHA-256, số mẫu train/validation.
Xem Docker guide §8.
