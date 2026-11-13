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

**Ví dụ mẫu cho agent viết code** (`/code/src/datamate/agent/examples.jsonl`, nếu có): câu hỏi của bộ **train**
kèm chương trình Python mà chính model đã viết và cho ra đúng đáp án tham chiếu. `eval/harvest.py` chạy lại từng
chương trình trong sandbox và chỉ giữ cái tự tái lập được đáp án:
```bash
python -m datamate.run --data <train> --out runs/train
python -m eval.harvest --train <train> --logs runs/train/logs/logs.jsonl
```
Lúc suy luận, vài ví dụ giống câu hỏi nhất (TF-IDF trên từ, 0 tham số) được đưa vào prompt. Đây là dữ liệu sinh
bằng model nên khi nộp bản có file này:
- copy `runs/train/logs/logs.jsonl` vào `training_code/data/`;
- ghi đường dẫn, SHA-256 (lệnh harvest in ra), số ví dụ vào `data_manifest.json`;
- đặt `"data_generation_performed": true`.
File không chứa câu hỏi hay đáp án của public/private test, và không có logic nào tra theo business id / question id.

Nếu sau này fine-tune LLM, thư mục này phải có đủ:
- `train.sh`, `configs/`, `src/`, `data_generation/`, `data/`, `logs/`;
- `checkpoints/` (nếu cần);
- `data_manifest.json`: đường dẫn, nguồn, kích thước, SHA-256, số mẫu train/validation.
Xem Docker guide §8.
