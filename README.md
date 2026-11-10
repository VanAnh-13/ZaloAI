# DataMate agent — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ. Kế hoạch tổng thể ở [PLAN.md](PLAN.md);
README nộp kèm image ở [docker/README.md](docker/README.md).

## Kiến trúc

Các tầng chỉ phụ thuộc theo một chiều (tầng dưới không biết tầng trên):

```
core        chuỗi tiếng Việt, làm tròn nửa lên, kỳ thời gian                 (thuần, không phụ thuộc gì)
nlu         hiểu câu hỏi: kỳ, kiểu đáp án, tên được nhắc, viết lại teencode   → core
ingest      file csv/xlsx/sqlite → bảng có kiểu (Business)                    → core
semantics   vai trò cột, trạng thái đơn, luật §8, mô hình bán hàng S, xếp hạng → core, ingest
workspace   ghép ingest + semantics, schema card, cache pickle, chạy song song → ingest, semantics
dm          façade cho code do model viết (re-export + tài liệu)              → core, semantics
agent       client LLM, prompt, gợi ý, sandbox, vòng code-act, kiểm tra, bỏ phiếu, độ tin cậy
pipeline    composition root (chuẩn bị workspace, chạy song song, ghi kết quả); run.py = CLI
```

Luồng một câu hỏi: `pipeline` → `agent.solver.solve`:
1. `agent.hints.analyze`: kỳ, kiểu đáp án, tên trong dữ liệu, cảnh báo kỳ nằm ngoài dữ liệu.
2. K lượt `agent.loop.run_attempt`: model viết Python → `agent.sandbox` chạy trong process riêng (có `S`, `T`, `dm`…)
   → … → code gọi `answer()` → `agent.answer.normalize` → `agent.verify.check` (nghi ngờ thì trả lại model kiểm tra một lần).
3. `vote` → `agent.confidence.Calibrator` → nộp hoặc abstain → `results.ResultWriter` ghi và fsync ngay.

```
datamate/
  config.py  pipeline.py  run.py  results.py
  core/        text.py numbers.py periods.py
  nlu/         periods.py answer_type.py entities.py normalize.py
  ingest/      values.py readme.py loaders.py columns.py model.py build.py
  semantics/   roles.py status.py order_rules.py sales.py ranking.py
  workspace/   build.py card.py cache.py batch.py __main__.py
  dm/          __init__.py (façade + DOC cho prompt)
  agent/       llm.py prompts.py hints.py sandbox.py sandbox_child.py answer.py verify.py loop.py
               confidence.py solver.py
eval/          scorer.py (luật chấm)  oracle.py (kiểm S bằng đáp án train)  splits.py  calibrate.py
docker/        Dockerfile  predict.sh  README.md (nộp kèm image)
training_code/ README.md  data_manifest.json (hiện không fine-tune)
tests/         pytest
```

## Lệnh hay dùng
Máy dev là Windows nên đặt `PYTHONUTF8=1` để in được tiếng Việt. `TRAIN=C:/Users/Admin/Downloads/train/train`.

```bash
uv sync                                                    # môi trường (.venv)
uv run pytest -q                                           # test, gồm e2e bằng mock LLM

# dựng workspace (bảng sạch + mô hình S + schema card) của mọi shop, xem cảnh báo và card
uv run python -m datamate.workspace --data $TRAIN --cards runs/cards

# kiểm tra mô hình S bằng đáp án train, không cần LLM (hiện: 841/841 câu khớp)
uv run python -m eval.oracle --train $TRAIN

# chạy thử pipeline không cần GPU (mock LLM chỉ xem bảng rồi abstain)
uv run python -m datamate.run --data $TRAIN --out runs/mock --limit 20 --mock
```

### Vòng phát triển với LLM thật
```bash
export DATAMATE_LLM_BASE_URL=http://<host>:8000/v1 DATAMATE_LLM_MODEL=qwen3-14b
uv run python -m eval.splits --train $TRAIN                # eval/splits/*.txt (đã có sẵn trong repo)

# 1. bộ hồi quy 300 câu: đo PA sau mỗi thay đổi
uv run python -m datamate.run --data $TRAIN --out runs/r1 --ids-file eval/splits/regression_300.txt
uv run python -m eval.scorer --pred runs/r1/submission.jsonl --gold $TRAIN/answers.jsonl \
    --questions $TRAIN/questions.jsonl --ids-file eval/splits/regression_300.txt

# 2. hiệu chỉnh ngưỡng abstain trên shop holdout (chưa dùng khi viết prompt), nên chạy với nhiều lượt bỏ phiếu
DATAMATE_ATTEMPTS=3 uv run python -m datamate.run --data $TRAIN --out runs/holdout --ids-file eval/splits/holdout_ids.txt
uv run python -m eval.calibrate --logs runs/holdout/logs/logs.jsonl --gold $TRAIN/answers.jsonl
# -> ghi datamate/agent/calibration.json, được nạp tự động lúc chạy (xoá file = trả lời mọi câu đã bỏ phiếu)
```

Biến môi trường dev (giá trị mặc định là giá trị dùng khi nộp): `DATAMATE_ATTEMPTS` (1), `DATAMATE_CONCURRENCY` (16),
`DATAMATE_THINKING` (1), `DATAMATE_MAX_TOKENS` (6144), `DATAMATE_TEMPLATE_KWARGS` (đặt 0 nếu host không nhận
`chat_template_kwargs`), `DATAMATE_WS` (thư mục cache workspace), `DATAMATE_DEADLINE_S` (19800).

## Docker (máy Linux có GPU)
```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/public_test:/data:ro -v /path/to/output:/result \
  zaic2026_datamate_<team>:final /code/predict.sh
```
Trước khi nộp:
- Chạy 2 lần, so PA (phải lệch ≤ 0.03).
- Thời gian ước tính cho 500 câu phải dưới 6 giờ.
- Điền revision của weights vào `docker/README.md` và `training_code/data_manifest.json`.
- Nếu đã fit, commit `datamate/agent/calibration.json`.
