# DataMate agent — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ. Kế hoạch ban đầu ở [PLAN.md](PLAN.md);
README nộp kèm image ở [docker/README.md](docker/README.md).

## Luồng trả lời một câu hỏi: cascade ba tầng bỏ phiếu

Điểm là +1 đúng / −1 sai / 0 bỏ qua, nên một đáp án chỉ đáng nộp khi nhiều khả năng đúng. Tín hiệu mạnh nhất
là **hai cách hiểu câu hỏi độc lập cho ra cùng một giá trị**. Vì vậy mỗi tầng dưới đây bỏ một phiếu, tầng rẻ
chạy trước, tầng đắt chỉ chạy khi còn thiếu phiếu:

```
câu hỏi ──► phân tích tất định (kỳ, kiểu đáp án, tên trong dữ liệu, kỳ ngoài dữ liệu)        ~ms, không LLM
   │
   ├─ Tầng 1  TEMPLATE   regex nhận dạng câu "plain" ─┐                                      0 lần gọi LLM
   │                                                  ├─► ENGINE truy vấn tất định trên S ─► phiếu
   ├─ Tầng 2  PLANNER    LLM điền 1 QUERY_SPEC (JSON) ┘   (đơn hợp lệ, kỳ, làm tròn,          1 lần gọi ngắn
   │                     hoặc chuyển "code"               xếp hạng, định dạng đáp án)         (~150 token)
   │
   └─ Tầng 3  CODE AGENT LLM viết Python trong phiên sandbox kiểu notebook ─► answer() ─► phiếu
                         chạy thêm lượt (tối đa DATAMATE_ATTEMPTS) chỉ khi chưa đủ đồng thuận  nhiều lần gọi dài

quyết định:  effort "balanced" → dừng ngay khi 2 phiếu trùng nhau; hết tầng mà không trùng → abstain
             effort "fast"     → phiếu đầu tiên thắng (chạy dev nhanh, hoặc khi run bị trễ giờ)
             có calibration.json → P(đúng) từ đặc trưng đồng thuận thay cho luật "2 phiếu"
```

- **Tầng 1** (`agent/templates.py`): câu hỏi mà sau khi bỏ từ chỉ kỳ và từ đệm không còn gì (không lọc theo
  mặt hàng, khách, nhóm…) được ánh xạ thẳng thành truy vấn: doanh thu, số đơn, số đơn huỷ, tỉ lệ huỷ, giá trị
  đơn trung bình, ngày doanh thu cao nhất, % thay đổi doanh thu. Đây chính là logic mà `eval/oracle.py` kiểm
  bằng đáp án train, nay dùng luôn lúc chạy.
- **Tầng 2** (`agent/planner.py`): model chỉ chọn *ý nghĩa* (metric, kỳ, bộ lọc, xếp hạng theo gì) và luôn có
  quyền trả `"route": "code"`. Mọi thứ nó viết đều được kiểm: tên không có trong dữ liệu, kiểu đáp án lệch
  với câu hỏi, tie ở biên top-k, kỳ vô lý → tầng này không bỏ phiếu.
- **Engine** (`agent/query.py`): `value | change | top | best_day` trên `S`, có bộ lọc; kỳ nằm hoàn toàn ngoài
  dữ liệu → `unanswerable` (§7). Cái gì engine không diễn đạt được thì ném `QueryError` và câu hỏi đi tiếp.
- **Tầng 3** (`agent/loop.py`): code-act như cũ nhưng chạy trong phiên sandbox bền vững, có ví dụ mẫu truy
  xuất theo độ giống câu hỏi (nếu đã thu), và chỉ chạy lượt 2, 3 cho những câu thật sự chưa chắc.

Hệ quả: câu plain không tốn LLM; phần lớn câu bán hàng tốn một lần gọi ngắn cộng một lượt code; thời gian của
model dồn vào đúng những câu khó.

### Sandbox (`agent/sandbox.py`, `agent/sandbox_child.py`)
Một tiến trình worker cho **mỗi lượt giải** (trước đây: một tiến trình cho mỗi khối code).
- Workspace của shop nạp một lần; **biến giữ nguyên giữa các bước** như Jupyter; giá trị dòng cuối tự in ra.
- Worker khởi động nền ngay khi mở phiên, tức trong lúc model còn đang viết khối code đầu → không cộng thời
  gian chờ. Đo trên shop tổng hợp 12k đơn (Linux): ~390 ms/bước trước đây → ~12 ms/bước.
- Lỗi trả về kèm gợi ý: cột gần giống khi `KeyError`, biến đã tạo khi `NameError`, `S` là None…
- Rào chắn (audit hook): code chỉ đọc file của shop đang hỏi (không đọc được `answers.jsonl`, shop khác,
  `/result`, cache workspace), chỉ ghi trong thư mục tạm của phiên, không subprocess, không mạng.
- Giới hạn bộ nhớ / CPU / kích thước file (POSIX); quá giờ thì kill cả nhóm tiến trình, phiên tự khởi động
  lại ở bước sau và model được báo là biến đã mất.

## Kiến trúc

Các tầng chỉ phụ thuộc theo một chiều (tầng dưới không biết tầng trên):

```
core        chuỗi tiếng Việt, làm tròn nửa lên, kỳ thời gian                 (thuần, không phụ thuộc gì)
nlu         hiểu câu hỏi: kỳ, kiểu đáp án, tên được nhắc, viết lại teencode   → core
ingest      file csv/xlsx/sqlite → bảng có kiểu (Business)                    → core
semantics   vai trò cột, trạng thái đơn, luật §8, mô hình bán hàng S, xếp hạng → core, ingest
workspace   ghép ingest + semantics, schema card, cache pickle, chạy song song → ingest, semantics
dm          façade cho code do model viết (re-export + tài liệu)              → core, semantics
agent       templates, planner, query engine, code agent, sandbox, kiểm tra, cascade, độ tin cậy
pipeline    composition root (chuẩn bị workspace, chạy song song, ghi kết quả); run.py = CLI
```

```
datamate/
  config.py  pipeline.py  run.py  results.py
  core/        text.py numbers.py periods.py
  nlu/         periods.py answer_type.py entities.py normalize.py
  ingest/      values.py readme.py loaders.py columns.py model.py build.py
  semantics/   roles.py status.py order_rules.py sales.py (S: where / rank / qty / find) ranking.py
  workspace/   build.py card.py cache.py batch.py __main__.py
  dm/          __init__.py (façade + DOC cho prompt)
  agent/       solver.py (cascade)  templates.py  planner.py  query.py  loop.py  prompts.py  examples.py
               sandbox.py sandbox_child.py  llm.py  hints.py  answer.py  verify.py  confidence.py
eval/          scorer.py  oracle.py (tầng 1 vs đáp án train)  report.py (lỗi + thời gian theo tầng)
               harvest.py (ví dụ mẫu)  calibrate.py  splits.py
docker/        Dockerfile  predict.sh  README.md (nộp kèm image)
training_code/ README.md  data_manifest.json
tests/         pytest (không cần GPU, không cần dữ liệu thật)
```

## Lệnh hay dùng
Máy dev là Windows nên đặt `PYTHONUTF8=1` để in được tiếng Việt. `TRAIN=C:/Users/Admin/Downloads/train/train`.

```bash
uv sync                                                    # môi trường (.venv)
uv run pytest -q                                           # test, gồm cascade end-to-end bằng model giả lập

# dựng workspace (bảng sạch + mô hình S + schema card) của mọi shop, xem cảnh báo và card
uv run python -m datamate.workspace --data $TRAIN --cards runs/cards

# tầng 1 so với đáp án train, không cần LLM: độ phủ và độ chính xác của các câu trả lời 0 token
uv run python -m eval.oracle --train $TRAIN
```

### Vòng phát triển với LLM thật
```bash
export DATAMATE_LLM_BASE_URL=http://<host>:8000/v1 DATAMATE_LLM_MODEL=qwen3-14b
export DATAMATE_LLM_CACHE=runs/llm_cache      # chỉ dev: request giống hệt không gọi lại model

# 1. lặp nhanh: phiếu đầu tiên thắng, một lượt code — để soi lỗi, không phải để đo điểm nộp
DATAMATE_EFFORT=fast uv run python -m datamate.run --data $TRAIN --out runs/f1 --ids-file eval/splits/regression_300.txt

# 2. đo như lúc nộp (balanced), rồi xem lỗi và thời gian theo từng tầng
uv run python -m datamate.run --data $TRAIN --out runs/r1 --ids-file eval/splits/regression_300.txt
uv run python -m eval.report --logs runs/r1/logs/logs.jsonl --gold $TRAIN/answers.jsonl --failures runs/r1/failures.jsonl

# mốc so sánh = flow cũ (chỉ code agent, một lượt, trả lời hết)
DATAMATE_TEMPLATES=0 DATAMATE_PLANNER=0 DATAMATE_EFFORT=fast uv run python -m datamate.run --data $TRAIN --out runs/base ...
```

`eval.report` in ra: PA và luật abstain đang thêm hay làm mất bao nhiêu PA; **độ chính xác của từng người bỏ
phiếu** (template / plan / code lượt 1, 2, 3); độ chính xác theo kiểu đồng thuận (`2 phiếu [code+plan]`,
`1 phiếu [code] · có phiếu khác`…); theo type và shop; số lần gọi, token và giây LLM của từng tầng; câu sai
gom theo loại lỗi kèm phiếu và code cuối. `runs/*/run_summary.json` có tổng hợp thời gian của cả run.

### Ví dụ mẫu (few-shot) và hiệu chỉnh abstain — làm sau khi đã có một run tốt trên train
```bash
# chương trình của các câu ĐÚNG được chạy lại trong sandbox; chỉ giữ cái tự tái lập được đáp án
uv run python -m eval.harvest --train $TRAIN --logs runs/r1/logs/logs.jsonl      # → datamate/agent/examples.jsonl

# đo lại trên train mà không tự xem đáp án của mình: không lấy ví dụ từ chính shop đang hỏi
DATAMATE_EXAMPLES_HOLDOUT=1 uv run python -m datamate.run --data $TRAIN --out runs/r2 --ids-file eval/splits/holdout_ids.txt

# P(đúng) từ đặc trưng đồng thuận, fit trên shop holdout → datamate/agent/calibration.json (tự nạp khi chạy)
uv run python -m eval.calibrate --logs runs/r2/logs/logs.jsonl --gold $TRAIN/answers.jsonl
```

### Biến môi trường (giá trị mặc định là giá trị dùng khi nộp)
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DATAMATE_EFFORT` | `balanced` | `balanced`: cần 2 phiếu trùng; `fast`: phiếu đầu tiên thắng |
| `DATAMATE_ATTEMPTS` | `3` | số lượt code agent tối đa cho một câu (chỉ chạy khi còn thiếu phiếu; `balanced` luôn cho ít nhất 2) |
| `DATAMATE_TEMPLATES` / `DATAMATE_PLANNER` | `1` / `1` | bật tắt tầng 1 / tầng 2 |
| `DATAMATE_PLAN_THINKING` | `0` | cho planner suy nghĩ (chậm hơn; spec đã có trường `analysis`) |
| `DATAMATE_THINKING` | `1` | code agent: `1` mọi bước, `0` tắt, `auto` = bước đầu + bước sau lỗi/challenge |
| `DATAMATE_FEWSHOT` | `3` | số ví dụ mẫu đưa vào prompt (khi có `examples.jsonl`) |
| `DATAMATE_EXAMPLES_HOLDOUT` | `0` | chỉ dev: không lấy ví dụ từ shop đang hỏi |
| `DATAMATE_LLM_CACHE` | (tắt) | chỉ dev: thư mục cache câu trả lời của model |
| `DATAMATE_SANDBOX_GUARD` / `DATAMATE_SANDBOX_MEM_MB` | `1` / `8192` | rào chắn và trần bộ nhớ của sandbox |
| `DATAMATE_CONCURRENCY` | `16` | số câu chạy song song |
| `DATAMATE_MAX_TOKENS`, `DATAMATE_TEMPLATE_KWARGS`, `DATAMATE_WS`, `DATAMATE_DEADLINE_S` | như cũ | |

Khi run bị trễ so với hạn 6 giờ, các câu còn lại tự chuyển sang `fast`. Nếu server model không lên, tầng 1 vẫn
trả lời các câu plain thay vì abstain toàn bộ.

## Docker (máy Linux có GPU)
```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/public_test:/data:ro -v /path/to/output:/result \
  zaic2026_datamate_<team>:final /code/predict.sh
```
Trước khi nộp:
- `eval.oracle` trên train: mọi dòng của tầng 1 phải 100 %.
- Chạy 2 lần, so PA (phải lệch ≤ 0.03).
- Thời gian ước tính cho 500 câu phải dưới 6 giờ (xem `run_summary.json`).
- Điền revision của weights vào `docker/README.md` và `training_code/data_manifest.json`.
- Nếu đã thu ví dụ / đã fit: commit `datamate/agent/examples.jsonl`, `datamate/agent/calibration.json` và ghi
  vào `training_code/data_manifest.json` (xem `training_code/README.md`).
