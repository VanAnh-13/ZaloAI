#!/usr/bin/env bash
# DataMate entry point, run by the organizers without arguments:
#   /data (read-only)  ->  /result/submission.jsonl + /result/logs/logs.jsonl
# Every setting is fixed here on purpose (Docker guide §5: organizers never change them).
set -uo pipefail
export DATAMATE_T0="$(date +%s)"  # the 6 h budget includes model loading

DATA_DIR=/data
OUT_DIR=/result
MODEL_DIR=/code/models/Qwen3-14B
SERVED_NAME=qwen3-14b
PORT=8000
WS_DIR=/tmp/datamate_ws

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/system"
export PYTHONPATH=/code/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 PYTHONUTF8=1
export DATAMATE_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" DATAMATE_LLM_MODEL="$SERVED_NAME" DATAMATE_WS="$WS_DIR"
# answering flow: template -> planner -> code agent; an answer needs two agreeing votes, at most 3 code attempts
export DATAMATE_EFFORT=balanced DATAMATE_ATTEMPTS=3 DATAMATE_TEMPLATES=1 DATAMATE_PLANNER=1 DATAMATE_THINKING=1
unset DATAMATE_LLM_CACHE DATAMATE_EXAMPLES_HOLDOUT  # development-only switches never apply here

echo "[predict.sh] start $(date -Is)"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --seed 0 \
    --enable-prefix-caching --reasoning-parser qwen3 \
    > "$OUT_DIR/system/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null' EXIT

# CPU work while the model loads: typed tables + sales model + schema card of every shop, cached
python3 -m datamate.workspace --data "$DATA_DIR" --ws "$WS_DIR" > "$OUT_DIR/system/workspace.log" 2>&1

# run.py waits for the server (GET /v1/models); if it never comes up, questions abstain
python3 -m datamate.run --data "$DATA_DIR" --out "$OUT_DIR" --ws "$WS_DIR" 2>&1 | tee "$OUT_DIR/system/run.log"
echo "[predict.sh] done $(date -Is)"
