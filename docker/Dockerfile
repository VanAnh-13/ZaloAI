# Build from the repo root (model weights must be in ./models/Qwen3-14B):
#   docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
FROM vllm/vllm-openai:v0.10.1.1

# plain `docker run` must start predict.sh, not the vLLM server
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 \
    PYTHONHASHSEED=0 PYTHONUTF8=1 PYTHONPATH=/code/src

WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
# organizers' fix for the two-GPU NCCL initialization hang (Docker guide §9)
RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7

# weights first: biggest layer, stays cached when only code changes
COPY models/ /code/models/
COPY training_code/ /code/training_code/
# evaluation + calibration code: how datamate/agent/calibration.json is fitted (Docker guide §8)
COPY eval/ /code/training_code/eval/
COPY docker/predict.sh /code/predict.sh
COPY docker/README.md /code/README.md
COPY datamate/ /code/src/datamate/
RUN chmod +x /code/predict.sh

CMD ["bash", "/code/predict.sh"]
