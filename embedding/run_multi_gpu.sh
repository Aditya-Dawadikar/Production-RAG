#!/bin/bash
set -euo pipefail

BUCKET="prod-rag-bucket"
INPUT_PREFIX="wiki-chunks/"
OUTPUT_PREFIX="wiki-embeddings/"

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=512

NUM_WORKERS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_WORKERS" -eq 0 ]; then
  echo "No CUDA GPUs visible. Exiting."
  exit 1
fi

echo "Detected $NUM_WORKERS CUDA GPU(s)"
echo "Starting $NUM_WORKERS worker(s)..."

for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
  echo "Launching worker $WORKER_ID on GPU $WORKER_ID"

  (
    echo "[$(date)] Worker $WORKER_ID started"

    CUDA_VISIBLE_DEVICES=$WORKER_ID \
    python embed_worker.py \
      --bucket "$BUCKET" \
      --input-prefix "$INPUT_PREFIX" \
      --output-prefix "$OUTPUT_PREFIX" \
      --worker-id "$WORKER_ID" \
      --num-workers "$NUM_WORKERS" \
      --model-name "$MODEL_NAME" \
      --batch-size "$BATCH_SIZE"

    echo "[$(date)] Worker $WORKER_ID finished"
  ) 2>&1 | tee "worker_${WORKER_ID}.log" &
done

wait

echo "All GPU workers completed."