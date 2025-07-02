#!/bin/bash

rm -f train_small.log

# Start GPU logging in background and record its PID
./log_gpu.sh > gpu_log.txt 2>&1 &
LOG_PID=$!

# Run training
./train_gpt3_small_custom.sh \
  data/checkpoints \
  data/tensorboard_logs \
  data/gpt2-vocab.json \
  data/gpt2-merges.txt \
  my-gpt2_text_document >> train_small.log 2>&1

# After training finishes, kill the logging process
kill $LOG_PID

#!/bin/bash

# Create next result_x folder
RESULT_BASE="data/Megatron-LM_result"
NEXT_INDEX=$(find "$RESULT_BASE" -maxdepth 1 -type d -name "result_*" | sed 's/.*result_//' | sort -n | tail -n 1)
NEXT_INDEX=$((NEXT_INDEX + 1))
RESULT_DIR="${RESULT_BASE}/result_${NEXT_INDEX}"
mkdir -p "$RESULT_DIR"

# Copy files
cp gpu_log.csv train_gpt3_small_custom.sh train_small.log "$RESULT_DIR"

