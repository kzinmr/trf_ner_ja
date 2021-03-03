# hub model
export BERT_MODEL=bert-base-uncased  # cl-tohoku/bert-base-japanese
# local model
export PRETRAINED_DIR=${WORK_DIR}/electra_small_wiki40b_ja_mecab_ipadic
export PRETRAINED_MODEL=${PRETRAINED_DIR}/model_discriminator.pt
export PRETRAINED_CONFIG=${PRETRAINED_DIR}/config.json
export PRETRAINED_TOKENIZER=${PRETRAINED_DIR}/tokenizer.json

export LABEL_PATH=${WORK_DIR}/label_types.txt

export DATA_DIR=${WORK_DIR}/conll03/
export OUTPUT_DIR=${WORK_DIR}/outputs/

# export CACHE=${WORK_DIR}/cache/
export SEED=42
mkdir -p $OUTPUT_DIR
# In Docker, the following error occurs due to not big enough memory:
# `RuntimeError: DataLoader worker is killed by signal: Killed.`
# Try to reduce NUM_WORKERS or MAX_LENGTH or BATCH_SIZE or increase docker memory
export NUM_WORKERS=8
export GPUS=1

export MAX_LENGTH=128
export BATCH_SIZE=16
# BATCH_SIZE * ACM_GRAD_BATCH is actual batch size
export ACM_GRAD_BATCH=1
export LEARNING_RATE=0.001
export WEIGHT_DECAY=0.01
export PATIENCE=3
export ANNEAL_FACTOR=0.5
export DROPOUT=0.1

export NUM_EPOCHS=10
export NUM_SAMPLES=20000

# --model_name_or_path=$PRETRAINED_MODEL \
# --config_path=$PRETRAINED_CONFIG \
# --tokenizer_path=$PRETRAINED_TOKENIZER \

python3 pl_main.py \
--model_name_or_path=$BERT_MODEL \
--labels=$LABEL_PATH \
--seed=$SEED \
--data_dir=$DATA_DIR \
--output_dir=$OUTPUT_DIR \
--cache_dir=/app/workspace/cache \
--max_epochs=$NUM_EPOCHS \
--num_workers=$NUM_WORKERS \
--tokens_per_batch=$MAX_LENGTH \
--window_stride=-1 \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--accumulate_grad_batches=$ACM_GRAD_BATCH \
--learning_rate=$LEARNING_RATE \
--patience=$PATIENCE \
--anneal_factor=$ANNEAL_FACTOR \
--adam_epsilon=1e-8 \
--weight_decay=$WEIGHT_DECAY \
--dropout=$DROPOUT \
--monitor="f1" \
--monitor_training \
--freeze_pretrained \
--num_samples=$NUM_SAMPLES \
--do_train \
--do_predict #\
# --gpus=$GPUS