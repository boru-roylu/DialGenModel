#! /bin/bash -xe

EXP_DIR=$1
SEED=$2
MODEL_PATH=$3

if [ $# -lt 3 ]
then
    echo "Usage: $0 <exp_dir> <seed> <model_path>"
    exit 1
fi

NUM_CPUS=`nproc`
NUM_GPUS=`nvidia-smi --list-gpus | wc -l`
NUM_WORKERS=`python -c "import math; print(math.floor($NUM_CPUS / $NUM_GPUS))"`

MODEL_BASENAME=`basename $MODEL_PATH`
MAIN_DIR=`dirname $EXP_DIR`
T5_DATA_DIR=$MAIN_DIR/t5_data
OUTPUT_DIR=$EXP_DIR

[ -d $OUTPUT_DIR ] && echo "$OUTPUT_DIR exists!" && exit 1

NUM_TRAIN_EPOCHS=10

# Make sure we have the same effective batch size = 32
# Effective batch size = (per_device_train_batch_size 
#                         * gradient_accumulation_steps * num_gpus)
# We use 2 A-40 for training.
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2

LEARNING_RATE=5e-4

export HF_DATASETS_CACHE=/tmp/$USER/.cache
mkdir -p $HF_DATASETS_CACHE

RANDOM_PORT=$(python src/find_free_port.py)
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.launch \
    --nproc_per_node $NUM_GPUS --master_port $RANDOM_PORT \
    ./src/run_summarization.py \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --do_eval \
    --do_predict \
    --seed $SEED \
    --train_file $T5_DATA_DIR/train.json \
    --validation_file $T5_DATA_DIR/dev.json \
    --test_file $T5_DATA_DIR/test.json \
    --source_prefix "" \
    --output_dir $OUTPUT_DIR \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --predict_with_generate \
    --text_column="dialogue" \
    --summary_column="target" \
    --report_to none \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --metric_for_best_model 'F1' \
    --save_total_limit 1 \
    --dataloader_num_workers $NUM_WORKERS \
    --load_best_model_at_end \
    --generation_max_length 256 \
    --max_target_length 256 \
    ${@:4}