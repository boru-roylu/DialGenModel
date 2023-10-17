#! /bin/bash

if [ $# -lt 3 ]
then
    echo "Usage: $0 <t5_data_dir> <output_dir> <model_path>"
    exit 1
fi

T5_DATA_DIR=$1
OUTPUT_DIR=$2
MODEL_PATH=$3

NUM_GPUS=`nvidia-smi --list-gpus | wc -l`

export HF_DATASETS_CACHE=/tmp/$USER/.cache
mkdir -p $HF_DATASETS_CACHE

RANDOM_PORT=$(python src/find_free_port.py)
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.launch --nproc_per_node $NUM_GPUS --master_port $RANDOM_PORT ./src/run_summarization.py \
    --model_name_or_path $MODEL_PATH \
    --do_predict \
    --validation_file $T5_DATA_DIR/test.json \
    --test_file $T5_DATA_DIR/test.json \
    --source_prefix "" \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size=16 \
    --predict_with_generate \
    --text_column="dialogue" \
    --summary_column="target" \
    --report_to none \
    --generation_max_length 256 \
    --max_target_length 256 \
    ${@:4}