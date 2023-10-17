#! /bin/bash -xe

MAIN_DIR=$1
SEED=$2
MODEL_PATH=$3
REF_DST_DIR=$4

if [ $# -ne 4 ]
then
    echo "Usage: $0 <main_dir> <seed> <model_path> <ref_dst_dir>"
    exit 1
fi

MODEL_BASENAME=`basename $MODEL_PATH`
EXP_DIR=$MAIN_DIR/exp_seed_${SEED}_${MODEL_BASENAME}

# 1. Preprocess.
[ -d $MAIN_DIR/t5_data ] || python src/preprocess.py --main_dir $MAIN_DIR --model_path $MODEL_PATH

# 2. Train T5.
bash scripts/train_t5.sh $EXP_DIR $SEED $MODEL_PATH

# 3. Post processing for TLB.
python src/postprocess.py --exp_dir $EXP_DIR

# 4. Post processing for DST.
python src/postprocess.py --aggregate_frames --exp_dir $EXP_DIR

# 5. Evaluating for TLB.
echo "TLB"
python src/eval.py --exp_dir $EXP_DIR
echo

# 6. Evaluating for DST. CB_avg (null), CB_1, CB_2, CB_3. CB_4.
echo "CB_avg"
python src/eval.py --exp_dir $EXP_DIR --ref_dst_main_dir $REF_DST_DIR
echo

echo "CB_1"
python src/eval.py \
    --exp_dir $EXP_DIR --ref_dst_main_dir $REF_DST_DIR --dst_percentile 1
echo

echo "CB_2"
python src/eval.py \
    --exp_dir $EXP_DIR --ref_dst_main_dir $REF_DST_DIR --dst_percentile 2
echo

echo "CB_3"
python src/eval.py \
    --exp_dir $EXP_DIR --ref_dst_main_dir $REF_DST_DIR --dst_percentile 3
echo

echo "CB_4"
python src/eval.py \
    --exp_dir $EXP_DIR --ref_dst_main_dir $REF_DST_DIR --dst_percentile 4
echo