#! /bin/bash -xe

set -e

MAIN_DIR=$1
SEED=$2
MODEL_PATH=$3
REF_TLB_DIR=$4
REF_DST_DIR=$5
# TURN = Last k turns for creating previous state.
TURN=$6

if [ $# -ne 6 ]
then
    echo "Usage: $0 <main_dir> <seed> <model_path> <ref_tlb_dir> <ref_dst_dir> <turn>"
    exit 1
fi

MODEL_BASENAME=`basename $MODEL_PATH`
EXP_DIR=$MAIN_DIR/exp_seed_${SEED}_${MODEL_BASENAME}

# 1. Preprocess.
python src/preprocess_sc.py --main_dir $MAIN_DIR --model_path $MODEL_PATH

# 2. Train T5.
bash scripts/train_t5.sh $EXP_DIR $SEED $MODEL_PATH --is_state_change

# 3. Run state-change inference with predicted results as the previous state.
# Different from T5 model, T5-SC requires turn-by-turn prediction because we 
# need to dynamically update the previous state.
# In `run_sc_inference.sh`, we predict the state change of each turn and use
# postprocessing to update the previous state.
# In the next turn, we run preprocessing script to create the t5-format data for
# the next turn prediction. We iterate this process until there is no more turns.
bash scripts/run_sc_inference.sh $EXP_DIR $MODEL_PATH $TURN

# 4. Evaluating for TLB.
MWOZ_PREDICTION_DIR=$EXP_DIR/test_inference/mwoz_predictions
echo "TLB"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_tlb_main_dir $REF_TLB_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR
echo

# 5. Evaluating for DST
MWOZ_PREDICTION_DIR=$EXP_DIR/test_inference/mwoz_predictions_aggregated_frames
echo "CB_avg"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_dst_main_dir $REF_DST_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR
echo

echo "CB_1"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_dst_main_dir $REF_DST_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR \
    --dst_percentile 1
echo

echo "CB_2"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_dst_main_dir $REF_DST_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR \
    --dst_percentile 2
echo

echo "CB_3"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_dst_main_dir $REF_DST_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR \
    --dst_percentile 3
echo

echo "CB_4"
python src/eval.py \
    --exp_dir $EXP_DIR \
    --ref_dst_main_dir $REF_DST_DIR \
    --mwoz_prediction_dir $MWOZ_PREDICTION_DIR \
    --dst_percentile 4
echo