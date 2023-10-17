#!/bin/bash

set -e

EXP_DIR=$1
MODEL_PATH=$2
LAST_K_TURNS_FOR_STATE_CHANGE=$3

if [ $# -ne 3 ]
then
    echo "Usage: bash $0 <exp_dir> <model_path> <last_k_turns_for_state_change>"
    exit 1
fi

TARGET_TURN_ID=1

while :
do
    # For each step, we only predict the state change for the target turn.
    # We enumerate all turns in each dialogue until there is no more turns.

    echo "=================================================="
    echo "Preprocessing state change for turn $TARGET_TURN_ID"
    echo "=================================================="
    python src/preprocess_sc_for_target_turn.py \
        --exp_dir $EXP_DIR \
        --model_path $MODEL_PATH \
        --target_turn_id $TARGET_TURN_ID \
        --last_k_turns_for_state_change $LAST_K_TURNS_FOR_STATE_CHANGE

    TEST_INFER_DIR=$EXP_DIR/test_inference
    TURN_DIR=$TEST_INFER_DIR/turn_$TARGET_TURN_ID
    T5_DIR=$TURN_DIR/t5_data

    NUM_LINES=$(wc -l $T5_DIR/test.json | awk '{print $1}')
    echo "Number of lines in test.json: $NUM_LINES"
    # Stop the while loop until there is no more lines in preprocessed T5 json.
    if [ $NUM_LINES -eq 0 ]
    then
        break
    fi

    echo "=================================================="
    echo "Predicting state change for turn $TARGET_TURN_ID"
    echo "=================================================="
    bash scripts/predict_t5.sh \
        $TURN_DIR/t5_data \
        $TURN_DIR \
        $EXP_DIR \
        --skip_compute_metrics

    echo "=================================================="
    echo "Postprocessing state change for TLB turn $TARGET_TURN_ID"
    echo "=================================================="
    python src/postprocess_sc.py \
        --exp_dir $TURN_DIR \
        --t5_data_dir $TURN_DIR/t5_data \
        --ref_data_dir $TEST_INFER_DIR/mwoz_predictions \
        --keep_ref_frames \
        --clean_ops \
        --target_turn_id $TARGET_TURN_ID

    echo "=================================================="
    echo "Postprocessing state change for DST turn $TARGET_TURN_ID"
    echo "=================================================="
    python src/postprocess_sc.py \
        --exp_dir $TURN_DIR \
        --t5_data_dir $TURN_DIR/t5_data \
        --ref_data_dir $TEST_INFER_DIR/mwoz_predictions_aggregated_frames \
        --aggregate_frames_state_change \
        --keep_ref_frames \
        --skip_op_if_error \
        --target_turn_id $TARGET_TURN_ID
    echo
    echo
    echo
    echo
    echo
    echo

    TARGET_TURN_ID=$((TARGET_TURN_ID+2))
done

echo "=================================================="
echo "Finish"
echo "=================================================="
