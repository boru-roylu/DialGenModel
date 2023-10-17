import argparse
import os

import glob
import json

import utils


if __name__ == '__main__':
    NFS_DIR = './'
    split = 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, type=str)
    parser.add_argument('--t5_data_dir', required=True, type=str)
    parser.add_argument('--ref_data_dir', required=True, type=str)
    parser.add_argument('--aggregate_frames_state_change', action='store_true')
    parser.add_argument('--target_turn_id', default=None, type=int)
    parser.add_argument('--keep_ref_frames', action='store_true')
    parser.add_argument('--skip_op_if_error', action='store_true')
    parser.add_argument('--clean_ops', action='store_true')
    #parser.add_argument('--non_oracle', action='store_true')
    args = parser.parse_args()

    if args.aggregate_frames_state_change:
        assert not args.clean_ops
    
    if args.clean_ops:
        assert not args.aggregate_frames_state_change

    if args.keep_ref_frames:
        assert args.target_turn_id is not None

    prediction_path = os.path.join(args.exp_dir, 'test_generations.txt')
    if not os.path.exists(prediction_path):
        prediction_path = os.path.join(args.exp_dir, 'generated_predictions.txt')

    if args.aggregate_frames_state_change:
        mwoz_prediction_dir = os.path.join(
            args.exp_dir, 'mwoz_predictions_aggregated_frames')
    else:
        mwoz_prediction_dir = os.path.join(
            args.exp_dir, 'mwoz_predictions')
    os.makedirs(mwoz_prediction_dir, exist_ok=True)

    with open(prediction_path) as f:
        predictions = list(map(lambda x: x.strip(), f.readlines()))

    path = os.path.join(args.t5_data_dir, f'{split}.json')
    with open(path, 'r') as f:
        examples = []
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    
    ref_mwoz_json_paths = glob.glob(
        os.path.join(args.ref_data_dir, f'{split}/*.json'))
    basename_to_dialogue_id_to_dialogue = \
        utils.read_mwoz_jsons(ref_mwoz_json_paths)

    # In state change, we reuse the flag ref_data_dir to get the predicted state 
    # in the previous turn and keep reference frames.
    json_basename_to_dialogue_id_to_dummy_dialog = \
        utils.get_ref_mwoz_format_jsons(
            basename_to_dialogue_id_to_dialogue,
            keep_frames=args.keep_ref_frames)

    post_func = utils.get_mwoz_format_json_from_predictions_non_oracle
    json_basename_to_dialogue_id_to_dialog = \
        utils.get_mwoz_format_json_from_predictions_non_oracle(
            predictions,
            examples,
            json_basename_to_dialogue_id_to_dummy_dialog)

    if args.clean_ops:
        utils.clean_ops_in_frames_for_state_change_to_tlb(
            json_basename_to_dialogue_id_to_dialog)

    if args.aggregate_frames_state_change:
        print('Aggregating frames state change')
        utils.aggregate_frames_state_change(
            json_basename_to_dialogue_id_to_dialog,
            target_turn_id=args.target_turn_id,
            skip_op_if_error=args.skip_op_if_error)
    
    subdir = os.path.join(mwoz_prediction_dir, split)
    os.makedirs(subdir, exist_ok=True)
    for json_basename, dialog_id_to_dialog in \
        json_basename_to_dialogue_id_to_dialog.items():
        json_path = os.path.join(subdir, json_basename)
        with open(json_path, 'w') as f:
            json.dump(list(dialog_id_to_dialog.values()), f, indent=4)
    print('Done')