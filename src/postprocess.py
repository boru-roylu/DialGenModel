import argparse
import os

import glob
import json

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--aggregate_frames', action='store_true')
    args = parser.parse_args()

    main_dir = os.path.dirname(args.exp_dir)
    t5_data_dir = os.path.join(main_dir, 't5_data')
    prediction_path = os.path.join(args.exp_dir, 'test_generations.txt')
    if not os.path.exists(prediction_path):
        prediction_path = os.path.join(args.exp_dir, 'generated_predictions.txt')

    if args.aggregate_frames:
        mwoz_prediction_dir = os.path.join(
            args.exp_dir, 'mwoz_predictions_aggregated_frames')
    else:
        mwoz_prediction_dir = os.path.join(
            args.exp_dir, 'mwoz_predictions')
    os.makedirs(mwoz_prediction_dir, exist_ok=True)

    with open(prediction_path) as f:
        predictions = list(map(lambda x: x.strip(), f.readlines()))

    path = os.path.join(t5_data_dir, f'{args.split}.json')
    with open(path, 'r') as f:
        examples = []
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    
    ref_mwoz_json_paths = glob.glob(
        os.path.join(main_dir, f'{args.split}/*.json'))
    basename_to_dialogue_id_to_dialogue = \
        utils.read_mwoz_jsons(ref_mwoz_json_paths)
    json_basename_to_dialogue_id_to_dummy_dialog = \
        utils.get_ref_mwoz_format_jsons(basename_to_dialogue_id_to_dialogue)

    json_basename_to_dialogue_id_to_dialog = \
        utils.get_mwoz_format_json_from_predictions_non_oracle(
            predictions,
            examples,
            json_basename_to_dialogue_id_to_dummy_dialog)

    if args.aggregate_frames:
        print('Aggregating frames')
        utils.aggregate_frames(json_basename_to_dialogue_id_to_dialog)
    
    subdir = os.path.join(mwoz_prediction_dir, args.split)
    os.makedirs(subdir, exist_ok=True)
    for json_basename, dialog_id_to_dialog in \
        json_basename_to_dialogue_id_to_dialog.items():
        json_path = os.path.join(subdir, json_basename)
        with open(json_path, 'w') as f:
            json.dump(list(dialog_id_to_dialog.values()), f, indent=4)
    print('Done')