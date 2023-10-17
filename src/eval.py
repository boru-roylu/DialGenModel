import argparse
import collections
import glob
import json

import os
import numpy as np
import pandas as pd

import eval_utils
import utils

NFS_DIR = './'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--main_dir', type=str, required=True)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--ref_dst_main_dir', default=None, type=str)
    parser.add_argument('--ref_tlb_main_dir', default=None, type=str)
    parser.add_argument('--mwoz_prediction_dir', default=None, type=str)
    parser.add_argument(
        '--dst_percentile',
        default=None, 
        type=int)
   #     choices=[1, 2, 3, 4])
    args = parser.parse_args()

    main_dir = os.path.dirname(args.exp_dir)

    if args.mwoz_prediction_dir is None:
        if args.ref_dst_main_dir is None:
            aggregated_frames = False
            mwoz_prediction_dir = os.path.join(args.exp_dir, 'mwoz_predictions')
        else:
            #assert args.dst_percentile is not None, 'Need to set dst_percentile.'
            aggregated_frames = True
            mwoz_prediction_dir = os.path.join(
                args.exp_dir, 'mwoz_predictions_aggregated_frames')
    else:
        mwoz_prediction_dir = args.mwoz_prediction_dir
        aggregated_frames = 'aggregated_frames' in mwoz_prediction_dir

        if not aggregated_frames:
            assert args.ref_tlb_main_dir is not None

    if args.dst_percentile is None:
        output_path = os.path.join(
            args.exp_dir,
            f'{args.split}_metrics_aggregate_frames_{aggregated_frames}.csv')
    else:
        output_path = os.path.join(
            args.exp_dir,
            f'{args.split}_metrics_aggregate_frames_{aggregated_frames}_q{args.dst_percentile}.csv')

    os.makedirs(mwoz_prediction_dir, exist_ok=True)
    
    if args.ref_dst_main_dir is None and args.ref_tlb_main_dir is None:
        ref_mwoz_format_paths = glob.glob(
            os.path.join(main_dir, f'{args.split}/*.json'))
    elif args.ref_dst_main_dir is not None:
        ref_mwoz_format_paths = glob.glob(
            os.path.join(args.ref_dst_main_dir, f'{args.split}/*.json'))
    elif args.ref_tlb_main_dir is not None:
        ref_mwoz_format_paths = glob.glob(
            os.path.join(args.ref_tlb_main_dir, f'{args.split}/*.json'))
    else:
        raise ValueError('Can not set ref_dst_main_dir and ref_tlb_main_dir at the same time.')

    pred_mwoz_format_paths = glob.glob(
        os.path.join(mwoz_prediction_dir, f'{args.split}/*.json'))

    ref_id_to_dialogs = eval_utils.flat_dict_of_dict(
        utils.read_mwoz_jsons(ref_mwoz_format_paths))
    pred_id_to_dialogs = eval_utils.flat_dict_of_dict(
        utils.read_mwoz_jsons(pred_mwoz_format_paths))
    assert ref_id_to_dialogs.keys() == pred_id_to_dialogs.keys(), \
        f'{ref_id_to_dialogs.keys() = } || {pred_id_to_dialogs.keys() = }'

    all_ref_tuples, all_pred_tuples = eval_utils.get_turn_tuples(
        ref_id_to_dialogs, pred_id_to_dialogs, dst_percentile=args.dst_percentile)
    ref_df = pd.DataFrame.from_dict(all_ref_tuples, orient='columns')
    pred_df = pd.DataFrame.from_dict(all_pred_tuples, orient='columns')
    result = eval_utils.get_result(ref_df, pred_df)

    for name, score in result.items():
        print(f"{name:>30}", f"{score * 100:.2f}")

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)


    # Old metrics.
    """
    turn_level_metrics = eval_utils.get_turn_level_metrics(
        ref_id_to_dialogs, pred_id_to_dialogs)

    df = pd.DataFrame.from_records(turn_level_metrics)
    metrics = df.mean().round(4).to_dict()
    """