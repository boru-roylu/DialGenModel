import argparse
import collections
import copy
import itertools
import os

import transformers as tfs
import glob
import json

import utils
import config


def convert_active_referent_slot_values_to_possible_comb(
    active_referent_slot_values,
    use_value_permutation,
    use_referent_permutation):
        
    # comb: r1 [rv] v1 [v] v2.
    referent_to_combs = collections.defaultdict(list)
    for referent_values in active_referent_slot_values:
        # v1, v2
        referent = referent_values['referent']
        values = referent_values['values']

        # referent_string: r1.
        referent_string = referent

        # v1, v2 -> (v1, v2), (v2, v1)
        # values_comb: (v1, v2)
        # values_combs: [(v1, v2), (v2, v1)]
        # values_string: v1 [v] v2.
        if use_value_permutation:
            values_combs = list(itertools.permutations(values))
        else:
            values_combs = [values]
        for values_comb in values_combs:
            values_string = config.value_sep.join(values_comb)
            comb = config.referent_value_sep.join([values_string, referent_string])
            referent_to_combs[referent].append(comb)

    # referent_to_combs: 
    # r1: [(r1, v1, v2), (r1, v2, v1)]
    # r2: [(r2, v3, v4, v5), (r2, v3, v5, v4) ...]
    # After product: list_of_referent_values: [(r1, v1, v2), (r2, v3, v4, v5)]
    cross_referent_values_comb_strings = []
    for list_of_referent_values in list(itertools.product(*referent_to_combs.values())):
        if use_referent_permutation:
            # list_of_referent_values: [(r1, v1, v2), (r2, v3, v4, v5)]
            # After permutations: 
            # [(r1, v1, v2), (r2, v3, v4, v5)] or 
            # [(r2, v3, v4, v5), (r1, v1, v2)]
            cross_referent_values_combs = list(itertools.permutations(list_of_referent_values))
        else:
            cross_referent_values_combs = [list_of_referent_values]

        for cross_referent_values_comb in cross_referent_values_combs:
            cross_referent_values_comb_strings.append(
                config.cross_referent_sep.join(cross_referent_values_comb))
    return cross_referent_values_comb_strings


def preprocess(
    tokenizer,
    pred_dialog,
    ref_dialog,
    target_turn_idx,
    add_slot,
    add_slot_description,
    add_possible_values,
    use_slot_permutation,
    use_value_permutation,
    use_referent_permutation,
    max_num_tokens_for_dialog,
    max_num_tokens_for_prompt,
    max_num_tokens_for_prev_state,
    last_k_turns=None,
    last_k_turns_for_state_change=None,
    excluded_domains=None,
    is_train=False):

    def truncate(text, max_num_tokens):
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_num_tokens,
            truncation=True)['input_ids']
        return tokenizer.decode(input_ids)

    if last_k_turns_for_state_change is not None:
        """
        # Due to the T5 input limit, we can only put limited tokens for the state.
        # Thus, we use the most recent k turns to create a approximate state.
        # To create the state of the most recent k turns, we need to compare 
        # the state in the previous turn (turn_idx - 2) and the previous k 
        # turn before turn_idx - 2 (turn_idx - 2 - k).
        #
        # Eg.
        # prev_prev_turn_idx = turn_idx - 2 - k
        # prev_turn_idx = turn_idx - 2
        # curr_turn_idx = turn_idx <--- this is the turn we want to predict.
        #
        # We compute the state difference between `prev_turn_idx` and 
        # `prev_prev_turn_idx` to create the approximate state.
        """
        last_k_turns_for_state_change += 2

    curr_turns = []
    #prev_frames = []
    for turn_idx, turn in enumerate(ref_dialog['turns']):
        turn_id = turn['turn_id']
        turn_text = f'[{turn["speaker"]}] {turn["utterance"]}'
        curr_turns.append(turn_text)

        if turn["speaker"] != "USER":
            assert turn_idx != target_turn_idx
            continue

        # We only generate data for target_turn_idx.
        if turn_idx < target_turn_idx:
            continue

        if turn_idx > target_turn_idx:
            break 

        domain_slot_name_to_active_referent_slot_values = \
            collections.defaultdict(list)

        for frame in turn['frames_state_change']:

            if not frame['state']['slot_values']:
                continue

            domain_name = frame['service']
            #active_domains.add(domain_name)
            for referent_slot, values in frame["state"]["slot_values"].items():
                referent, slot_name = referent_slot.split("-")

                domain_slot_name = f'{domain_name}-{slot_name}'

                domain_slot_name_to_active_referent_slot_values[domain_slot_name].append({
                    'referent': referent,
                    'values': values,
                })
        
        domain_slot_name_to_prev_active_referent_slot_values = \
            collections.defaultdict(list)

        # Gets predictions from the previous predicted turn.
        if turn_idx - 2 >= 0:
            prev_pred_turn = pred_dialog['turns'][turn_idx - 2]
            prev_frames = prev_pred_turn['frames']
        else:
            prev_frames = []

        if last_k_turns_for_state_change is not None:
            prev_prev_frames = []
            if turn_idx - last_k_turns_for_state_change >= 0:
                prev_prev_pred_turn = pred_dialog['turns'][turn_idx - last_k_turns_for_state_change]
                prev_prev_frames = copy.deepcopy(prev_prev_pred_turn['frames'])
            
            if prev_prev_frames:
                #print(prev_prev_frames)
                assert len(prev_frames) == len(prev_prev_frames)
                for prev_frame, prev_prev_frame in zip(prev_frames, prev_prev_frames):
                    assert prev_frame['service'] == prev_prev_frame['service']

                    for referent_slot, prev_prev_values in prev_prev_frame["state"]["slot_values"].items():
                        if referent_slot not in prev_frame['state']['slot_values']:
                            continue

                        new_values = []
                        for prev_value in prev_frame['state']['slot_values'][referent_slot]:
                            # If the value is already in the prev prev turn, skip.
                            # Because we only take the most recent k turns to build the state.
                            #tobe_removed_values = []
                            for prev_prev_value in prev_prev_values:
                                if prev_prev_value in prev_value:
                                    #tobe_removed_values.append(prev_prev_value)
                                    break
                            else:
                                # No `prev_prev_value` is in `prev_value``.
                                # So `prev_value` is added in the most recent 
                                # k turns (from idx - k - 2 to idx - 2).
                                new_values.append(prev_value) 

                            #for tobe_removed_value in tobe_removed_values:
                            #    prev_prev_values.remove(tobe_removed_value)

                        prev_frame["state"]["slot_values"][referent_slot] = new_values

        for frame in prev_frames:

            if not frame['state']['slot_values']:
                continue

            domain_name = frame['service']

            for referent_slot, values in frame["state"]["slot_values"].items():
                #referent, slot_name = referent_slot.split("-")
                # sometimes, the prediction has more than 1 '-'.
                tmp = referent_slot.split("-")
                referent, slot_name = tmp[0], tmp[1]
                domain_slot_name = f'{domain_name}-{slot_name}'

                domain_slot_name_to_prev_active_referent_slot_values[domain_slot_name].append({
                    'referent': referent,
                    'values': values,
                })
        
        for domain in config.schema:
            if (excluded_domains is not None
                and domain['service_name'] in excluded_domains):
                continue

            domain_name = domain['service_name']
            
            # Input.
            prompt = []
            prompt.append(f'[domain] {domain_name}')

            possible_slots = []
            if add_slot:
                for slot in domain['slots']:
                    domain_slot_name = slot['name']
                    domain_name, slot_name = domain_slot_name.split('-')

                    slot_string = slot_name
                    if add_slot_description:
                        slot_string = f'{slot_string} ({slot["description"]})'

                    if (add_possible_values
                        and slot['is_categorical']):
                        possible_values = config.value_sep.join(slot['possible_values'])
                        slot_string = f'{slot_string} [possible values] {possible_values}'

                    possible_slots.append(slot_string)

                if add_slot or add_slot_description or add_possible_values:
                    possible_slots = config.slot_sep.join(possible_slots)
                    prompt.append(f'[possible slots] {possible_slots}')

            if last_k_turns:
                dialog_text = ' '.join(curr_turns[-last_k_turns:][::-1])
            else:
                dialog_text = ' '.join(curr_turns[::-1])

            prompt_text = ' '.join(prompt)
            dialog_len = len(tokenizer(dialog_text, add_special_tokens=False)['input_ids'])
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])

            dialog_text = truncate(dialog_text, max_num_tokens_for_dialog)
            prompt_text = truncate(prompt_text, max_num_tokens_for_prompt)

            input_text = f'{dialog_text} {prompt_text}'

            # Output target.
            possible_referent_values = []
            prev_state_texts = []
            prev_state_text = config.empty_state
            for slot in domain['slots']:
                domain_slot_name = slot['name']
                domain_name, slot_name = domain_slot_name.split('-')
                active_referent_slot_values = \
                    domain_slot_name_to_active_referent_slot_values.get(
                    #domain_slot_name, config.dummy_referent_slot_values)
                    domain_slot_name, None)

                prev_active_referent_slot_values = \
                    domain_slot_name_to_prev_active_referent_slot_values.get(
                    domain_slot_name, None)
                if prev_active_referent_slot_values is not None:
                    prev_state = convert_active_referent_slot_values_to_possible_comb(
                        prev_active_referent_slot_values, False, False)
                    assert len(prev_state) == 1
                    prev_state_text = f'{slot_name}{config.slot_referent_value_sep}{prev_state[0]}'
                    prev_state_texts.append(prev_state_text)

                # No active referent slot values; Add NONE target later.
                if not active_referent_slot_values:
                    continue

                for possible_comb in \
                    convert_active_referent_slot_values_to_possible_comb(
                    active_referent_slot_values,
                    use_value_permutation and is_train,
                    use_referent_permutation and is_train):
                    possible_comb = f'{slot_name}{config.slot_referent_value_sep}{possible_comb}'
                    possible_referent_values.append(possible_comb)

            if prev_state_texts:
                prev_state_text = config.slot_sep.join(prev_state_texts)
            else:
                prev_state_text = config.empty_state

            if not possible_referent_values:
                possible_comb = convert_active_referent_slot_values_to_possible_comb(
                    config.dummy_referent_slot_values, False, False)[0]
                possible_comb = f'NONE{config.slot_referent_value_sep}{possible_comb}'
                possible_referent_values.append(possible_comb)

            if use_slot_permutation and is_train:
                possible_combs = list(itertools.permutations(possible_referent_values))
            else:
                possible_combs = [possible_referent_values]

            for possible_comb in possible_combs:
                target = config.slot_sep.join(possible_comb)

                prev_state_text = f'{config.prev_state} {prev_state_text}'
                prev_state_len = len(tokenizer(prev_state_text, add_special_tokens=False)['input_ids'])
                prev_state_text = truncate(prev_state_text, max_num_tokens_for_prev_state)
                input_text = f'{dialog_text} {prev_state_text} {prompt_text}'

                example = {
                    'dialogue': input_text,
                    'target': target,
                    'turn_id': turn['turn_id'],
                    'domain': domain_name,
                }
                target_len = len(tokenizer(target, add_special_tokens=False)['input_ids'])
                yield turn_id, example, dialog_len, prompt_len, target_len, prev_state_len


if __name__ == '__main__':
    NFS_DIR = './'
    SPLITS = ['test']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target_turn_id', type=int, required=True)
    parser.add_argument(
        '--last_k_turns_for_state_change',
        type=int,
        required=True,
        help='Use the last k turns in the dialogue history to create the previous state.')
    args = parser.parse_args()

    main_dir = os.path.dirname(args.exp_dir)

    # Paths.
    t5_data_dir = os.path.join(args.exp_dir, f'test_inference/turn_{args.target_turn_id}/t5_data')
    setting_path = os.path.join(main_dir, 'data_setting.yaml')
    os.makedirs(t5_data_dir, exist_ok=True)

    tokenizer = tfs.AutoTokenizer.from_pretrained(args.model_path)
    setting = utils.read_yaml(setting_path)

    for split in SPLITS:
        full_dialog_lens = []
        dialog_lens = []
        prompt_lens = []
        target_lens = []
        prev_state_lens = []
        examples = []

        reference_dir = os.path.join(main_dir, f'{split}')
        reference_pattern = os.path.join(reference_dir, '*.json')

        for ref_path in glob.glob(reference_pattern):
            basename = os.path.basename(ref_path)

            with open(ref_path) as f:
                ref_dialogs = json.load(f)

            # DST.
            if args.target_turn_id > 2:
                pred_path = os.path.join(
                    args.exp_dir,
                    f'test_inference/turn_{args.target_turn_id-2}/mwoz_predictions_aggregated_frames/{split}/{basename}')
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                assert os.path.exists(pred_path), pred_path
                with open(pred_path, 'r') as f:
                    pred_dialogs = json.load(f)
            else:
                assert args.target_turn_id == 1
                pred_dialogs = []
                for ref_dialog in ref_dialogs:
                    pred_dialog = copy.deepcopy(ref_dialog)
                    pred_dialog['turns'] = []
                    pred_dialogs.append(pred_dialog)

            prediction_dir = os.path.join(
                args.exp_dir, f'test_inference/mwoz_predictions_aggregated_frames')
            pred_path = os.path.join(prediction_dir, f'{split}/{basename}')
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            pred_dialogs = sorted(pred_dialogs, key=lambda x: x['dialogue_id'])
            ref_dialogs = sorted(ref_dialogs, key=lambda x: x['dialogue_id'])
            for ref_dialog, pred_dialog in zip(ref_dialogs, pred_dialogs):

                shift_target_turn_id = args.target_turn_id
                # In synthetic data, we always have AGENT turn first.
                # But in real data, we have USER turn first, so we need to shift 
                # index.
                if ref_dialog['turns'][0]['speaker'] == 'USER':
                    shift_target_turn_id = max(args.target_turn_id - 1, 0)
                tmp_turns = copy.deepcopy(
                    ref_dialog['turns'][max(shift_target_turn_id-1, 0):shift_target_turn_id+1])

                for tmp_turn in tmp_turns:
                    # Clean up as a empty frame.
                    tmp_turn['frames'] = copy.deepcopy(config.dummy_frames)
                    pred_dialog['turns'].append(tmp_turn)

            with open(pred_path, 'w') as f:
                json.dump(pred_dialogs, f, indent=2)

            # TLB.
            if args.target_turn_id > 2:
                tlb_pred_path = os.path.join(
                    args.exp_dir,
                    f'test_inference/turn_{args.target_turn_id-2}/mwoz_predictions/{split}/{basename}')
                os.makedirs(os.path.dirname(tlb_pred_path), exist_ok=True)
                assert os.path.exists(pred_path), tlb_pred_path
                with open(tlb_pred_path, 'r') as f:
                    tlb_pred_dialogs = json.load(f)
            else:
                assert args.target_turn_id == 1
                tlb_pred_dialogs = []
                # No issue to take state change reference dialogue, because 
                # we do not use the frame in it.
                for ref_dialog in ref_dialogs:
                    tlb_pred_dialog = copy.deepcopy(ref_dialog)
                    tlb_pred_dialog['turns'] = []
                    tlb_pred_dialogs.append(tlb_pred_dialog)

            prediction_dir = os.path.join(
                args.exp_dir, f'test_inference/mwoz_predictions')
            tlb_pred_path = os.path.join(prediction_dir, f'{split}/{basename}')
            os.makedirs(os.path.dirname(tlb_pred_path), exist_ok=True)
            tlb_pred_dialogs = sorted(tlb_pred_dialogs, key=lambda x: x['dialogue_id'])
            ref_dialogs = sorted(ref_dialogs, key=lambda x: x['dialogue_id'])
            for ref_dialog, tlb_pred_dialog in zip(ref_dialogs, tlb_pred_dialogs):

                shift_target_turn_id = args.target_turn_id
                if ref_dialog['turns'][0]['speaker'] == 'USER':
                    shift_target_turn_id = max(args.target_turn_id - 1, 0)
                tmp_turns = \
                    copy.deepcopy(
                        ref_dialog['turns'][max(
                            shift_target_turn_id-1, 0):shift_target_turn_id+1])

                for tmp_turn in tmp_turns:
                    # Clean up as a empty frame.
                    tmp_turn['frames'] = copy.deepcopy(config.dummy_frames)
                    tlb_pred_dialog['turns'].append(tmp_turn)

            with open(tlb_pred_path, 'w') as f:
                json.dump(tlb_pred_dialogs, f, indent=2)

            dialog_id = [d['dialogue_id'] for d in ref_dialogs]
            # sort by dialog_id.

            for pred_dialog, ref_dialog in zip(pred_dialogs, ref_dialogs):
                is_train = split == 'train'

                if args.target_turn_id >= len(ref_dialog['turns']):
                    continue

                shift_target_turn_id = args.target_turn_id
                if ref_dialog['turns'][0]['speaker'] == 'USER':
                    shift_target_turn_id = max(args.target_turn_id - 1, 0)

                example_generator = preprocess(
                    tokenizer,
                    pred_dialog,
                    ref_dialog,
                    shift_target_turn_id,
                    **setting,
                    last_k_turns_for_state_change=args.last_k_turns_for_state_change,
                    is_train=is_train)

                curr_turns = []
                for turn in ref_dialog['turns']:
                    turn_text = f'[{turn["speaker"]}] {turn["utterance"]}'
                    curr_turns.append(turn_text)

                input_ids = tokenizer(
                    ' '.join(curr_turns),
                    add_special_tokens=False)['input_ids']
                full_dialog_lens.append(len(input_ids))
                
                for (turn_id,
                    example,
                    dialog_len,
                    prompt_len,
                    target_len,
                    prev_state_len) in example_generator:

                    example['dialogue_id'] = ref_dialog['dialogue_id']
                    example['json_basename'] = basename
                    examples.append(example)
                    dialog_lens.append(dialog_len)
                    prompt_lens.append(prompt_len)
                    target_lens.append(target_len)
                    prev_state_lens.append(prev_state_len)

        path = os.path.join(t5_data_dir, f'{split}.json')
        with open(path, 'w') as f:
            for example in examples:
                ex = json.dumps(example)
                print(ex, file=f)