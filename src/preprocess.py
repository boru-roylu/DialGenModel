import argparse
import collections
import itertools
import os
import pprint as pp

import transformers as tfs
import glob
import json

import numpy as np
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
            comb = \
                config.referent_value_sep.join([values_string, referent_string])
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
    dialog,
    add_slot,
    add_slot_description,
    add_possible_values,
    use_slot_permutation,
    use_value_permutation,
    use_referent_permutation,
    max_num_tokens_for_dialog,
    max_num_tokens_for_prompt,
    last_k_turns=None,
    excluded_domains=None,
    is_train=False):

    def truncate(text, max_num_tokens):
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_num_tokens,
            truncation=True)['input_ids']
        return tokenizer.decode(input_ids)

    curr_turns = []
    for turn in dialog['turns']:
        turn_text = f'[{turn["speaker"]}] {turn["utterance"]}'
        curr_turns.append(turn_text)

        if turn["speaker"] != "USER":
            continue

        domain_slot_name_to_active_referent_slot_values = collections.defaultdict(list)
        active_domains = set()
        for frame in turn['frames']:

            if not frame['state']['slot_values']:
                continue

            domain_name = frame['service']
            active_domains.add(domain_name)
            for referent_slot, values in frame["state"]["slot_values"].items():
                referent, slot_name = referent_slot.split("-")

                domain_slot_name = f'{domain_name}-{slot_name}'

                domain_slot_name_to_active_referent_slot_values[
                    domain_slot_name].append({
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
                        possible_values = \
                            config.value_sep.join(slot['possible_values'])
                        slot_string = \
                            f'{slot_string} [possible values] {possible_values}'

                    possible_slots.append(slot_string)

                if add_slot or add_slot_description or add_possible_values:
                    possible_slots = config.slot_sep.join(possible_slots)
                    prompt.append(f'[possible slots] {possible_slots}')

            if last_k_turns:
                dialog_text = ' '.join(curr_turns[-last_k_turns:][::-1])
            else:
                dialog_text = ' '.join(curr_turns[::-1])
            prompt_text = ' '.join(prompt)

            # Tokenize to get stats, so ignore the length limit warning.
            dialog_len = len(
                tokenizer(dialog_text, add_special_tokens=False)['input_ids'])
            prompt_len = len(
                tokenizer(prompt_text, add_special_tokens=False)['input_ids'])

            dialog_text = truncate(dialog_text, max_num_tokens_for_dialog)
            prompt_text = truncate(prompt_text, max_num_tokens_for_prompt)

            input_text = f'{dialog_text} {prompt_text}'

            # Output target.
            possible_referent_values = []
            for slot in domain['slots']:
                domain_slot_name = slot['name']
                domain_name, slot_name = domain_slot_name.split('-')
                active_referent_slot_values = \
                    domain_slot_name_to_active_referent_slot_values.get(
                    domain_slot_name, None)

                if not active_referent_slot_values:
                    continue

                for possible_comb in \
                    convert_active_referent_slot_values_to_possible_comb(
                    active_referent_slot_values,
                    use_value_permutation and is_train,
                    use_referent_permutation and is_train):
                    possible_comb = (f'{slot_name}'
                                     f'{config.slot_referent_value_sep}'
                                     f'{possible_comb}')
                    possible_referent_values.append(possible_comb)

            if not possible_referent_values:
                possible_comb = \
                    convert_active_referent_slot_values_to_possible_comb(
                        config.dummy_referent_slot_values, False, False)[0]
                possible_comb = \
                    f'NONE{config.slot_referent_value_sep}{possible_comb}'
                possible_referent_values.append(possible_comb)

            if use_slot_permutation and is_train:
                possible_combs = \
                    list(itertools.permutations(possible_referent_values))
            else:
                possible_combs = [possible_referent_values]

            for possible_comb in possible_combs:
                target = config.slot_sep.join(possible_comb)
                example = {
                    'dialogue': input_text,
                    'target': target,
                    'turn_id': turn['turn_id'],
                    'domain': domain_name,
                }
                target_len = len(
                    tokenizer(target, add_special_tokens=False)['input_ids'])
                yield example, dialog_len, prompt_len, target_len


if __name__ == '__main__':
    NFS_DIR = './'
    SPLITS = ['train', 'dev', 'test']

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    # Paths.
    t5_data_dir = os.path.join(args.main_dir, 't5_data')
    setting_path = os.path.join(args.main_dir, 'data_setting.yaml')
    os.makedirs(t5_data_dir, exist_ok=True)

    tokenizer = tfs.AutoTokenizer.from_pretrained(args.model_path)
    setting = utils.read_yaml(setting_path)

    print('Paths')
    print('main_dir = ', args.main_dir)
    print('t5_data_dir = ', t5_data_dir)
    print('Setting')
    pp.pprint(setting)
    print()

    for split in SPLITS:
        full_dialog_lens = []
        dialog_lens = []
        prompt_lens = []
        target_lens = []
        print('='*20, f'{split}', '='*20)
        examples = []
        for path in glob.glob(os.path.join(args.main_dir, f'{split}/*.json')):
            basename = os.path.basename(path)
            with open(path) as f:
                dialogs = json.load(f)

            for dialog_idx, dialog in enumerate(dialogs):
                is_train = split == 'train'
                example_generator = preprocess(
                    tokenizer,
                    dialog,
                    **setting,
                    is_train=is_train)

                curr_turns = []
                for turn in dialog['turns']:
                    turn_text = f'[{turn["speaker"]}] {turn["utterance"]}'
                    curr_turns.append(turn_text)

                input_ids = tokenizer(
                    ' '.join(curr_turns),
                    add_special_tokens=False)['input_ids']
                full_dialog_lens.append(len(input_ids))
                

                for example, dialog_len, prompt_len, target_len in \
                    example_generator:
                    example['dialogue_id'] = dialog['dialogue_id']
                    example['json_basename'] = basename
                    examples.append(example)
                    dialog_lens.append(dialog_len)
                    prompt_lens.append(prompt_len)
                    target_lens.append(target_len)

        path = os.path.join(t5_data_dir, f'{split}.json')
        with open(path, 'w') as f:
            for example in examples:
                ex = json.dumps(example)
                print(ex, file=f)

        print('# examples = ', len(examples))
        print(f'full_dialog_len max  = {max(full_dialog_lens):.2f}')
        print(f'full_dialog_len min  = {min(full_dialog_lens):.2f}')
        print(f'full_dialog_len mean = {np.mean(full_dialog_lens):.2f}')
        print(f'full_dialog_len std  = {np.std(full_dialog_lens):.2f}')
        print(f'-'*10)
        print(f'dialog_len max       = {max(dialog_lens):.2f}')
        print(f'dialog_len min       = {min(dialog_lens):.2f}')
        print(f'dialog_len mean      = {np.mean(dialog_lens):.2f}')
        print(f'dialog_len std       = {np.std(dialog_lens):.2f}')
        print(f'-'*10)
        print(f'prompt_len max       = {max(prompt_lens):.2f}')
        print(f'prompt_len min       = {min(prompt_lens):.2f}')
        print(f'prompt_len mean      = {np.mean(prompt_lens):.2f}')
        print(f'prompt_len std       = {np.std(prompt_lens):.2f}')
        print(f'-'*10)
        print(f'target_len max       = {max(target_lens):.2f}')
        print(f'target_len min       = {min(target_lens):.2f}')
        print(f'target_len mean      = {np.mean(target_lens):.2f}')
        print(f'target_len std       = {np.std(target_lens):.2f}')
        print()