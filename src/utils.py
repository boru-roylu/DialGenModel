import collections
import copy
import json
import yaml
import os

import torch.distributed as dist

import config


def read_yaml(path):
    with open(path, 'r') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    return y


def save_yaml(filename, dic):
    with open(filename, 'w') as f:
        yaml.Dumper.ignore_aliases = lambda self, data: True
        yaml.dump(dic, f, Dumper=yaml.Dumper, sort_keys=False)


def read_schema(schema_path):
    schema = json.load(open(schema_path))
    slot_to_domain = {}
    domains = set()
    for domain in schema:
        for domain_slot in domain["slots"]:
            domain_name, slot_name = domain_slot["name"].split("-")
            slot_to_domain[slot_name] = domain_name
            domains.add(domain)
    return domains, slot_to_domain


def read_mwoz_jsons(mwoz_format_paths):
    basename_to_dialogue_id_to_dialogue = {}
    for path in mwoz_format_paths:
        basename = os.path.basename(path)
        with open(path) as f:
            dialogs = json.load(f)
        
        dialogue_id_to_dialogue = {}
        for dialog in dialogs:
            assert dialog['dialogue_id'] not in dialogue_id_to_dialogue
            dialogue_id_to_dialogue[dialog['dialogue_id']] = dialog
        basename_to_dialogue_id_to_dialogue[basename] = dialogue_id_to_dialogue
    return basename_to_dialogue_id_to_dialogue


def get_ref_mwoz_format_jsons(basename_to_dialogs, keep_frames=False):
    """Get dummy jsons in mwoz format.
    Args:
        basename_to_dialogs: dict of basename to dialogues.
        keep_frames: If True, keep the frames in the dummy jsons. This is for 
            using predicted frames from the previous turn.
    Returns:
        json_basename_to_dialogue_id_to_dummy_dialog: dict of basename to 
            dialogue_id to dummy dialogue.
    """

    #dummy_frames = []
    #for domain_name in config.domain_names:
    #    dummy_frames.append({
    #        'actions': None,
    #        'service': domain_name,
    #        "slots": None,
    #        'state': {
    #            "active_intent": None,
    #            "requested_slots": None,
    #            'slot_values':{},
    #        },
    #    })

    json_basename_to_dialogue_id_to_dummy_dialog = {}
    for basename, dialogs in basename_to_dialogs.items():

        dialogue_id_to_dummy_dialog = collections.OrderedDict()
        for dialog in dialogs.values():
            dialogue_id = dialog['dialogue_id']
            dummy_dialog = collections.OrderedDict(
                dialogue_id=dialogue_id,
                services=dialog['services'],
                turns=[],
            )
            for turn in dialog['turns']:
                if turn['speaker'] == 'USER':
                    if keep_frames:
                        frames = turn['frames']
                    else:
                        frames = copy.deepcopy(config.dummy_frames)
                    turn = collections.OrderedDict(
                        speaker='USER',
                        turn_id=turn['turn_id'],
                        utterance=turn['utterance'],
                        frames=frames,
                    )
                elif turn['speaker'] == "SYSTEM":
                    turn = collections.OrderedDict(
                        speaker='SYSTEM',
                        turn_id=turn['turn_id'],
                        utterance=turn['utterance'],
                        frames=[],
                    )
                else:
                   raise ValueError(f"Unknown speaker: {turn['speaker']}")
                dummy_dialog['turns'].append(turn)
            assert dialogue_id not in dialogue_id_to_dummy_dialog, dialogue_id + ' already exists'
            dialogue_id_to_dummy_dialog[dialogue_id] = dummy_dialog

        json_basename_to_dialogue_id_to_dummy_dialog[basename] = dialogue_id_to_dummy_dialog
    return json_basename_to_dialogue_id_to_dummy_dialog


def get_mwoz_format_json_from_predictions(
    predictions,
    examples,
    json_basename_to_dialogue_id_to_dialog):

    for ex, pred in zip(examples, predictions):
        dummy_dialog = json_basename_to_dialogue_id_to_dialog[
            ex['json_basename']][ex['dialogue_id']]
        turn_id = ex['turn_id']
        frame_idx = config.domain_names.index(ex['domain'])
        turn = dummy_dialog['turns'][turn_id]
        frame = turn['frames'][frame_idx]
        assert ex['domain'] == frame['service']
        assert turn['speaker'] == 'USER'

        referent_values_list = pred.split(config.cross_referent_sep)
        for referent_values in referent_values_list:
            if config.referent_value_sep not in referent_values:
                continue
            if len(referent_values.split(config.referent_value_sep)) != 2:
                continue
            values, referent = referent_values.split(config.referent_value_sep)
            values = values.split(config.value_sep)
            referent = referent.strip()
            values = list(map(lambda x: x.strip(), values))

            if set(values) == {'NONE'}:
                continue

            referent_slot = f'{referent}-{ex["slot"]}'
            frame['state']['slot_values'][referent_slot] = values
    return json_basename_to_dialogue_id_to_dialog


def get_mwoz_format_json_from_predictions_non_oracle(
    predictions,
    examples,
    json_basename_to_dialogue_id_to_dialog):

    for ex, pred in zip(examples, predictions):
        dummy_dialog = json_basename_to_dialogue_id_to_dialog[
            ex['json_basename']][ex['dialogue_id']]
        turn_id = ex['turn_id']
        frame_idx = config.domain_names.index(ex['domain'])
        turn = dummy_dialog['turns'][turn_id]
        frame = turn['frames'][frame_idx]
        assert ex['domain'] == frame['service']
        assert turn['speaker'] == 'USER'

        slots_referent_values_list = pred.split(config.slot_sep)
        for slot_referent_values_list in slots_referent_values_list:
            if config.slot_referent_value_sep not in slot_referent_values_list:
                continue
            if len(slot_referent_values_list.split(config.slot_referent_value_sep)) != 2:
                continue
            slot, referent_values_list = slot_referent_values_list.split(
                config.slot_referent_value_sep)
            slot = slot.strip()

            if slot == 'NONE':
                continue

            # Damage Part [srv] Front [rv] Caller [cv] Back [rv] Other Driver
            # referent_values_list: Front [rv] Caller [cv] Back [rv] Other Driver
            # referent_values_list: ["Front [rv] Caller", "Back [rv] Other Driver"]
            referent_values_list = referent_values_list.split(config.cross_referent_sep)
            for referent_values in referent_values_list:
                if config.referent_value_sep not in referent_values:
                    continue
                if len(referent_values.split(config.referent_value_sep)) != 2:
                    continue
                values, referent = referent_values.split(config.referent_value_sep)
                values = values.split(config.value_sep)
                referent = referent.strip()

                values = list(map(lambda x: x.strip(), values))

                if set(values) == {'NONE'}:
                    continue

                #print(f'{referent_slot = }')
                #print(f'{values = }')

                referent_slot = f'{referent}-{slot}'
                frame['state']['slot_values'][referent_slot] = values

    return json_basename_to_dialogue_id_to_dialog


def aggregate_frames(json_basename_to_dialogue_id_to_dialog):
    for basename, dialog_id_to_dialog in \
        json_basename_to_dialogue_id_to_dialog.items():
        for dialog_id, dialog in dialog_id_to_dialog.items():
            touch_first_user_turn = False
            prev_frames = None
            for turn_idx, turn in enumerate(dialog['turns']):
                if turn['speaker'] != 'USER':
                    continue

                if not touch_first_user_turn:
                    touch_first_user_turn = True
                    continue

                if prev_frames is None:
                    prev_frames = copy.deepcopy(turn['frames'])
                    continue

                for frame_idx, (prev_frame, curr_frame) in \
                    enumerate(zip(prev_frames, turn['frames'])):
                    for referent_slot_name, prev_values in \
                        prev_frame['state']['slot_values'].items():
                        curr_values = curr_frame['state']['slot_values'].get(
                            referent_slot_name, [])
                        prev_values = copy.deepcopy(prev_values)
                        curr_values.extend(prev_values)
                        curr_frame['state']['slot_values'][referent_slot_name] = \
                            curr_values
                prev_frames = copy.deepcopy(turn['frames'])


def clean_ops_in_frames_for_state_change_to_tlb(
    json_basename_to_dialogue_id_to_dialog):
    for basename, dialog_id_to_dialog in \
        json_basename_to_dialogue_id_to_dialog.items():
        for dialog_id, dialog in dialog_id_to_dialog.items():
            for turn_idx, turn in enumerate(dialog['turns']):
                if turn['speaker'] != 'USER':
                    continue

                for frame_idx, curr_frame in enumerate(turn['frames']):

                    pop_referent_slot_names = []
                    for referent_slot_name, curr_values in \
                        curr_frame['state']['slot_values'].items():

                        if not curr_values:
                            continue

                        tlb_values = []
                        for curr_value in curr_values:
                            if config.value_op_sep in curr_value:
                                tmp = curr_value.split(config.value_op_sep)
                                curr_value, curr_op = tmp[0], tmp[1]
                                # keep and same will be kept.
                                if curr_op in {'delete', 'concat'}:
                                    continue
                            tlb_values.append(curr_value)

                        if not tlb_values:
                            pop_referent_slot_names.append(referent_slot_name)

                        curr_frame['state']['slot_values'][referent_slot_name] = tlb_values

                    for pop_referent_slot_name in pop_referent_slot_names:
                        curr_frame['state']['slot_values'].pop(pop_referent_slot_name)


def aggregate_frames_state_change(
    json_basename_to_dialogue_id_to_dialog,
    target_turn_id=None,
    skip_op_if_error=False):

    for basename, dialog_id_to_dialog in \
        json_basename_to_dialogue_id_to_dialog.items():
        for dialog_id, dialog in dialog_id_to_dialog.items():
            touch_first_user_turn = False
            prev_frames = None
            if target_turn_id is not None:
                curr_target_turn_id = target_turn_id
                if dialog['turns'][0]['speaker'] == 'USER':
                    curr_target_turn_id = target_turn_id - 1
            for turn_idx, turn in enumerate(dialog['turns']):
                if turn['speaker'] != 'USER':
                    continue

                if not touch_first_user_turn:
                    touch_first_user_turn = True
                    continue

                if prev_frames is None:
                    prev_frames = copy.deepcopy(turn['frames'])
                    continue

                assert len(prev_frames) == len(turn['frames'])

                if curr_target_turn_id is not None:
                    if turn_idx < curr_target_turn_id:
                        prev_frames = copy.deepcopy(turn['frames'])
                        continue
                    assert turn_idx == curr_target_turn_id

                for frame_idx, (prev_frame, curr_frame) in \
                    enumerate(zip(prev_frames, turn['frames'])):
                    assert prev_frame['service'] == curr_frame['service']

                    curr_referent_slot_names = set(curr_frame['state']['slot_values'].keys()) 
                    prev_referent_slot_names = set(prev_frame['state']['slot_values'].keys())
                    referent_slot_names = curr_referent_slot_names.union(prev_referent_slot_names)

                    for referent_slot_name in referent_slot_names:
                        curr_values = curr_frame['state']['slot_values'].get(
                            referent_slot_name, [])
                        prev_values = prev_frame['state']['slot_values'].get(
                            referent_slot_name, [])
                        
                        #print(f'{turn_idx = } {referent_slot_name = } {curr_values = }')
                        updated_curr_values = copy.deepcopy(prev_values)
                        if not curr_values:
                            curr_frame['state']['slot_values'][referent_slot_name] = \
                                updated_curr_values
                            continue

                        concat_values = []
                        #print(f'{referent_slot_name = } ||| {curr_values = }')
                        for curr_value in curr_values:
                            # No operation, just add the value.
                            if config.value_op_sep not in curr_value:
                                updated_curr_values.append(curr_value)
                                continue

                            tmp = curr_value.split(config.value_op_sep)
                            curr_value, curr_op = tmp[0], tmp[1]

                            # The operation is either concat or delete.
                            # Remove the value from the previous values.
                            if not skip_op_if_error:
                                assert curr_value in prev_values

                            # We do not consider same operation in CB 
                            # because it already appear in the previous state.
                            if curr_op == config.op_to_token['same']:
                                continue
                                #if curr_value in updated_curr_values:
                                #    continue
                                #else:
                                #    print(f'[same] op: {curr_value = } is not in {updated_curr_values = }')
                                #updated_curr_values.append(curr_value)
                            elif curr_op == config.op_to_token['delete']:
                                if curr_value in updated_curr_values:
                                    updated_curr_values.remove(curr_value)
                                else:
                                    print(f'[delete] op: {curr_value = } is not in {updated_curr_values = }')
                            elif curr_op == config.op_to_token['concat']:
                                concat_values.append(curr_value)  
                            elif skip_op_if_error:
                                continue
                            else:
                                raise ValueError(f'The op is not available! {curr_op}')

                        if concat_values:
                            if not skip_op_if_error:
                                assert len(concat_values) > 1
                            for concat_value in concat_values:
                                if concat_value in updated_curr_values:
                                    updated_curr_values.remove(concat_value)
                            updated_curr_values.append(' '.join(concat_values))
                            
                        curr_frame['state']['slot_values'][referent_slot_name] = \
                            updated_curr_values
                        
                prev_frames = copy.deepcopy(turn['frames'])
    #return json_basename_to_dialogue_id_to_dialog
    

def get_rank():
    if not dist.is_available():
        return -1
    if not dist.is_initialized():
        return -1
    return dist.get_rank()


def is_local_master():
    return get_rank() in [-1, 0]
