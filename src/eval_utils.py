import difflib
import os
import re 
import string

import numpy as np
import pandas as pd

import config

# TODO(roylu): use function argument.
IS_STATE_CHANGE = os.environ.get('IS_STATE_CHANGE', False)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score_fn(x, y):
    return int(set(x) == set(y))


def digit_to_word(number):
    # create a dictionary that maps each digit to its English word equivalent
    word_dict = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
    }

    tens_dict = {
        2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty',
        6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'
    }
    
    # if the input number is a single digit, return its English word equivalent
    if number in word_dict:
        return word_dict[number]
    
    # if the input number is a two-digit number, convert its tens and ones digits to English words
    elif number < 100:
        tens_digit = number // 10
        ones_digit = number % 10
        if tens_digit == 1:
            # for numbers between 10 and 19, use special English words
            special_dict = {
                10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
                15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'
            }
            return special_dict[number]
        else:
            # for other two-digit numbers, use a combination of tens and ones digits
            if ones_digit == 0:
                return tens_dict[tens_digit]
            else:
                return tens_dict[tens_digit] + ' ' + word_dict[ones_digit]
    
    # if the input number is a three-digit number, convert its hundreds digit and the remaining two digits to English words
    elif number < 1000:
        hundreds_digit = number // 100
        remainder = number % 100
        if remainder == 0:
            # for numbers like 100, 200, 300, etc., use the word 'hundred'
            return word_dict[hundreds_digit] + ' hundred'
        else:
            # for other three-digit numbers, use a combination of hundreds, tens, and ones digits
            return word_dict[hundreds_digit] + ' hundred ' + digit_to_word(remainder)
    
    # if the input number is a four-digit number or larger, convert its thousands digit and the remaining digits to English words
    elif number < 1000000:
        thousands = number // 1000
        remainder = number % 1000
        if remainder == 0:
            # for numbers like 1000, 2000, 3000, etc., use the word 'thousand'
            return digit_to_word(thousands) + ' thousand'
        else:
            # for other four-digit numbers or larger, use a combination of thousands, hundreds, tens, and ones digits
            return digit_to_word(thousands) + ' thousand ' + digit_to_word(remainder)
    
    # if the input number is too large, raise an error
    else:
        return ' '.join([digit_to_word(int(num)) for num in str(number)])


def replace_numbers(text):
    #numbers = re.findall(r'^(\d+(?:[\.\,]\d{3})?)$', text)
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    numbers_wo_punctuation = [re.sub(r'[^\w\s]','',number) for number in numbers]
    words = list(map(digit_to_word, map(int, numbers_wo_punctuation)))
    number_to_word = dict(zip(numbers, words))

    for number, word in number_to_word.items():
        text = text.replace(number, word)
    return text


def convert_to_normal_dict(dic):
    return {k: dict(v) for k, v in dic.items()}


def get_tuples_from_frames(frames):
    tuples = []
    for frame in frames:
        domain = frame['service']

        referent_slot_to_values = frame['state']['slot_values']
        # Empty frame.
        if not referent_slot_to_values:
            continue

        for referent_slot, values in referent_slot_to_values.items():
            if len(referent_slot.split('-')) != 2:
                continue
            referent, slot = referent_slot.split('-')

            if IS_STATE_CHANGE:
                value_and_op_list = values
                values = []
                for value_and_op in value_and_op_list:
                    op_token = '[keep]'
                    if config.value_op_sep in value_and_op:
                        # Sometimes, the model predict more than 1 op. 
                        # We take the 1st op.
                        tmp = value_and_op.split(config.value_op_sep)
                        value, op_token = tmp[0], tmp[1]
                    else:
                        value = value_and_op

                    value = normalize_answer(value)
                    if op_token == '[keep]':
                        values.append(value)
                    else:
                        values.append(f'{value}{config.value_op_sep}{op_token}')
            else:
                values = list(map(normalize_answer, values))
            referent = normalize_answer(referent)

            domain_slot = f'{domain}-{slot}'
            is_categorical = \
                config.domain_slot_to_is_categorical.get(domain_slot, None)
            slot_type = 'categorical' if is_categorical else 'extractive'
            tuples.append({
                'domain': domain,
                'slot': slot,
                'slot_type': slot_type,
                'referent': referent,
                'values': values,
            })
    return tuples


def is_empty_frames(frames):
    for frame in frames:
        if frame['state']['slot_values']:
            return False
    return True


def enumerate_pred_and_ref_frames(
    pred_dialog,
    ref_dialog,
    dst_percentile=None,
    frame_key='frames'):

    assert ref_dialog['dialogue_id'] == pred_dialog['dialogue_id']
    dialog_id = pred_dialog['dialogue_id']
    assert len(pred_dialog['turns']) == len(ref_dialog['turns'])

    if dst_percentile is not None:
        import math
        assert dst_percentile <= 4
        num_turns = len(ref_dialog['turns'])

        if dst_percentile == 4:
            target_idx = num_turns - 1
            if ref_dialog['turns'][target_idx]['speaker'] == 'SYSTEM':
                target_idx -= 1
        else:
            #target_idx = math.floor(num_turns * 0.25 * dst_percentile) - 1
            target_idx = math.ceil(num_turns * 0.25 * dst_percentile)
            if ref_dialog['turns'][target_idx]['speaker'] == 'SYSTEM':
                target_idx += 1
            #target_idx = min(math.ceil(num_turns * 0.01 * dst_percentile), num_turns - 1)
            #if ref_dialog['turns'][target_idx]['speaker'] == 'SYSTEM':
            #    if (target_idx + 1) < num_turns:
            #        target_idx += 1
            #    else:
            #        target_idx -= 1
        ref_turns = [ref_dialog['turns'][target_idx]]
        pred_turns = [pred_dialog['turns'][target_idx]]
    else:
        ref_turns = ref_dialog['turns']
        pred_turns = pred_dialog['turns']

    for pred_turn, ref_turn in zip(pred_turns, ref_turns):
        assert ref_turn['speaker'] == pred_turn['speaker']
        assert ref_turn['turn_id'] == pred_turn['turn_id']
        turn_id = pred_turn['turn_id']
        if ref_turn['speaker'] == 'SYSTEM':
            continue
        ref_frames = ref_turn[frame_key]
        # All DST, TLB, State-change use frames to store output.
        # Thus, postprocessing needed for state-change predictions.
        pred_frames = pred_turn['frames']
        if is_empty_frames(ref_frames) and is_empty_frames(pred_frames):
            continue
        yield dialog_id, turn_id, pred_frames, ref_frames


def flat_dict_of_dict_of_int(dic):
    flat_list = [value for inner_dict in dic.values() for value in inner_dict.values()]
    return flat_list


def flat_dict_of_dict(dic):
    ret = {}
    for inner_dic in dic.values():
        ret.update(inner_dic)
    return ret


def group(df, cond):
    key_to_subdf = {}
    for idx, g in df.groupby(cond):
        key_to_subdf[idx] = g
    return key_to_subdf

    
def safe_div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def partial_true_positive_for_freeform_values(list1, list2):
    # list1: predicted values.
    # list2: gold values.
    list1 = list(map(normalize_answer, list1))
    list2 = list(map(normalize_answer, list2))
    result = []
    for string1 in list1:
        best_match = ("", 0)
        string1 = string1.split(' ')
        for string2 in list2:
            string2 = string2.split(' ')
            ratio = difflib.SequenceMatcher(None, string1, string2).ratio()
            if ratio > best_match[1]:
                best_match = (' '.join(string2), ratio)

        # Not matched anything in the reference (list2).
        if best_match[1] == 0:
            continue

        if list2:
            result.append(best_match[1])
            if best_match[0] in list2:
                list2.remove(best_match[0])  # Each string in list2 can only be used once
    num_tp = sum(result)
    num_match = len(result)
    return num_tp, num_match


def true_positive_for_categorical_values(preds, golds):
    preds = list(map(normalize_answer, preds))
    golds = list(map(normalize_answer, golds))
    num_tp = 0
    for pred in preds:
        if pred in golds:
            num_tp += 1
            golds.remove(pred)

    # We return 2 num_tp here because we use the unified function 
    # `true_positive_fn` to compute both categorical and freeform values.
    return num_tp, num_tp


def true_positive_fn(slot_type):
    if slot_type == 'categorical':
        return true_positive_for_categorical_values
    elif slot_type == 'extractive':
        return partial_true_positive_for_freeform_values
    else:
        raise ValueError(f'Unknown slot type: {slot_type}')


def merge_turn_fn(row):
    ems = row['em'].tolist()
    precisions = row['precision'].tolist()
    recalls = row['recall'].tolist()
    valid_precisions = row['valid_precision'].tolist()
    valid_recalls = row['valid_recall'].tolist()

    precisions = [p for p, v in zip(precisions, valid_precisions) if v]
    recalls = [r for r, v in zip(recalls, valid_recalls) if v]

    em = np.prod(ems) if ems else 0
    precision = np.mean(precisions) if precisions else 0
    recall = np.mean(recalls) if recalls else 0

    return pd.Series({
        'em': em,
        'precision': precision,
        'recall': recall,
        'valid_precision': np.any(valid_precisions),
        'valid_recall': np.any(valid_recalls),
    })


def merge_dialog_fn(row):
    ems = row['em'].tolist()
    precisions = row['precision'].tolist()
    recalls = row['recall'].tolist()
    valid_precisions = row['valid_precision'].tolist()
    valid_recalls = row['valid_recall'].tolist()

    precisions = [p for p, v in zip(precisions, valid_precisions) if v]
    recalls = [r for r, v in zip(recalls, valid_recalls) if v]

    em = np.mean(ems) if ems else 0
    precision = np.mean(precisions) if precisions else 0
    recall = np.mean(recalls) if recalls else 0

    return pd.Series({
        'em': em,
        'precision': precision,
        'recall': recall,
        'valid_precision': np.any(valid_precisions),
        'valid_recall': np.any(valid_recalls),
    })


def group(df, cond):
    key_to_subdf = {}
    for idx, g in df.groupby(cond):
        key_to_subdf[idx] = g
    return key_to_subdf


def safe_div(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def get_value_em_f1(ref_df, pred_df, grouping_cond, avg_by='turn_id'):
    ref_key_to_subdf = group(ref_df, grouping_cond)
    pred_key_to_subdf = group(pred_df, grouping_cond)
    
    all_keys = set(ref_key_to_subdf.keys()) | set(pred_key_to_subdf.keys())
    dialog_id_idx = grouping_cond.index('dialog_id')
    all_keys = sorted(all_keys, key=lambda x: x[dialog_id_idx])

    # Each row contain EM, precision, recall, f1 for a condition.
    rows = []
    for idx, key in enumerate(all_keys):
        ref_subdf = ref_key_to_subdf.get(key, None)
        pred_subdf = pred_key_to_subdf.get(key, None)
        dialog_id_idx = grouping_cond.index('dialog_id')
        turn_id_idx = grouping_cond.index('turn_id')

        if avg_by == 'referent':
            referent_idx = grouping_cond.index('referent')
    
        em, total_num_tp, total_num_fp, total_num_fn = 0, 0, 0, 0

        if ref_subdf is not None and pred_subdf is not None:
            # Prediction something in the turns with at least one annotations.

            # Gets all slot types for the turn.
            slot_types = (set(ref_subdf['slot_type'].tolist()) 
                          | set(pred_subdf['slot_type'].tolist()))

            # Compute true positives, false positives, and false negatives 
            # for `categorical` and `extractive` slot types, respectively.
            for slot_type in slot_types:
                slot_type_ref_subdf = \
                    ref_subdf[ref_subdf['slot_type'] == slot_type]
                slot_type_pred_subdf = \
                    pred_subdf[pred_subdf['slot_type'] == slot_type]
                ref_values = sum(slot_type_ref_subdf['values'].tolist(), [])
                pred_values = sum(slot_type_pred_subdf['values'].tolist(), [])

                #assert ref_values or pred_values
                if not (ref_values or pred_values):
                    continue

                # We might predict categorical slots but the reference only has 
                # extractive slots.
                if not ref_values or not pred_values:
                    total_num_fp += len(pred_values)
                    total_num_fn += len(ref_values)
                    continue

                em = em_score_fn(pred_values, ref_values)
                num_tp, num_match = \
                    true_positive_fn(slot_type)(pred_values, ref_values)

                total_num_tp += num_tp
                # FN = num reference - num match.
                total_num_fn += (len(ref_values) - num_match)
                # FP = num predict - num match.
                total_num_fp += (len(pred_values) - num_match)

        # Underprediction.
        elif ref_subdf is not None and pred_subdf is None:
            refs = sum(ref_subdf['values'].tolist(), [])
            total_num_fn += len(refs)
            
        # Overprediction.
        elif ref_subdf is None and pred_subdf is not None:
            preds = sum(pred_subdf['values'].tolist(), [])
            total_num_fp += (len(preds))

        else:
            continue

        valid_precision = (total_num_tp + total_num_fp) > 0
        valid_recall = (total_num_tp + total_num_fn) > 0
        precision = safe_div(total_num_tp, total_num_tp + total_num_fp)
        recall = safe_div(total_num_tp, total_num_tp + total_num_fn)

        row = {
            'dialog_id': key[dialog_id_idx],
            'turn_id': key[turn_id_idx],
            'em': em,
            'precision': precision,
            'recall': recall,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
        }
        if avg_by == 'referent':
            assert 'referent' in grouping_cond
            row['referent'] = key[referent_idx]
        rows.append(row)

    df = pd.DataFrame.from_dict(rows, orient='columns')

    df = df.groupby(['dialog_id', avg_by]).apply(merge_turn_fn).reset_index()
    df = df.groupby(['dialog_id']).apply(merge_dialog_fn).reset_index()
    #df = df[['em', 'precision', 'recall', 'f1']]
    #df = df[['em', 'precision', 'recall']]
    result = {}
    result['EM'] = df['em'].mean()
    result['Precision'] = p = (df['precision'] * df['valid_precision']).sum() / df['valid_precision'].sum()
    result['Recall'] = r = (df['recall'] * df['valid_recall']).sum() / df['valid_recall'].sum()
    result['F1'] = safe_div(2 * p * r, p + r)
    return result


def get_referent_or_slot_em_f1(ref_df, pred_df, grouping_cond, target_col):
    ref_key_to_subdf = group(ref_df, grouping_cond)
    pred_key_to_subdf = group(pred_df, grouping_cond)
    
    all_keys = set(ref_key_to_subdf.keys()) | set(pred_key_to_subdf.keys())
    dialog_id_idx = grouping_cond.index('dialog_id')
    all_keys = sorted(all_keys, key=lambda x: x[dialog_id_idx])
    rows = []
    for idx, key in enumerate(all_keys):
        ref_subdf = ref_key_to_subdf.get(key, None)
        pred_subdf = pred_key_to_subdf.get(key, None)
        dialog_id_idx = grouping_cond.index('dialog_id')
        turn_id_idx = grouping_cond.index('turn_id')

        # Because we compute referent or slot. They must be categorical.
        slot_type = 'categorical'

        em, total_num_tp, total_num_fp, total_num_fn = 0, 0, 0, 0
        if ref_subdf is not None and pred_subdf is not None:
            ref_targets = ref_subdf[target_col].tolist()
            pred_targets = pred_subdf[target_col].tolist()
            if not ref_targets or not pred_targets:
                total_num_fp += len(pred_targets)
                total_num_fn += len(ref_targets)
                continue

            em = em_score_fn(pred_targets, ref_targets)
            num_tp, num_match = \
                true_positive_fn(slot_type)(pred_targets, ref_targets)

            total_num_tp += num_tp
            # FN = num reference - num match.
            total_num_fn += (len(ref_targets) - num_match)
            # FP = num predict - num match.
            total_num_fp += (len(pred_targets) - num_match)

        # Underprediction.
        elif ref_subdf is not None and pred_subdf is None:
            ref_targets = ref_subdf[target_col].tolist()
            total_num_fn += len(ref_targets)
            
        # Overprediction.
        elif ref_subdf is None and pred_subdf is not None:
            pred_targets = pred_subdf[target_col].tolist()
            total_num_fp += (len(pred_targets))

        else:
            continue

        valid_precision = (total_num_tp + total_num_fp) > 0
        valid_recall = (total_num_tp + total_num_fn) > 0
        precision = safe_div(total_num_tp, total_num_tp + total_num_fp)
        recall = safe_div(total_num_tp, total_num_tp + total_num_fn)

        row = {
            'dialog_id': key[dialog_id_idx],
            'turn_id': key[turn_id_idx],
            'em': em,
            'precision': precision,
            'recall': recall,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
        }
        rows.append(row)

    df = pd.DataFrame.from_dict(rows, orient='columns')
    df = df.groupby(['dialog_id', 'turn_id']).apply(merge_turn_fn).reset_index()
    df = df.groupby(['dialog_id']).apply(merge_dialog_fn).reset_index()

    result = {}
    result['EM'] = df['em'].mean()
    result['Precision'] = p = (df['precision'] * df['valid_precision']).sum() / df['valid_precision'].sum()
    result['Recall'] = r = (df['recall'] * df['valid_recall']).sum() / df['valid_recall'].sum()
    result['F1'] = safe_div(2 * p * r, p + r)
    return result


def get_turn_tuples(
    ref_id_to_dialogs,
    pred_id_to_dialogs,
    dst_percentile=None,
    frame_key='frames'):

    all_ref_tuples = []
    all_pred_tuples = []
    for dialog_id in ref_id_to_dialogs.keys():

        pred_dialog = pred_id_to_dialogs[dialog_id]
        ref_dialog = ref_id_to_dialogs[dialog_id]

        for dialog_id, turn_id, pred_frames, ref_frames in \
            enumerate_pred_and_ref_frames(
                pred_dialog, ref_dialog, dst_percentile, frame_key):
            if len(pred_frames) == 0 and len(ref_frames) == 0:
                continue

            pred_tuples = get_tuples_from_frames(pred_frames)
            ref_tuples = get_tuples_from_frames(ref_frames)

            def add_id(name, _id, tuples):
                for tup in tuples:
                    tup[name] = _id

            add_id('dialog_id', dialog_id, pred_tuples)
            add_id('dialog_id', dialog_id, ref_tuples)
            add_id('turn_id', turn_id, pred_tuples)
            add_id('turn_id', turn_id, ref_tuples)

            all_ref_tuples.extend(ref_tuples)
            all_pred_tuples.extend(pred_tuples)
    return all_ref_tuples, all_pred_tuples


def get_result(ref_df, pred_df):
    """Compute the result for all dialogs."""

    if len(pred_df) == 0:
        return {
            'EM': 0,
            'Precision': 0,
            'Recall': 0,
            'F1': 0,

            'R-EM': 0,
            'R-Precision': 0,
            'R-Recall': 0,
            'R-F1': 0,

            'S-EM': 0,
            'S-Precision': 0,
            'S-Recall': 0,
            'S-F1': 0,

            'RS-EM': 0,
            'RS-Precision': 0,
            'RS-Recall': 0,
            'RS-F1': 0,

            'SV-EM': 0,
            'SV-Precision': 0,
            'SV-Recall': 0,
            'SV-F1': 0,

            'v2-EM': 0,
            'v2-Precision': 0,
            'v2-Recall': 0,
            'v2-F1': 0,
        }

    result = {}

    # TLB-F1 or DST-F1.
    grouping_cond = ['dialog_id', 'turn_id', 'referent']
    res = get_value_em_f1(ref_df, pred_df, grouping_cond)
    result.update(res)

    # R-F1.
    grouping_cond = ['dialog_id', 'turn_id']
    res = get_referent_or_slot_em_f1(ref_df, pred_df, grouping_cond, 'referent')
    res = {f'R-{k}': v for k, v in res.items()}
    result.update(res)

    ### S-F1
    grouping_cond = ['dialog_id', 'turn_id']
    res = get_referent_or_slot_em_f1(ref_df, pred_df, grouping_cond, 'slot')
    res = {f'S-{k}': v for k, v in res.items()}
    result.update(res)

    # RS-F1.
    grouping_cond = ['dialog_id', 'turn_id', 'referent']
    res = get_referent_or_slot_em_f1(ref_df, pred_df, grouping_cond, 'slot')
    res = {f'RS-{k}': v for k, v in res.items()}
    result.update(res)

    # SV-F1.
    grouping_cond = ['dialog_id', 'turn_id']
    res = get_value_em_f1(ref_df, pred_df, grouping_cond)
    res = {f'SV-{k}': v for k, v in res.items()}
    result.update(res)

    # TLB-F1 or DST-F1 (version 2)
    grouping_cond = ['dialog_id', 'turn_id', 'referent']
    res = get_value_em_f1(ref_df, pred_df, grouping_cond, avg_by='referent')
    res = {f'v2-{k}': v for k, v in res.items()}
    result.update(res)

    return result