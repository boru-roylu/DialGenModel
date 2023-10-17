import collections
import json


schema_path = "./metadata/schema.json"
with open(schema_path) as f:
    schema = json.load(f)

slot_sep = ' [s] '
value_sep = ' [v] '
referent_value_sep = ' [rv] '
slot_referent_value_sep = ' [srv] '
cross_referent_sep = ' [cv] '
value_op_sep = ' [vo] '

empty_state = '[empty state]'
prev_state = '[state]'

op_to_token = {
    'delete': '[delete]',
    'concat': '[concat]',
    'same': '[same]',
}

domain_names = [
    'AccidentDetails',
    'Adjuster', 
    'CarInfo',
    'ContactInfo',
    'DriverActions',
    'Evidences',
    'InjuryDetails',
    'TrafficEnvironment',
    'Trip']


domain_slot_to_is_categorical = {}
domain_name_to_slot_names = collections.defaultdict(list)
domain_slot_to_values = collections.defaultdict(list)
slot_to_values = collections.defaultdict(list)
for domain in schema:
    domain_name = domain['service_name']
    for slot in domain['slots']:
        domain_slot = slot["name"]
        _, slot_name = domain_slot.split('-')
        assert _ == domain_name
        domain_name_to_slot_names[domain_name].append(slot_name)
        domain_slot_to_is_categorical[domain_slot] = slot['is_categorical']
        if slot['is_categorical']:
            domain_slot_to_values[domain_slot] = slot["possible_values"] 
            slot_to_values[slot_name] = slot["possible_values"]


dummy_referent_slot_values = [
    {
        'referent': 'Global',
        'values': ['NONE']
    }
]


dummy_frames = []
for domain_name in domain_names:
    dummy_frames.append({
        'actions': None,
        'service': domain_name,
        "slots": None,
        'state': {
            "active_intent": None,
            "requested_slots": None,
            'slot_values':{},
        },
    })