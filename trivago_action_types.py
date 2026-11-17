"""Stable action vocabulary for the Trivago item-event adapter."""

ACTION_TO_ID = {
    "clickout item": 1,
    "interaction item rating": 2,
    "interaction item info": 3,
    "interaction item image": 4,
    "interaction item deals": 5,
    "search for item": 6,
    "change of sort order": 7,
    "filter selection": 8,
    "search for destination": 9,
    "search for poi": 10,
}
PAD_ACTION_ID = 0
UNK_ACTION_ID = 11
NUM_ACTIONS = 12
ITEM_ACTIONS = frozenset({
    "clickout item", "interaction item rating", "interaction item info",
    "interaction item image", "interaction item deals", "search for item",
})
ACTION_SCHEMA_VERSION = "trivago_item_event_actions_v1"
ACTION_FUSION = "concat_post_graph_gru_v1"
ACTION_ENCODER = "gru_chronological_zero_init_v1"

ACTION_INPUT_SCOPE = "six_item_reference_actions"
ACTION_HISTORY_POLICY = "train_clickouts_appended; val_test_clickouts_masked"

# Experimental extension: build a per-session graph over aligned item/action
# events, then read that graph separately for each clickout candidate.
EVENT_GRAPH_ENCODER = "item_action_session_graph_v1"
EVENT_GRAPH_FUSION = "candidate_conditioned_event_readout_v1"

# A separate vocabulary of (action, reference), not hotel IDs. Unknown
# references retain their action type using IDs 1..10; zero is padding.
ALL_EVENT_SCHEMA_VERSION = "trivago_all_event_pairs_v1"
ALL_EVENT_INPUT_SCOPE = "ten_action_reference_pairs"
ALL_EVENT_ENCODER = "gru_joint_pairs_chronological_zero_init_v1"
ALL_EVENT_FUSION = "gated_session_joint_event_gru_v1"
ALL_EVENT_HISTORY_POLICY = "train_clickouts_reference_appended; val_test_clickouts_type_only"


def event_token(vocabulary, action, reference, masked=False):
    action_id = ACTION_TO_ID[action]
    if masked:
        return action_id
    return vocabulary.get((action_id, (reference or "").strip()), action_id)


def all_event_schema(num_tokens, max_events):
    return {
        "action_to_id": ACTION_TO_ID,
        "action_schema_version": ALL_EVENT_SCHEMA_VERSION,
        "action_input_scope": ALL_EVENT_INPUT_SCOPE,
        "action_history_policy": ALL_EVENT_HISTORY_POLICY,
        "num_actions": NUM_ACTIONS,
        "pad_action_id": PAD_ACTION_ID,
        "unk_action_id": UNK_ACTION_ID,
        "num_event_tokens": num_tokens,
        "max_session_events": max_events,
        "event_vocabulary_policy": "train_sessions_only; joint_action_reference; per_action_unknown_1_to_10",
        "event_embedding_policy": "joint_pair_residual_plus_shared_action_type; unknown_reference_type_only",
    }
