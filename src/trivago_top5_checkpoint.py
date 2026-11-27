"""Checkpoint compatibility and RNG restoration for the Trivago trainer."""

import math
import hashlib
import json

METADATA_LOSS_REDUCTION = "valid_pair_mean_v1"
CANDIDATE_POLICY = "unique_item_first_impression_v1"
EARLY_STOPPING_VERSION = "val_ndcg_delta_or_ce_divergence_v3"
TRAINING_POLICY_VERSION = "ndcg_plateau_lossguard_v2"


def prepared_data_signature(config):
    """Return the versioned identity used by trainer and notebook restore."""
    price_config_hash = hashlib.sha256(json.dumps(
        config.get('price_features'), ensure_ascii=False, sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    return (
        config.get('converter_version'), config.get('data_schema_version'),
        config.get('action_schema_version'), config.get('action_schema_sha256'),
        config['item_id_map_sha256'], config['metadata_schema_sha256'],
        config['source_train_sha256'], config['source_metadata_sha256'],
        tuple(sorted(config['artifact_sha256'].items())),
        tuple(config['field_names']), price_config_hash,
    )


def checkpoint_training_policy(args):
    return dict(version=TRAINING_POLICY_VERSION, scheduler='ReduceLROnPlateau',
                mode='max', threshold_mode='abs', min_delta=args.min_delta,
                lr_factor=args.lr_factor, lr_patience=args.lr_patience,
                min_lr=args.min_lr, cooldown=0,
                patience=args.patience, loss_patience=args.loss_patience,
                loss_min_delta=args.loss_min_delta)


def new_early_stopping(patience=3, loss_patience=3, min_delta=1e-4,
                       loss_min_delta=1e-3, ndcg_anchor=None):
    return dict(version=EARLY_STOPPING_VERSION, patience=patience,
                loss_patience=loss_patience, min_delta=min_delta,
                loss_min_delta=loss_min_delta,
                ndcg_anchor=ndcg_anchor, ndcg_bad_epochs=0,
                loss_bad_epochs=0, previous_train_ce=None, previous_val_ce=None,
                stopped=False, reason=None, triggered=[])


def update_early_stopping(state, ndcg, train_ce, val_ce):
    """OR stopping; online train CE vs eval validation CE is a heuristic.

    The NDCG anchor moves only for improvements exceeding min_delta. Small
    gains can accumulate against this anchor; checkpoint best is independent.
    Loss divergence requires BOTH train CE falling and validation CE rising by
    at least loss_min_delta relative to the previous full epoch.
    Missing/incomplete legacy train losses break the consecutive loss streak.
    """
    if not all(math.isfinite(v) for v in (ndcg, val_ce) if v is not None) or (
            train_ce is not None and not math.isfinite(train_ce)):
        raise FloatingPointError('Non-finite monitoring value; stopping state was not updated')
    anchor = state['ndcg_anchor']
    improved = anchor is None or ndcg > anchor + state['min_delta']
    if improved:
        state['ndcg_anchor'] = ndcg
    state['ndcg_bad_epochs'] = 0 if improved else state['ndcg_bad_epochs'] + 1
    previous_train, previous_val = state['previous_train_ce'], state['previous_val_ce']
    if train_ce is not None and previous_train is not None and previous_val is not None:
        train_drop = (previous_train - train_ce) / max(abs(previous_train), 1e-12)
        val_rise = (val_ce - previous_val) / max(abs(previous_val), 1e-12)
        # Subtraction/division may put an exact decimal boundary a few double
        # precision rounding units below the threshold (e.g. 1 -> 1.001).
        delta = state['loss_min_delta']
        divergent = (train_ce < previous_train and val_ce > previous_val
                     and (train_drop >= delta or math.isclose(
                         train_drop, delta, rel_tol=0., abs_tol=1e-15))
                     and (val_rise >= delta or math.isclose(
                         val_rise, delta, rel_tol=0., abs_tol=1e-15)))
    else:
        divergent = False
    state['loss_bad_epochs'] = state['loss_bad_epochs'] + 1 if divergent else 0
    state['previous_train_ce'], state['previous_val_ce'] = train_ce, val_ce
    state['triggered'] = []
    if state['ndcg_bad_epochs'] >= state['patience']:
        state['triggered'].append('val_ndcg_no_improvement')
    if state['loss_bad_epochs'] >= state['loss_patience']:
        state['triggered'].append('ce_loss_divergence')
    state['stopped'] = bool(state['triggered'])
    state['reason'] = state['triggered'][0] if state['triggered'] else None
    return state['stopped']

MODEL_ARGUMENT_NAMES = (
    "emb_size", "batch_size", "sample_order", "metadata_projection",
    "num_heads", "beta", "alpha", "si_dc", "action_mode", "num_actions",
    "action_encoder", "action_fusion", "price_mode",
)


def best_validation_ndcg(checkpoint: dict) -> float:
    """Never compare an old MRR-selected best score against NDCG@5."""
    if checkpoint.get("selection_metric") != "ndcg_at_5":
        return -1.0
    return float(checkpoint.get("best_ndcg_at_5", -1.0))


def checkpoint_model_arguments(args) -> dict:
    values = {name: getattr(args, name, 'off' if name == 'price_mode' else None)
              for name in MODEL_ARGUMENT_NAMES}
    values['metadata_loss_reduction'] = METADATA_LOSS_REDUCTION
    values['candidate_policy'] = CANDIDATE_POLICY
    if args.action_mode == 'all_event':
        values['num_event_tokens'] = args.num_event_tokens
    return values


def validate_checkpoint(checkpoint: dict, signature, expected_arguments: dict) -> None:
    if checkpoint.get("signature") != signature:
        raise ValueError("Checkpoint does not match prepared data")
    saved_arguments = checkpoint.get("model_arguments")
    if not isinstance(saved_arguments, dict):
        raise ValueError(
            "Checkpoint lacks complete model_arguments (older trainer). "
            "The original attention/loss/graph settings cannot be verified; "
            "use a new run or migrate only with the verified original settings."
        )
    mismatches = [
        name for name, expected in expected_arguments.items()
        if name not in saved_arguments or saved_arguments[name] != expected
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint model arguments differ or are missing: "
            + ", ".join(mismatches)
            + ". Use a separate output directory for a changed model or objective."
        )


def restore_checkpoint_rng(checkpoint: dict, device, torch_module) -> None:
    # torch.load(map_location='cuda') also moves the saved RNG ByteTensors.
    # CPU and CUDA RNG setters receive states on CPU.
    torch_module.set_rng_state(checkpoint["cpu_rng"].cpu())
    cuda_rng = checkpoint.get("cuda_rng")
    if device.type == "cuda" and cuda_rng is not None:
        torch_module.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
