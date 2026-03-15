#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for victory prediction models.

Supports three tuning modes (--mode):
    params     Tune model hyperparameters only (default). Uses model's DEFAULT_FEATURES.
    variables  Tune feature variants only. Uses model's default hyperparameters.
    both       Tune features and hyperparameters simultaneously (legacy behavior).

Usage:
    python tune_model.py --model grouped_mlp --n-trials 100
    python tune_model.py --model grouped_mlp --n-trials 100 --mode variables
    python tune_model.py --model grouped_mlp --n-trials 100 --mode both
    python tune_model.py --model all --n-trials 50
    python tune_model.py --model xgboost --n-trials 100 --resample undersample
"""

import argparse
import gc
import multiprocessing
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Literal

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

import numpy as np
from utils.model_evaluator import evaluate_fold
from utils.model_registry import MODEL_REGISTRY
from utils.data_utils import load_and_prepare_base_data, load_and_prepare_data, FEATURE_GROUPS


# ============================================================================
# Feature variant tuning
# ============================================================================

# Feature families: for each family, Optuna picks one variant (or 'none' to exclude)
FEATURE_FAMILIES = {
    # Per-turn rates: 3 variants (adj, share, raw_share)
    'science':    {'adj': 'science_adj',    'share': 'science_share',    'raw_share': 'science_raw_share'},
    'culture':    {'adj': 'culture_adj',    'share': 'culture_share',    'raw_share': 'culture_raw_share'},
    'tourism':    {'adj': 'tourism_adj',    'share': 'tourism_share',    'raw_share': 'tourism_raw_share'},
    'gold':       {'adj': 'gold_adj',       'share': 'gold_share',       'raw_share': 'gold_raw_share'},
    'faith':      {'adj': 'faith_adj',      'share': 'faith_share',      'raw_share': 'faith_raw_share'},
    'production': {'adj': 'production_adj', 'share': 'production_share', 'raw_share': 'production_raw_share'},
    'food':       {'adj': 'food_adj',       'share': 'food_share',       'raw_share': 'food_raw_share'},
    # Military: 2 variants
    'military':   {'adj': 'military_adj', 'share': 'military_share'},
    # Counts: 2 variants each
    'cities':       {'none': '', 'raw': 'cities',       'share': 'cities_share'},
    'population':   {'raw': 'population',   'share': 'population_share'},
    'votes':        {'raw': 'votes',        'share': 'votes_share'},
    'minor_allies': {'raw': 'minor_allies', 'share': 'minor_allies_share'},
}

# Always included (derived from FEATURE_GROUPS to stay in sync with data_utils)
FIXED_FEATURES = FEATURE_GROUPS['progress'] + FEATURE_GROUPS['gaps']

# Toggleable features (on/off)
TOGGLE_FEATURES = FEATURE_GROUPS['percentages']


def suggest_feature_variants(trial: 'optuna.Trial') -> list:
    """Let Optuna select one variant per feature family and toggle optional features.

    Returns list of column names to use as include_features.
    """
    selected = list(FIXED_FEATURES)

    # Pick one variant per family (or 'none' to exclude)
    for family_name, variants in FEATURE_FAMILIES.items():
        variant_names = list(variants.keys())
        chosen = trial.suggest_categorical(f'feat_{family_name}', variant_names)
        if chosen != 'none':
            selected.append(variants[chosen])

    # Toggle features on/off
    for feat_name in TOGGLE_FEATURES:
        if trial.suggest_categorical(f'feat_{feat_name}', [True, False]):
            selected.append(feat_name)

    return selected


def reconstruct_include_features(raw_params: dict) -> list:
    """Reconstruct include_features from feat_* trial params stored by Optuna."""
    selected = list(FIXED_FEATURES)

    for family_name, variants in FEATURE_FAMILIES.items():
        key = f'feat_{family_name}'
        chosen = raw_params.get(key, 'share')  # default to share
        if chosen != 'none':
            selected.append(variants[chosen])

    for feat_name in TOGGLE_FEATURES:
        key = f'feat_{feat_name}'
        if raw_params.get(key, True):
            selected.append(feat_name)

    return selected


# ============================================================================
# Search space definitions
# ============================================================================

def suggest_xgboost_params(trial: 'optuna.Trial') -> Dict:
    """Define XGBoost hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-1, 20.0, log=True),
    }

    calibrate = trial.suggest_categorical('calibrate', [True, False])
    params['calibrate'] = calibrate
    if calibrate:
        params['calibration_method'] = trial.suggest_categorical(
            'calibration_method', ['isotonic', 'sigmoid']
        )
    else:
        params['calibration_method'] = 'sigmoid'  # unused but needs a value

    return params


def suggest_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define MLP hyperparameter search space (PyTorch, GPU-enabled).

    Uses constant-width layers (same architecture as grouped MLP).
    """
    n_layers = trial.suggest_int('n_layers', 1, 32)
    layer_size = trial.suggest_int('layer_size', 16, 256)

    # Constant width for all layers (residual connections require matching dims)
    layer_sizes = tuple([layer_size] * n_layers) if n_layers > 0 else ()

    params = {
        'layer_sizes': layer_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        'loss_tp_alpha': trial.suggest_float('loss_tp_alpha', 0.5, 2.0),
    }

    return params


def suggest_grouped_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define grouped MLP hyperparameter search space.

    Uses constant-width layers (required for residual skip connections).
    Supports up to 10 layers with the residual _UtilityNet architecture.
    """
    n_layers = trial.suggest_int('n_layers', 1, 8)
    layer_size = trial.suggest_int('layer_size', 32, 256)

    # Constant width for all layers (residual connections require matching dims)
    layer_sizes = tuple([layer_size] * n_layers) if n_layers > 0 else ()

    params = {
        'layer_sizes': layer_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        #'batch_size_groups': trial.suggest_categorical(
        #    'batch_size_groups', [1024, 2048, 4096]
        #),
        'loss_tp_alpha': trial.suggest_float('loss_tp_alpha', 0.5, 2.0),
    }

    return params


def suggest_interaction_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define interaction MLP (DeepSets) hyperparameter search space.

    Separate encoder and decoder architectures with constant-width residual blocks.
    """
    n_encoder_layers = trial.suggest_int('n_encoder_layers', 1, 16)
    encoder_size = trial.suggest_int('encoder_size', 16, 256)
    n_decoder_layers = trial.suggest_int('n_decoder_layers', 1, 16)
    decoder_size = trial.suggest_int('decoder_size', 16, 256)

    params = {
        'encoder_sizes': tuple([encoder_size] * n_encoder_layers),
        'decoder_sizes': tuple([decoder_size] * n_decoder_layers),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        'loss_tp_alpha': trial.suggest_float('loss_tp_alpha', 0.5, 2.0),
    }

    return params


def suggest_attention_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define attention MLP hyperparameter search space.

    Encoder-attention-decoder architecture with multi-head self-attention.
    """
    n_encoder_layers = trial.suggest_int('n_encoder_layers', 1, 8)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    # embed_dim must be divisible by num_heads; sample a multiplier instead
    encoder_mult = trial.suggest_int('encoder_mult', 2, 256 // num_heads)
    encoder_size = encoder_mult * num_heads
    n_decoder_layers = trial.suggest_int('n_decoder_layers', 1, 8)
    decoder_size = trial.suggest_int('decoder_size', 16, 256)

    params = {
        'encoder_sizes': tuple([encoder_size] * n_encoder_layers),
        'decoder_sizes': tuple([decoder_size] * n_decoder_layers),
        'num_heads': num_heads,
        'n_attn_layers': trial.suggest_int('n_attn_layers', 1, 4),
        'attn_dropout': trial.suggest_float('attn_dropout', 0.0, 0.3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        'loss_tp_alpha': trial.suggest_float('loss_tp_alpha', 0.5, 2.0),
    }

    return params


SEARCH_SPACES = {
    'xgboost': suggest_xgboost_params,
    'mlp': suggest_mlp_params,
    'grouped_mlp': suggest_grouped_mlp_params,
    'interaction_mlp': suggest_interaction_mlp_params,
    'attention_mlp': suggest_attention_mlp_params,
}

# Per-model __init__ parameter metadata: (param_name, type_annotation, formatter)
# formatter: 'g6' = 6 sig figs, 'tuple_repeat' = compact (x,)*n, None = repr()
PARAM_SIGNATURES = {
    'xgboost': [
        ('n_estimators',       'int',   None),
        ('max_depth',          'int',   None),
        ('learning_rate',      'float', 'g6'),
        ('subsample',          'float', 'g6'),
        ('colsample_bytree',   'float', 'g6'),
        ('min_child_weight',   'int',   None),
        ('gamma',              'float', 'g6'),
        ('reg_alpha',          'float', 'g6'),
        ('reg_lambda',         'float', 'g6'),
        ('calibrate',          'bool',  None),
        ('calibration_method', 'str',   None),
    ],
    'mlp': [
        ('layer_sizes',  'tuple', 'tuple_repeat'),
        ('dropout',      'float', 'g6'),
        ('lr',           'float', 'g6'),
        ('weight_decay', 'float', 'g6'),
        ('epochs',       'int',   None),
        ('batch_size',   'int',   None),
        ('loss_tp_alpha','float', 'g6'),
    ],
    'grouped_mlp': [
        ('layer_sizes',       'Tuple[int, ...]', 'tuple_repeat'),
        ('dropout',           'float',           'g6'),
        ('lr',                'float',           'g6'),
        ('weight_decay',      'float',           'g6'),
        ('epochs',            'int',             None),
        ('batch_size_groups', 'int',             None),
        ('loss_tp_alpha',     'float',           'g6'),
    ],
    'interaction_mlp': [
        ('encoder_sizes',     'Tuple[int, ...]', 'tuple_repeat'),
        ('decoder_sizes',     'Tuple[int, ...]', 'tuple_repeat'),
        ('dropout',           'float',           'g6'),
        ('lr',                'float',           'g6'),
        ('weight_decay',      'float',           'g6'),
        ('epochs',            'int',             None),
        ('batch_size_groups', 'int',             None),
        ('loss_tp_alpha',     'float',           'g6'),
    ],
    'attention_mlp': [
        ('encoder_sizes',     'Tuple[int, ...]', 'tuple_repeat'),
        ('decoder_sizes',     'Tuple[int, ...]', 'tuple_repeat'),
        ('num_heads',         'int',             None),
        ('n_attn_layers',     'int',             None),
        ('attn_dropout',      'float',           'g6'),
        ('dropout',           'float',           'g6'),
        ('lr',                'float',           'g6'),
        ('weight_decay',      'float',           'g6'),
        ('epochs',            'int',             None),
        ('batch_size_groups', 'int',             None),
        ('loss_tp_alpha',     'float',           'g6'),
    ],
}


def _format_value(value, formatter):
    """Format a single parameter value for code output."""
    if formatter == 'tuple_repeat':
        if isinstance(value, (tuple, list)) and len(value) > 1 and len(set(value)) == 1:
            return f"({value[0]},) * {len(value)}"
        return repr(value)
    elif formatter == 'g6':
        return f"{value:.6g}"
    elif isinstance(value, str):
        return repr(value)
    else:
        return repr(value)


def generate_init_snippet(model_name: str, converted_params: dict) -> str:
    """Generate a Python code snippet for pasting into a model's __init__ defaults."""
    if model_name not in PARAM_SIGNATURES:
        return ""

    lines = []

    # Handle include_features -> DEFAULT_FEATURES class attribute
    if 'include_features' in converted_params:
        features = converted_params['include_features']
        lines.append("DEFAULT_FEATURES = [")
        for f in features:
            lines.append(f"    '{f}',")
        lines.append("]")
        lines.append("")  # blank separator

    # Handle __init__ params
    for param_name, type_ann, formatter in PARAM_SIGNATURES[model_name]:
        if param_name not in converted_params:
            continue
        value = converted_params[param_name]
        formatted = _format_value(value, formatter)
        lines.append(f"{param_name}: {type_ann} = {formatted},")

    return "\n".join(lines)


def convert_best_params(model_name: str, best_params: Dict) -> Dict:
    """Convert raw Optuna best_params to model kwargs.

    Optuna stores intermediate trial parameters (e.g. n_layers, layer_size,
    feat_* variant choices) that the suggest functions convert into derived
    parameters. This function applies the same conversion so best_params
    can be passed directly to model constructors.
    """
    params = dict(best_params)

    # Reconstruct include_features from feat_* params
    if any(k.startswith('feat_') for k in params):
        params['include_features'] = reconstruct_include_features(params)
        for k in [k for k in params if k.startswith('feat_')]:
            del params[k]

    if model_name in ('mlp', 'grouped_mlp'):
        n_layers = params.pop('n_layers', None)
        layer_size = params.pop('layer_size', None)

        if n_layers is not None and layer_size is not None:
            # Both MLP and grouped MLP use constant-width residual architecture
            sizes = tuple([layer_size] * n_layers) if n_layers > 0 else ()
            params['layer_sizes'] = sizes

    if model_name in ('interaction_mlp', 'attention_mlp'):
        n_enc = params.pop('n_encoder_layers', None)
        enc_size = params.pop('encoder_size', None)
        enc_mult = params.pop('encoder_mult', None)
        num_heads = params.get('num_heads', None)
        n_dec = params.pop('n_decoder_layers', None)
        dec_size = params.pop('decoder_size', None)

        # attention_mlp stores encoder_mult instead of encoder_size
        if enc_size is None and enc_mult is not None and num_heads is not None:
            enc_size = enc_mult * num_heads

        if n_enc is not None and enc_size is not None:
            params['encoder_sizes'] = tuple([enc_size] * n_enc) if n_enc > 0 else ()
        if n_dec is not None and dec_size is not None:
            params['decoder_sizes'] = tuple([dec_size] * n_dec) if n_dec > 0 else ()

    return params


# ============================================================================
# Objective function
# ============================================================================

def create_objective(
    model_name: str,
    metric: str = 'log_loss',
    random_state: int = 42,
    resample_method: Optional[str] = None,
    precomputed_data=None,
    mode: str = 'params',
):
    """Create an Optuna objective function for a given model.

    Args:
        mode: 'params' (tune hyperparams only), 'variables' (tune features only),
              or 'both' (tune both simultaneously).
    """

    model_class = MODEL_REGISTRY[model_name]
    suggest_fn = SEARCH_SPACES[model_name]

    # Metrics where lower is better
    minimize_metrics = {'brier_score', 'log_loss'}

    def objective(trial: 'optuna.Trial') -> float:
        if mode == 'variables':
            # Only tune feature variants; use model defaults for hyperparams
            params = {'include_features': suggest_feature_variants(trial)}
        elif mode == 'params':
            # Only tune hyperparams; use model's DEFAULT_FEATURES
            params = suggest_fn(trial)
        else:  # 'both'
            params = suggest_fn(trial)
            params['include_features'] = suggest_feature_variants(trial)

        # Unpack precomputed data for per-fold iteration
        df, X, y, cv_splits = precomputed_data

        fold_penalized_values = []
        fold_raw_values = []
        fold_gaps = []
        resample_skip_warned = set()

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                clusters_train = df.iloc[train_idx]['game_id']

                # Instantiate a fresh model for each fold
                model = model_class(random_state=random_state, **params)

                fold_metrics = evaluate_fold(
                    model, X_train, y_train, X_val, y_val,
                    clusters_train=clusters_train,
                    resample_method=resample_method,
                    random_state=random_state,
                    _resample_skip_warned=resample_skip_warned,
                )
            except Exception as e:
                print(f"  Trial {trial.number} failed on fold {fold_idx}: {e}")
                if metric in minimize_metrics:
                    return float('inf')
                else:
                    return 0.0

            fold_val = fold_metrics[metric]
            fold_train = fold_metrics.get(f'train_{metric}', None)
            fold_raw_values.append(fold_val)

            # Apply per-fold overfitting penalty before reporting to pruner
            if fold_train is not None:
                if metric in minimize_metrics:
                    gap = fold_val - fold_train
                else:
                    gap = fold_train - fold_val
                fold_gaps.append(gap)
                penalty = gap**2 * 10
                if metric in minimize_metrics:
                    fold_val += penalty
                else:
                    fold_val -= penalty

            fold_penalized_values.append(fold_val)

            # Report penalized running mean to Optuna for fold-level pruning
            running_mean = np.mean(fold_penalized_values)
            trial.report(running_mean, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # All folds completed
        value = np.mean(fold_penalized_values)
        raw_value = np.mean(fold_raw_values)

        # Log results
        if fold_gaps:
            gap = np.mean(fold_gaps)
            train_value = raw_value - gap if metric in minimize_metrics else raw_value + gap
            if gap > 0.05:
                print(f"  Trial {trial.number}: {metric} = {raw_value:.6f} ⚠ OVERFITTING (gap={gap:.4f})  "
                      f"(params: {_format_params(params)})")
            else:
                print(f"  Trial {trial.number}: {metric} = {raw_value:.6f} (train={train_value:.6f}, gap={gap:+.4f})  "
                      f"(params: {_format_params(params)})")
        else:
            print(f"  Trial {trial.number}: {metric} = {raw_value:.6f}  "
                  f"(params: {_format_params(params)})")

        # Free stale model/optimizer references between trials
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return value

    return objective, metric in minimize_metrics


def _format_params(params: Dict) -> str:
    """Format parameters for concise display."""
    skip = {'exclude_features', 'include_features', 'calibration_method', 'validation_fraction'}
    parts = []
    for k, v in params.items():
        if k in skip:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, tuple):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ', '.join(parts)


# ============================================================================
# Main
# ============================================================================

def _mp_worker(
    model_name: str,
    n_trials: int,
    csv_path: str,
    metric: str,
    n_splits: int,
    random_state: int,
    resample_method: Optional[str],
    storage: str,
    full_data: bool,
    mode: str,
    study_name: str,
    result_file: str,
):
    """Worker process for multi-process Optuna tuning.

    Each worker independently loads data, creates its objective, and runs
    a portion of the trials. Coordination happens via Optuna's shared storage.
    """
    # Each process must re-import and set up its own state
    model_class = MODEL_REGISTRY[model_name]

    use_variants = (
        mode in ('variables', 'both')
        and model_name in ('xgboost', 'mlp', 'grouped_mlp', 'interaction_mlp', 'attention_mlp')
    )
    preloaded_df = load_and_prepare_base_data(csv_path, keep_variants=use_variants)

    needs_full_data = (
        (callable(model_class) and not isinstance(model_class, type)) or
        (hasattr(model_class, 'NEEDS_FULL_DATA') and model_class.NEEDS_FULL_DATA)
    )
    if full_data:
        phase_filter = None
    elif needs_full_data:
        phase_filter = (1, [0.5])
    else:
        phase_filter = (1, [0.8])

    precomputed_data = load_and_prepare_data(
        csv_path, n_splits=n_splits, random_state=random_state,
        phase_filter=phase_filter, preloaded_df=preloaded_df,
        use_variant_columns=use_variants,
    )

    objective, minimize = create_objective(
        model_name, metric, random_state, resample_method,
        precomputed_data=precomputed_data,
        mode=mode,
    )

    study = optuna.load_study(study_name=study_name, storage=storage)

    def save_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            converted = convert_best_params(model_name, study.best_params)
            code_snippet = generate_init_snippet(model_name, converted)
            features = converted.get('include_features')
            if features is None and hasattr(model_class, 'DEFAULT_FEATURES') and model_class.DEFAULT_FEATURES is not None:
                features = list(model_class.DEFAULT_FEATURES)
            direction = 'minimize' if minimize else 'maximize'
            best = {
                'model': model_name,
                'mode': mode,
                'metric': metric,
                'direction': direction,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'best_trial': trial.number,
                'n_trials_so_far': len(study.trials),
                'resample_method': resample_method,
                'features': features,
                'generated_code': code_snippet,
            }
            with open(result_file, 'w') as f:
                json.dump(best, f, indent=2, default=str)
            print(f"  ★ New best! Trial {trial.number}: {metric} = {study.best_value:.6f} → saved to {result_file}")

    study.optimize(objective, n_jobs=1, n_trials=n_trials,
                   callbacks=[save_best_callback])


def tune_model(
    model_name: str,
    n_trials: int = 100,
    csv_path: str = '../turn_data.csv',
    metric: str = 'log_loss',
    n_splits: int = 5,
    random_state: int = 42,
    resample_method: Optional[str] = None,
    storage: Optional[str] = None,
    full_data: bool = False,
    n_jobs: int = 1,
    mode: str = 'params',
) -> 'optuna.Study':
    """Run Optuna hyperparameter tuning for a single model.

    Args:
        mode: 'params' (tune hyperparams only, default), 'variables' (tune
              feature variants only), or 'both' (tune both simultaneously).
    """

    if not HAS_OPTUNA:
        print("Error: optuna is required. Install with: pip install optuna", file=sys.stderr)
        sys.exit(1)

    if model_name not in SEARCH_SPACES:
        print(f"Error: No search space defined for '{model_name}'. "
              f"Available: {', '.join(SEARCH_SPACES.keys())}", file=sys.stderr)
        sys.exit(1)

    model_class = MODEL_REGISTRY[model_name]

    # Metrics where lower is better
    minimize_metrics = {'brier_score', 'log_loss'}
    direction = 'minimize' if metric in minimize_metrics else 'maximize'
    study_name = f"tune_{model_name}_{mode}"
    if resample_method:
        study_name += f"_{resample_method}"

    # Output file for best params
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    if mode == 'params':
        result_file = output_dir / f"best_{model_name}_params.json"
    else:
        result_file = output_dir / f"best_{model_name}_{mode}_params.json"

    # For multi-process, require storage for cross-process coordination
    if n_jobs > 1 and storage is None:
        storage = f"sqlite:///output/tuning_{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )

    print("=" * 80)
    print(f"TUNING: {model_name.upper()} ({mode} mode)")
    print("=" * 80)
    print(f"Mode:       {mode}")
    print(f"Metric:     {metric} ({direction})")
    print(f"Trials:     {n_trials}")
    print(f"CV Splits:  {n_splits}")
    print(f"Resample:   {resample_method or 'none'}")
    print(f"Full Data:  {full_data}")
    print(f"Workers:    {n_jobs}")
    if storage:
        print(f"Storage:    {storage}")
    print("=" * 80)

    if n_jobs > 1:
        # Multi-process: use fork on Linux (required for Colab/notebooks where
        # spawn hangs trying to re-import __main__), spawn on Windows.
        mp_method = 'fork' if sys.platform != 'win32' else 'spawn'
        ctx = multiprocessing.get_context(mp_method)
        trials_per_worker = n_trials // n_jobs
        remainder = n_trials % n_jobs

        processes = []
        for i in range(n_jobs):
            worker_trials = trials_per_worker + (1 if i < remainder else 0)
            p = ctx.Process(
                target=_mp_worker,
                kwargs=dict(
                    model_name=model_name,
                    n_trials=worker_trials,
                    csv_path=csv_path,
                    metric=metric,
                    n_splits=n_splits,
                    random_state=random_state,
                    resample_method=resample_method,
                    storage=storage,
                    full_data=full_data,
                    mode=mode,
                    study_name=study_name,
                    result_file=str(result_file),
                ),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Reload study to get all results from workers
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        # Single-process: original code path
        # Preload data and precompute k-fold splits once
        use_variants = (
            mode in ('variables', 'both')
            and model_name in ('xgboost', 'mlp', 'grouped_mlp', 'interaction_mlp', 'attention_mlp')
        )
        preloaded_df = load_and_prepare_base_data(csv_path, keep_variants=use_variants)

        needs_full_data = (
            (callable(model_class) and not isinstance(model_class, type)) or
            (hasattr(model_class, 'NEEDS_FULL_DATA') and model_class.NEEDS_FULL_DATA)
        )
        if full_data:
            phase_filter = None
        elif needs_full_data:
            phase_filter = (1, [0.5])
        else:
            phase_filter = (1, [0.8])

        precomputed_data = load_and_prepare_data(
            csv_path, n_splits=n_splits, random_state=random_state,
            phase_filter=phase_filter, preloaded_df=preloaded_df,
            use_variant_columns=use_variants,
        )

        objective, _ = create_objective(
            model_name, metric, random_state, resample_method,
            precomputed_data=precomputed_data,
            mode=mode,
        )

        def save_best_callback(study, trial):
            if study.best_trial.number == trial.number:
                converted = convert_best_params(model_name, study.best_params)
                code_snippet = generate_init_snippet(model_name, converted)
                features = converted.get('include_features')
                if features is None and hasattr(model_class, 'DEFAULT_FEATURES') and model_class.DEFAULT_FEATURES is not None:
                    features = list(model_class.DEFAULT_FEATURES)
                best = {
                    'model': model_name,
                    'mode': mode,
                    'metric': metric,
                    'direction': direction,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'best_trial': trial.number,
                    'n_trials_so_far': len(study.trials),
                    'resample_method': resample_method,
                    'features': features,
                    'generated_code': code_snippet,
                }
                with open(result_file, 'w') as f:
                    json.dump(best, f, indent=2, default=str)
                print(f"  ★ New best! Trial {trial.number}: {metric} = {study.best_value:.6f} → saved to {result_file}")

        study.optimize(objective, n_jobs=1, n_trials=n_trials, show_progress_bar=True,
                       callbacks=[save_best_callback])

    # Print best result summary
    print("\n" + "=" * 80)
    print(f"BEST RESULT: {model_name.upper()}")
    print("=" * 80)
    print(f"Best {metric}: {study.best_value:.6f}")
    print(f"\nBest params:")
    best_model_kwargs = convert_best_params(model_name, study.best_params)
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

    # Save final best params to JSON
    code_snippet = generate_init_snippet(model_name, best_model_kwargs)
    features = best_model_kwargs.get('include_features')
    if features is None and hasattr(model_class, 'DEFAULT_FEATURES') and model_class.DEFAULT_FEATURES is not None:
        features = list(model_class.DEFAULT_FEATURES)

    result = {
        'model': model_name,
        'mode': mode,
        'metric': metric,
        'direction': direction,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'resample_method': resample_method,
        'features': features,
        'generated_code': code_snippet,
    }
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nBest params saved to: {result_file}")

    return study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna for victory prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune_model.py --model grouped_mlp --n-trials 100                  # params mode (default)
  python tune_model.py --model grouped_mlp --n-trials 100 --mode variables # feature variants only
  python tune_model.py --model grouped_mlp --n-trials 100 --mode both      # legacy: both at once
  python tune_model.py --model all --n-trials 50
        """
    )

    parser.add_argument(
        '--model', type=str, required=True,
        help=f"Model to tune: {', '.join(list(SEARCH_SPACES.keys()) + ['all'])}"
    )
    parser.add_argument(
        '--n-trials', type=int, default=1000,
        help="Number of Optuna trials (default: 1000)"
    )
    parser.add_argument(
        '--metric', type=str, default='brier_score',
        choices=['brier_score', 'log_loss', 'roc_auc', 'balanced_accuracy'],
        help="Metric to optimize (default: log_loss)"
    )
    parser.add_argument(
        '--data', type=str, default='../turn_data.csv',
        help="Path to turn data CSV (default: ../turn_data.csv)"
    )
    parser.add_argument(
        '--n-splits', type=int, default=5,
        help="Number of CV splits (default: 5)"
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--resample', type=str, default='none',
        choices=['none', 'oversample', 'undersample', 'combined'],
        help="Resampling method (default: none)"
    )
    parser.add_argument(
        '--storage', type=str, default=None,
        help="Optuna storage URL for resumability (e.g., sqlite:///tuning.db)"
    )
    parser.add_argument(
        '--full-data',
        action='store_true',
        help="Use all turn data (no phase filtering)"
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help="Number of parallel Optuna jobs (default: 1)"
    )
    parser.add_argument(
        '--mode', type=str, default='params',
        choices=['params', 'variables', 'both'],
        help="Tuning mode: 'params' (hyperparams only, default), "
             "'variables' (feature variants only), 'both' (simultaneous)"
    )

    args = parser.parse_args()

    resample_method = None if args.resample == 'none' else args.resample

    models_to_tune = list(SEARCH_SPACES.keys()) if args.model == 'all' else [args.model]

    for model_name in models_to_tune:
        tune_model(
            model_name=model_name,
            n_trials=args.n_trials,
            csv_path=args.data,
            metric=args.metric,
            n_splits=args.n_splits,
            random_state=args.random_state,
            resample_method=resample_method,
            storage=args.storage,
            full_data=args.full_data,
            n_jobs=args.n_jobs,
            mode=args.mode,
        )
        print()


if __name__ == '__main__':
    main()
