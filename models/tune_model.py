#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for victory prediction models.

Usage:
    python tune_models.py --model xgboost --n-trials 100
    python tune_models.py --model random_forest --n-trials 100 --metric brier_score
    python tune_models.py --model mlp --n-trials 100
    python tune_models.py --model all --n-trials 50
    python tune_models.py --model xgboost --n-trials 100 --resample undersample
"""

import argparse
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
from utils.model_evaluator import run_kfold_evaluation, evaluate_fold
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
    'cities':       {'raw': 'cities',       'share': 'cities_share'},
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
        variant_names = list(variants.keys()) # + ['none']
        chosen = trial.suggest_categorical(f'feat_{family_name}', variant_names)
        # if chosen != 'none':
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
        # if chosen != 'none':
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

    params['include_features'] = suggest_feature_variants(trial)

    return params


def suggest_random_forest_params(trial: 'optuna.Trial') -> Dict:
    """Define Random Forest hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 100),  # Changed from 2 to 10 to prevent overfitting
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),      # Changed from 1 to 5 to prevent overfitting
        'max_features': trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', None]
        ),
        'class_weight': trial.suggest_categorical(
            'class_weight', ['balanced', 'balanced_subsample', None]
        ),
    }

    calibrate = trial.suggest_categorical('calibrate', [True, False])
    params['calibrate'] = calibrate
    if calibrate:
        params['calibration_method'] = trial.suggest_categorical(
            'calibration_method', ['isotonic', 'sigmoid']
        )
    else:
        params['calibration_method'] = 'sigmoid'

    exclude_turn = trial.suggest_categorical('exclude_turn_progress', [True, False])
    if exclude_turn:
        params['exclude_features'] = ['turn_progress']

    return params


def suggest_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define MLP hyperparameter search space (PyTorch, GPU-enabled).

    Uses constant-width layers (same architecture as grouped MLP).
    """
    n_layers = trial.suggest_int('n_layers', 1, 16)
    layer_size = trial.suggest_int('layer_size', 16, 256)

    # Constant width for all layers (residual connections require matching dims)
    layer_sizes = tuple([layer_size] * n_layers) if n_layers > 0 else ()

    params = {
        'layer_sizes': layer_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        'batch_size': trial.suggest_categorical(
            'batch_size', [4096, 8192, 16384]
        ),
    }

    params['include_features'] = suggest_feature_variants(trial)

    return params


def suggest_grouped_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define grouped MLP hyperparameter search space.

    Uses constant-width layers (required for residual skip connections).
    Supports up to 10 layers with the residual _UtilityNet architecture.
    """
    n_layers = trial.suggest_int('n_layers', 1, 16)
    layer_size = trial.suggest_int('layer_size', 32, 256)

    # Constant width for all layers (residual connections require matching dims)
    layer_sizes = tuple([layer_size] * n_layers) if n_layers > 0 else ()

    params = {
        'layer_sizes': layer_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 5, 30),
        'batch_size_groups': trial.suggest_categorical(
            'batch_size_groups', [1024, 2048, 4096]
        ),
    }

    params['include_features'] = suggest_feature_variants(trial)

    return params


SEARCH_SPACES = {
    'xgboost': suggest_xgboost_params,
    'random_forest': suggest_random_forest_params,
    'mlp': suggest_mlp_params,
    'grouped_mlp': suggest_grouped_mlp_params,
}


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
):
    """Create an Optuna objective function for a given model."""

    model_class = MODEL_REGISTRY[model_name]
    suggest_fn = SEARCH_SPACES[model_name]

    # Metrics where lower is better
    minimize_metrics = {'brier_score', 'log_loss'}

    def objective(trial: 'optuna.Trial') -> float:
        params = suggest_fn(trial)

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

                # Create epoch-level callback for MLP models
                max_epochs = params.get('epochs', 0)
                if max_epochs > 0:
                    def epoch_cb(epoch, loss, _fold=fold_idx, _max=max_epochs):
                        step = _fold * _max + epoch
                        trial.report(loss, step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        return True
                else:
                    epoch_cb = None

                fold_metrics = evaluate_fold(
                    model, X_train, y_train, X_val, y_val,
                    clusters_train=clusters_train,
                    resample_method=resample_method,
                    random_state=random_state,
                    _resample_skip_warned=resample_skip_warned,
                    epoch_callback=epoch_cb,
                )
            except optuna.TrialPruned:
                raise  # Re-raise pruning signal to Optuna
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

            # Report penalized running mean to Optuna for pruning
            # Use step after all epoch steps to avoid conflicts
            max_epochs = params.get('epochs', 0)
            fold_step = len(cv_splits) * max_epochs + fold_idx if max_epochs > 0 else fold_idx
            running_mean = np.mean(fold_penalized_values)
            trial.report(running_mean, fold_step)
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
) -> 'optuna.Study':
    """Run Optuna hyperparameter tuning for a single model."""

    if not HAS_OPTUNA:
        print("Error: optuna is required. Install with: pip install optuna", file=sys.stderr)
        sys.exit(1)

    if model_name not in SEARCH_SPACES:
        print(f"Error: No search space defined for '{model_name}'. "
              f"Available: {', '.join(SEARCH_SPACES.keys())}", file=sys.stderr)
        sys.exit(1)

    # Preload data and precompute k-fold splits once - shared across all trials
    # Use keep_variants=True to preserve raw/adj columns for feature variant tuning
    use_variants = model_name in ('xgboost', 'mlp', 'grouped_mlp')
    preloaded_df = load_and_prepare_base_data(csv_path, keep_variants=use_variants)

    model_class = MODEL_REGISTRY[model_name]
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
    )

    direction = 'minimize' if minimize else 'maximize'
    study_name = f"tune_{model_name}"
    if resample_method:
        study_name += f"_{resample_method}"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )

    print("=" * 80)
    print(f"TUNING: {model_name.upper()}")
    print("=" * 80)
    print(f"Metric:     {metric} ({direction})")
    print(f"Trials:     {n_trials}")
    print(f"CV Splits:  {n_splits}")
    print(f"Resample:   {resample_method or 'none'}")
    print(f"Full Data:  {full_data}")
    if storage:
        print(f"Storage:    {storage}")
    print("=" * 80)

    # Callback to save best params whenever a new best trial is found
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"best_{model_name}_params.json"

    def save_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            best = {
                'model': model_name,
                'metric': metric,
                'direction': direction,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'best_trial': trial.number,
                'n_trials_so_far': len(study.trials),
                'resample_method': resample_method,
            }
            with open(result_file, 'w') as f:
                json.dump(best, f, indent=2, default=str)
            print(f"  ★ New best! Trial {trial.number}: {metric} = {study.best_value:.6f} → saved to {result_file}")

    study.optimize(objective, n_jobs=n_jobs, n_trials=n_trials, show_progress_bar=True,
                   callbacks=[save_best_callback])

    # Re-evaluate best trial to get full metrics including train performance
    print("\n" + "=" * 80)
    print(f"BEST RESULT: {model_name.upper()}")
    print("=" * 80)
    print(f"Best {metric}: {study.best_value:.6f}")
    print(f"\nBest params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

    # Get full evaluation on best params to extract train metrics and overfitting gaps
    print(f"\nRe-evaluating best params to extract train metrics...")
    best_model_kwargs = convert_best_params(model_name, study.best_params)
    best_summary, _, _ = run_kfold_evaluation(
        model_class=model_class,
        model_kwargs=best_model_kwargs,
        csv_path=csv_path,
        n_splits=n_splits,
        random_state=random_state,
        verbose=False,
        save_importance_path=None,
        resample_method=resample_method,
        precomputed_data=precomputed_data,
    )

    # Print overfitting diagnostics
    print(f"\n{'='*80}")
    print("OVERFITTING DIAGNOSTICS")
    print(f"{'='*80}")

    train_metric_key = f'train_{metric}_mean'
    gap_metric_key = f'overfitting_gap_{metric}_mean'

    if train_metric_key in best_summary:
        print(f"\nValidation {metric}: {best_summary[f'{metric}_mean']:.6f}")
        print(f"Train {metric}:      {best_summary[train_metric_key]:.6f}")

        if gap_metric_key in best_summary:
            gap = best_summary[gap_metric_key]
            print(f"Overfitting Gap:     {gap:+.6f}")

            # Interpret the gap
            if abs(gap) < 0.02:
                print("✓ Minimal overfitting detected")
            elif abs(gap) < 0.05:
                print("⚠ Mild overfitting detected")
            else:
                print("⚠⚠ SIGNIFICANT overfitting detected - consider stronger regularization!")

    print("=" * 80)

    # Save final best params to JSON with overfitting diagnostics
    # Extract train metrics and overfitting gaps from best_summary
    train_metrics = {}
    overfitting_gaps = {}
    for key in best_summary.keys():
        if key.startswith('train_'):
            train_metrics[key] = best_summary[key]
        elif key.startswith('overfitting_gap_'):
            overfitting_gaps[key] = best_summary[key]

    result = {
        'model': model_name,
        'metric': metric,
        'direction': direction,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'resample_method': resample_method,
        # Add train metrics and overfitting diagnostics
        'validation_metrics': {
            'roc_auc': best_summary.get('roc_auc_mean'),
            'brier_score': best_summary.get('brier_score_mean'),
            'log_loss': best_summary.get('log_loss_mean'),
            'balanced_accuracy': best_summary.get('balanced_accuracy_mean'),
        },
        'train_metrics': train_metrics,
        'overfitting_gaps': overfitting_gaps,
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
  python tune_models.py --model xgboost --n-trials 100
  python tune_models.py --model mlp --n-trials 50 --metric log_loss
  python tune_models.py --model random_forest --n-trials 100 --resample undersample
  python tune_models.py --model all --n-trials 50
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
        )
        print()


if __name__ == '__main__':
    main()
