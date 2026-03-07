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

from utils.model_evaluator import run_kfold_evaluation
from utils.model_registry import MODEL_REGISTRY
from utils.data_utils import load_and_prepare_base_data


# ============================================================================
# Search space definitions
# ============================================================================

def suggest_xgboost_params(trial: 'optuna.Trial') -> Dict:
    """Define XGBoost hyperparameter search space."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
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

    # Test whether excluding turn_progress helps
    exclude_turn = trial.suggest_categorical('exclude_turn_progress', [True, False])
    if exclude_turn:
        params['exclude_features'] = ['turn_progress']

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
    """Define MLP hyperparameter search space."""
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_size = trial.suggest_int('layer_size', 8, 128)

    # Build hidden_layer_sizes tuple with decreasing sizes
    if n_layers == 1:
        hidden = (layer_size,)
    elif n_layers == 2:
        hidden = (layer_size, max(8, layer_size // 2))
    else:
        hidden = (layer_size, max(8, layer_size // 2), max(8, layer_size // 4))

    params = {
        'hidden_layer_sizes': hidden,
        'activation': trial.suggest_categorical('activation', ['relu', 'logistic']),
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 0.01, log=True),
        'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
        'max_iter': trial.suggest_int('max_iter', 1000, 4000),
        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
        'validation_fraction': 0.1,
    }

    params['batch_size'] = 'auto'

    # lbfgs doesn't support early_stopping
    if params['solver'] == 'lbfgs':
        params['early_stopping'] = False

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


def suggest_grouped_mlp_params(trial: 'optuna.Trial') -> Dict:
    """Define grouped MLP hyperparameter search space."""
    n_layers = trial.suggest_int('n_layers', 0, 3)
    layer_size = trial.suggest_int('layer_size', 8, 256)

    if n_layers == 0:
        layer_sizes = ()
    elif n_layers == 1:
        layer_sizes = (layer_size,)
    elif n_layers == 2:
        layer_sizes = (layer_size, max(8, layer_size // 2))
    else:
        layer_sizes = (layer_size, max(8, layer_size // 2), max(8, layer_size // 4))

    params = {
        'layer_sizes': layer_sizes,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 3, 20),
        'batch_size_groups': trial.suggest_categorical(
            'batch_size_groups', [128, 256, 512, 1024]
        ),
    }

    return params


SEARCH_SPACES = {
    'xgboost': suggest_xgboost_params,
    'random_forest': suggest_random_forest_params,
    'mlp': suggest_mlp_params,
    'grouped_mlp': suggest_grouped_mlp_params,
}


# ============================================================================
# Objective function
# ============================================================================

def create_objective(
    model_name: str,
    csv_path: str,
    metric: str = 'brier_score',
    n_splits: int = 5,
    random_state: int = 42,
    resample_method: Optional[str] = None,
    preloaded_df=None,
    full_data: bool = False,
):
    """Create an Optuna objective function for a given model."""

    model_class = MODEL_REGISTRY[model_name]
    suggest_fn = SEARCH_SPACES[model_name]

    # Metrics where lower is better
    minimize_metrics = {'brier_score', 'log_loss'}

    def objective(trial: 'optuna.Trial') -> float:
        params = suggest_fn(trial)

        try:
            summary, _, _ = run_kfold_evaluation(
                model_class=model_class,
                model_kwargs=params,
                csv_path=csv_path,
                n_splits=n_splits,
                random_state=random_state,
                verbose=False,
                save_importance_path=None,
                resample_method=resample_method,
                preloaded_df=preloaded_df,
                full_data=full_data,
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            # Return worst possible value
            if metric in minimize_metrics:
                return float('inf')
            else:
                return 0.0

        value = summary[f'{metric}_mean']

        # Check for overfitting using train metrics
        train_metric_key = f'train_{metric}_mean'
        if train_metric_key in summary:
            train_value = summary[train_metric_key]
            val_value = value

            # Compute overfitting gap (direction-aware)
            if metric in minimize_metrics:
                # For loss metrics: train should be lower (better) than validation
                # Gap = val - train (positive gap = overfitting)
                gap = val_value - train_value
            else:
                # For performance metrics: train should be higher (better) than validation
                # Gap = train - val (positive gap = overfitting)
                gap = train_value - val_value

            # Detect excessive overfitting
            overfitting_threshold = 0.05  # 5% gap threshold
            if gap > overfitting_threshold:
                print(f"  Trial {trial.number}: {metric} = {value:.6f} ⚠ OVERFITTING (gap={gap:.4f})  "
                      f"(params: {_format_params(params)})")
            else:
                # Print normal progress with train-val info
                print(f"  Trial {trial.number}: {metric} = {value:.6f} (train={train_value:.6f}, gap={gap:+.4f})  "
                      f"(params: {_format_params(params)})")

            # Constantly apply penalty to discourage overfitting
            penalty = gap**2 * 10  # gap ^ 2 * 10 as penalty (e.g. 0.1 => 0.1, 0.05 => 0.025, 0.01 => 0.001 etc)
            if metric in minimize_metrics:
                value += penalty  # Penalize by increasing the loss
            else:
                value -= penalty  # Penalize by decreasing the score
        else:
            # Fall back to original progress print if train metrics not available
            print(f"  Trial {trial.number}: {metric} = {value:.6f}  "
                  f"(params: {_format_params(params)})")

        return value

    return objective, metric in minimize_metrics


def _format_params(params: Dict) -> str:
    """Format parameters for concise display."""
    skip = {'exclude_features', 'calibration_method', 'validation_fraction'}
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
    metric: str = 'brier_score',
    n_splits: int = 5,
    random_state: int = 42,
    resample_method: Optional[str] = None,
    storage: Optional[str] = None,
    full_data: bool = False,
) -> 'optuna.Study':
    """Run Optuna hyperparameter tuning for a single model."""

    if not HAS_OPTUNA:
        print("Error: optuna is required. Install with: pip install optuna", file=sys.stderr)
        sys.exit(1)

    if model_name not in SEARCH_SPACES:
        print(f"Error: No search space defined for '{model_name}'. "
              f"Available: {', '.join(SEARCH_SPACES.keys())}", file=sys.stderr)
        sys.exit(1)

    # Preload data once - shared across all trials and re-evaluation
    preloaded_df = load_and_prepare_base_data(csv_path)

    objective, minimize = create_objective(
        model_name, csv_path, metric, n_splits, random_state, resample_method,
        preloaded_df=preloaded_df,
        full_data=full_data,
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

    study.optimize(objective, n_jobs=2, n_trials=n_trials, show_progress_bar=True)

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

    # Get model class from registry for re-evaluation
    model_class = MODEL_REGISTRY[model_name]

    # Get full evaluation on best params to extract train metrics and overfitting gaps
    print(f"\nRe-evaluating best params to extract train metrics...")
    best_summary, _, _ = run_kfold_evaluation(
        model_class=model_class,
        model_kwargs=study.best_params,
        csv_path=csv_path,
        n_splits=n_splits,
        random_state=random_state,
        verbose=False,
        save_importance_path=None,
        resample_method=resample_method,
        preloaded_df=preloaded_df,
        full_data=full_data,
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

    # Save best params to JSON with overfitting diagnostics
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / f"best_{model_name}_params.json"

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
        '--n-trials', type=int, default=100,
        help="Number of Optuna trials (default: 100)"
    )
    parser.add_argument(
        '--metric', type=str, default='brier_score',
        choices=['brier_score', 'log_loss', 'roc_auc', 'balanced_accuracy'],
        help="Metric to optimize (default: brier_score)"
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
        )
        print()


if __name__ == '__main__':
    main()
