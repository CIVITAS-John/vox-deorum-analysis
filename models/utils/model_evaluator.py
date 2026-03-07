#!/usr/bin/env python3
"""
Model-agnostic evaluation utilities for victory prediction models.
Supports k-fold cross-validation, phase-wise evaluation, and feature importance aggregation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    balanced_accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Literal
import sys
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.base_predictor import BasePredictor
from utils.data_utils import load_and_prepare_data, load_and_prepare_base_data, apply_resampling


def _strip_id_columns_if_not_needed(X: pd.DataFrame, model: BasePredictor) -> pd.DataFrame:
    """
    Strip ID columns from X if model doesn't require them.

    This is the inverse of injecting IDs - instead of adding IDs when needed,
    we remove them when NOT needed. More efficient since most models don't need IDs.

    Args:
        X: Feature DataFrame (may include game_id, turn, player_id)
        model: BasePredictor instance

    Returns:
        DataFrame with IDs stripped if model doesn't need them
    """
    required_ids = getattr(model, 'REQUIRES_ID_COLUMNS', None)

    if required_ids is None:
        # Model doesn't need IDs - strip common ID columns
        id_cols_to_strip = ['game_id', 'turn', 'player_id', 'experiment']
        cols_to_keep = [c for c in X.columns if c not in id_cols_to_strip]
        return X[cols_to_keep]
    else:
        # Model needs IDs - keep them
        return X


def evaluate_fold(
    model: BasePredictor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    clusters_train: pd.Series = None,
    resample_method: Optional[Literal['oversample', 'undersample', 'combined']] = None,
    random_state: int = 42,
    _resample_skip_warned: Optional[set] = None
) -> Dict[str, float]:
    """
    Train and evaluate model on a single fold.

    Args:
        model: BasePredictor instance
        X_train, y_train: Training data (may include ID columns game_id, turn, player_id)
        X_val, y_val: Validation data (may include ID columns)
        clusters_train: Cluster IDs (game_id) for training data
        resample_method: Resampling method to apply to training data
        random_state: Random seed for resampling
        _resample_skip_warned: Internal tracking set for warning messages

    Returns:
        Dictionary of metrics

    Note:
        X_train and X_val may include ID columns. These will be stripped if the model
        doesn't require them (i.e., REQUIRES_ID_COLUMNS is None).
    """
    # Strip ID columns if model doesn't need them
    X_train = _strip_id_columns_if_not_needed(X_train, model)
    X_val = _strip_id_columns_if_not_needed(X_val, model)

    # Apply resampling to training data if requested (and model allows it)
    if resample_method is not None:
        # Check if model disables resampling
        if hasattr(model, 'DISABLE_RESAMPLING') and model.DISABLE_RESAMPLING:
            # Only warn once per model class
            if _resample_skip_warned is not None:
                model_name = model.__class__.__name__
                if model_name not in _resample_skip_warned:
                    print(f"  ⚠ Skipping resampling for {model_name} (model disables resampling)")
                    _resample_skip_warned.add(model_name)
        else:
            # Apply resampling
            X_train, y_train, clusters_train = apply_resampling(
                X_train, y_train, clusters_train,
                method=resample_method,
                random_state=random_state
            )

    # Train model with cluster information
    model.fit(X_train, y_train, clusters=clusters_train)

    # Predict on BOTH training and validation sets
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]  # P(win) on train
    y_train_pred = model.predict(X_train)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # P(win) on val
    y_val_pred = model.predict(X_val)

    # Compute metrics for BOTH train and validation
    metrics = {
        # Validation metrics
        'roc_auc': roc_auc_score(y_val, y_val_pred_proba),
        'brier_score': brier_score_loss(y_val, y_val_pred_proba),
        'log_loss': log_loss(y_val, y_val_pred_proba),
        'balanced_accuracy': balanced_accuracy_score(y_val, y_val_pred),
        # Train metrics (for overfitting detection)
        'train_roc_auc': roc_auc_score(y_train, y_train_pred_proba),
        'train_brier_score': brier_score_loss(y_train, y_train_pred_proba),
        'train_log_loss': log_loss(y_train, y_train_pred_proba),
        'train_balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        # Overfitting gaps (positive = overfitting for ROC-AUC/Accuracy, negative for Brier/LogLoss)
        'overfitting_gap_roc_auc': roc_auc_score(y_train, y_train_pred_proba) - roc_auc_score(y_val, y_val_pred_proba),
        'overfitting_gap_brier': brier_score_loss(y_val, y_val_pred_proba) - brier_score_loss(y_train, y_train_pred_proba),
        'overfitting_gap_log_loss': log_loss(y_val, y_val_pred_proba) - log_loss(y_train, y_train_pred_proba),
        'overfitting_gap_balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred) - balanced_accuracy_score(y_val, y_val_pred),
        # Dataset info
        'n_train': len(y_train),
        'n_val': len(y_val),
        'train_win_rate': y_train.mean(),
        'val_win_rate': y_val.mean(),
        'n_clusters_train': clusters_train.nunique() if clusters_train is not None else 0
    }

    return metrics


def evaluate_by_turn_phase(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model: BasePredictor,
    phases: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across different game phases.

    Args:
        df: Full DataFrame with turn column
        X, y: Feature matrix and target
        train_idx, val_idx: Indices for train/val split
        model: Fitted BasePredictor model
        phases: Dictionary mapping phase name to (turn_min, turn_max) tuples
                Default: {'mid': (100, 250), 'late': (250, inf)}

    Returns:
        Dictionary mapping phase name to metrics
    """
    if phases is None:
        # Default phases
        phases = {
            'mid': (100, 250),
            'late': (250, float('inf'))
        }

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    df_val = df.iloc[val_idx]

    phase_metrics = {}

    for phase_name, (turn_min, turn_max) in phases.items():
        # Filter validation set for this phase
        phase_mask = (df_val['turn'] >= turn_min) & (df_val['turn'] < turn_max)
        X_phase = X_val[phase_mask]
        y_phase = y_val[phase_mask]

        if len(y_phase) == 0:
            continue

        # Strip IDs if model doesn't need them
        X_phase = _strip_id_columns_if_not_needed(X_phase, model)

        # Predict
        y_pred_proba = model.predict_proba(X_phase)[:, 1]
        y_pred = model.predict(X_phase)

        # Compute metrics
        phase_metrics[phase_name] = {
            'roc_auc': roc_auc_score(y_phase, y_pred_proba),
            'brier_score': brier_score_loss(y_phase, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(y_phase, y_pred),
            'n_samples': len(y_phase)
        }

    return phase_metrics


def aggregate_feature_importance(
    models: List[BasePredictor],
    use_robust_se: bool = True
) -> Optional[pd.DataFrame]:
    """
    Aggregate feature importance across all folds.

    Args:
        models: List of fitted BasePredictor models from each fold
        use_robust_se: Include cluster-robust standard errors if available

    Returns:
        DataFrame with mean importance across folds, or None if not supported
    """
    importance_dfs = []

    for i, model in enumerate(models):
        imp_df = model.get_feature_importance()
        if imp_df is None:
            # Model doesn't support feature importance
            return None
        imp_df['fold'] = i
        importance_dfs.append(imp_df)

    if not importance_dfs:
        return None

    # Combine all folds
    all_importance = pd.concat(importance_dfs, ignore_index=True)

    # Aggregate by feature
    agg_cols = {
        'coefficient': ['mean', 'std']
    }

    # Add robust SE columns if available
    if use_robust_se and 'robust_se' in all_importance.columns:
        agg_cols['robust_se'] = ['mean', 'std']
        agg_cols['z_statistic'] = ['mean', 'std']
        agg_cols['significant_95'] = 'sum'  # Count how many folds are significant

    agg_importance = all_importance.groupby('feature').agg(agg_cols).reset_index()

    # Flatten column names
    if use_robust_se and 'robust_se' in all_importance.columns:
        agg_importance.columns = [
            'feature', 'coef_mean', 'coef_std',
            'robust_se_mean', 'robust_se_std', 'z_stat_mean', 'z_stat_std', 'n_folds_significant'
        ]
    else:
        agg_importance.columns = [
            'feature', 'coef_mean', 'coef_std'
        ]

    # Sort by mean coefficient magnitude
    agg_importance = agg_importance.sort_values('coef_mean', ascending=False, key=abs)

    return agg_importance


def run_kfold_evaluation(
    model_class: type,
    model_kwargs: Dict = None,
    csv_path: str = "../turn_data.csv",
    filter_experiments: List[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
    phases: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True,
    save_importance_path: Optional[str] = None,
    resample_method: Optional[Literal['oversample', 'undersample', 'combined']] = None,
    full_data: bool = False,
    preloaded_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict, Optional[pd.DataFrame], List[BasePredictor]]:
    """
    Run full k-fold cross-validation evaluation for any BasePredictor model.

    Args:
        model_class: BasePredictor subclass (e.g., BaselineVictoryPredictor)
        model_kwargs: Keyword arguments to pass to model constructor
        csv_path: Path to turn data
        filter_experiments: Experiment filter
        n_splits: Number of folds
        random_state: Random seed
        phases: Phase definitions for evaluate_by_turn_phase
        verbose: Print progress and results
        save_importance_path: Path to save feature importance CSV (if None, auto-saves)
        resample_method: Resampling method ('oversample', 'undersample', 'combined', or None)
        full_data: If True, use all turn data (no phase filtering)
        preloaded_df: Pre-processed DataFrame from load_and_prepare_base_data.
                      When provided, skips CSV loading and feature engineering.

    Returns:
        Tuple of (metrics_summary, feature_importance, fitted_models)
    """
    if model_kwargs is None:
        model_kwargs = {}

    if verbose:
        print("=" * 80)
        print(f"{model_class.__name__.upper()} - K-FOLD EVALUATION")
        print("=" * 80)
        if resample_method:
            print(f"Resampling Method: {resample_method}")
        else:
            print("Resampling Method: None (using original class distribution)")

    # Check if model needs full game data
    needs_full_data = (
        (callable(model_class) and not isinstance(model_class, type)) or  # It's a factory function
        model_class.__name__ == 'PhaseEnsemblePredictor' or
        (hasattr(model_class, 'NEEDS_FULL_DATA') and model_class.NEEDS_FULL_DATA)
    )

    # Load and prepare data with appropriate phase filtering
    if full_data:
        phase_filter = None
    elif needs_full_data:
        # Filter after mid-game (50%+)
        phase_filter = (1, [0.5])
    else:
        # For non-phased models, filter to late game (80%+)
        phase_filter = (1, [0.8])

    df, X, y, cv_splits = load_and_prepare_data(
        csv_path, filter_experiments, n_splits, random_state, phase_filter,
        preloaded_df=preloaded_df
    )

    # Store results
    fold_metrics = []
    fold_models = []
    phase_metrics_all = []
    resample_skip_warned = set()  # Track which models we've warned about

    # Run k-fold CV
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING AND EVALUATION")
        print("=" * 80)
        print(f"\n{'Fold':<6} {'Val ROC-AUC':<12} {'Train ROC-AUC':<13} {'Gap':<8} {'Val Brier':<11} {'Train Brier':<13} {'Gap':<8}")
        print("-" * 80)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clusters_train = df.iloc[train_idx]['game_id']
        clusters_val = df.iloc[val_idx]['game_id']

        # Train and evaluate
        # Check if model_class is actually a factory function (for phase ensembles)
        if callable(model_class) and not isinstance(model_class, type):
            # It's a factory function, call it with kwargs
            model = model_class(random_state=random_state, **model_kwargs)
        else:
            # It's a class, instantiate normally
            model = model_class(random_state=random_state, **model_kwargs)
        metrics = evaluate_fold(
            model, X_train, y_train, X_val, y_val,
            clusters_train=clusters_train,
            resample_method=resample_method,
            random_state=random_state,
            _resample_skip_warned=resample_skip_warned
        )

        n_train_games = clusters_train.nunique()
        n_val_games = clusters_val.nunique()

        if verbose:
            # Display train vs validation metrics with overfitting gaps
            gap_roc = metrics['overfitting_gap_roc_auc']
            gap_brier = metrics['overfitting_gap_brier']

            # Add warning indicator if gap is excessive
            warning_roc = " ⚠" if gap_roc > 0.05 else ""
            warning_brier = " ⚠" if gap_brier > 0.01 else ""

            print(f"{fold_idx + 1:<6} "
                  f"{metrics['roc_auc']:<12.4f} "
                  f"{metrics['train_roc_auc']:<13.4f} "
                  f"{gap_roc:+.4f}{warning_roc:<6} "
                  f"{metrics['brier_score']:<11.4f} "
                  f"{metrics['train_brier_score']:<13.4f} "
                  f"{gap_brier:+.4f}{warning_brier}")

        fold_metrics.append(metrics)
        fold_models.append(model)

        # Evaluate by turn phase
        phase_metrics = evaluate_by_turn_phase(df, X, y, train_idx, val_idx, model, phases)
        phase_metrics['fold'] = fold_idx
        phase_metrics_all.append(phase_metrics)

    # Aggregate metrics
    if verbose:
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        # Validation metrics
        'roc_auc_mean': metrics_df['roc_auc'].mean(),
        'roc_auc_std': metrics_df['roc_auc'].std(),
        'brier_score_mean': metrics_df['brier_score'].mean(),
        'brier_score_std': metrics_df['brier_score'].std(),
        'log_loss_mean': metrics_df['log_loss'].mean(),
        'log_loss_std': metrics_df['log_loss'].std(),
        'balanced_accuracy_mean': metrics_df['balanced_accuracy'].mean(),
        'balanced_accuracy_std': metrics_df['balanced_accuracy'].std(),
        # Train metrics
        'train_roc_auc_mean': metrics_df['train_roc_auc'].mean(),
        'train_roc_auc_std': metrics_df['train_roc_auc'].std(),
        'train_brier_score_mean': metrics_df['train_brier_score'].mean(),
        'train_brier_score_std': metrics_df['train_brier_score'].std(),
        'train_log_loss_mean': metrics_df['train_log_loss'].mean(),
        'train_log_loss_std': metrics_df['train_log_loss'].std(),
        'train_balanced_accuracy_mean': metrics_df['train_balanced_accuracy'].mean(),
        'train_balanced_accuracy_std': metrics_df['train_balanced_accuracy'].std(),
        # Overfitting gaps
        'overfitting_gap_roc_auc_mean': metrics_df['overfitting_gap_roc_auc'].mean(),
        'overfitting_gap_brier_mean': metrics_df['overfitting_gap_brier'].mean(),
        'overfitting_gap_log_loss_mean': metrics_df['overfitting_gap_log_loss'].mean(),
        'overfitting_gap_balanced_accuracy_mean': metrics_df['overfitting_gap_balanced_accuracy'].mean(),
    }

    if verbose:
        print(f"\nVALIDATION Metrics:")
        print(f"  ROC-AUC:           {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
        print(f"  Brier:             {summary['brier_score_mean']:.4f} ± {summary['brier_score_std']:.4f}")
        print(f"  Log Loss:          {summary['log_loss_mean']:.4f} ± {summary['log_loss_std']:.4f}")
        print(f"  Balanced Accuracy: {summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}")

        print(f"\nTRAIN Metrics:")
        print(f"  ROC-AUC:           {summary['train_roc_auc_mean']:.4f} ± {summary['train_roc_auc_std']:.4f}")
        print(f"  Brier:             {summary['train_brier_score_mean']:.4f} ± {summary['train_brier_score_std']:.4f}")
        print(f"  Log Loss:          {summary['train_log_loss_mean']:.4f} ± {summary['train_log_loss_std']:.4f}")
        print(f"  Balanced Accuracy: {summary['train_balanced_accuracy_mean']:.4f} ± {summary['train_balanced_accuracy_std']:.4f}")

        print(f"\nOVERFITTING GAPS (Train - Val):")
        print(f"  ROC-AUC Gap:       {summary['overfitting_gap_roc_auc_mean']:+.4f}")
        print(f"  Brier Gap:         {summary['overfitting_gap_brier_mean']:+.4f}")
        print(f"  Log Loss Gap:      {summary['overfitting_gap_log_loss_mean']:+.4f}")
        print(f"  Bal.Acc Gap:       {summary['overfitting_gap_balanced_accuracy_mean']:+.4f}")

        # Add overfitting warnings
        if summary['overfitting_gap_roc_auc_mean'] > 0.05:
            print(f"\n⚠ WARNING: ROC-AUC gap > 5% - model may be overfitting!")
        if summary['overfitting_gap_brier_mean'] > 0.01:
            print(f"\n⚠ WARNING: Brier score gap > 0.01 - model may be overfitting!")
        if summary['overfitting_gap_log_loss_mean'] > 0.05:
            print(f"\n⚠ WARNING: Log loss gap > 0.05 - model may be overfitting!")

    # Feature importance (if supported)
    feature_importance = None
    if verbose:
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE (Top 15)")
        print("=" * 80)

    feature_importance = aggregate_feature_importance(fold_models, use_robust_se=True)

    if feature_importance is not None and verbose:
        # Display format with robust SEs if available
        if 'robust_se_mean' in feature_importance.columns:
            display_cols = ['feature', 'coef_mean', 'robust_se_mean', 'z_stat_mean', 'n_folds_significant']
            display_df = feature_importance[display_cols].head(15)
        else:
            display_df = feature_importance.head(15)

        # Use tabulate if available, otherwise fall back to pandas
        if HAS_TABULATE:
            print("\n" + tabulate(display_df, headers='keys', tablefmt='simple', showindex=False, floatfmt='.4f'))
        else:
            print("\n" + display_df.to_string(index=False))

        if 'robust_se_mean' in feature_importance.columns:
            print("\nNote: n_folds_significant = number of folds where 95% CI excludes zero")
    elif verbose:
        print("\n(Feature importance not supported by this model)")

    # Auto-save feature importance if available
    if feature_importance is not None:
        if save_importance_path is None:
            # Auto-generate filename based on model class name
            model_name = model_class.__name__.lower().replace('predictor', '').replace('victory', '')
            save_importance_path = f"{model_name}_feature_importance.csv"

        feature_importance.to_csv(save_importance_path, index=False)
        if verbose:
            print(f"\nFeature importance saved to: {save_importance_path}")

    # Phase-wise performance
    if verbose and phase_metrics_all:
        print("\n" + "=" * 80)
        print("PERFORMANCE BY GAME PHASE")
        print("=" * 80)

        # Get unique phase names
        phase_names = set()
        for pm in phase_metrics_all:
            phase_names.update(k for k in pm.keys() if k != 'fold')

        for phase in sorted(phase_names):
            phase_data = []
            for pm in phase_metrics_all:
                if phase in pm:
                    phase_data.append(pm[phase])

            if phase_data:
                phase_df = pd.DataFrame(phase_data)
                print(f"\n{phase.upper()} game:")
                print(f"  ROC-AUC: {phase_df['roc_auc'].mean():.4f} ± {phase_df['roc_auc'].std():.4f}")
                print(f"  Brier Score: {phase_df['brier_score'].mean():.4f} ± {phase_df['brier_score'].std():.4f}")
                print(f"  Balanced Accuracy: {phase_df['balanced_accuracy'].mean():.4f} ± {phase_df['balanced_accuracy'].std():.4f}")
                print(f"  Samples: {phase_df['n_samples'].sum()}")

    return summary, feature_importance, fold_models


def run_full_prediction(
    model_class: type,
    model_kwargs: Dict = None,
    csv_path: str = "../turn_data.csv",
    filter_experiments: List[str] = None,
    random_state: int = 42,
    verbose: bool = True,
    save_predictions_path: Optional[str] = None,
    resample_method: Optional[Literal['oversample', 'undersample', 'combined']] = None,
    full_data: bool = False
) -> Tuple[BasePredictor, pd.DataFrame]:
    """
    Train model on full dataset and generate predictions for all turns.

    This mode is for production use: trains on ALL available data and outputs
    predictions for each turn. No cross-validation is performed.

    Args:
        model_class: BasePredictor subclass (e.g., BaselineVictoryPredictor)
        model_kwargs: Keyword arguments to pass to model constructor
        csv_path: Path to turn data
        filter_experiments: Experiment filter
        random_state: Random seed
        verbose: Print progress and results
        save_predictions_path: Path to save predictions CSV (if None, auto-generates)
        resample_method: Resampling method (typically None for full predictions)
        full_data: If True, use all turn data (no phase filtering)

    Returns:
        Tuple of (fitted_model, predictions_df)
            predictions_df contains all original turn data + 'predicted_win_probability' column
    """
    if model_kwargs is None:
        model_kwargs = {}

    if verbose:
        print("=" * 80)
        print(f"{model_class.__name__.upper()} - FULL PREDICTION MODE")
        print("=" * 80)
        print("Training on 100% of data (no cross-validation)")
        if resample_method:
            print(f"Resampling Method: {resample_method}")
        else:
            print("Resampling Method: None")

    # Load data (without creating CV splits)
    from utils.data_utils import load_turn_data, apply_city_adjustments, add_relative_features, add_competitive_features, prepare_features, drop_transformed_columns

    # Check if model needs full game data (e.g., PhaseEnsemble)
    needs_full_data = (
        (callable(model_class) and not isinstance(model_class, type)) or  # It's a factory function
        model_class.__name__ == 'PhaseEnsemblePredictor' or
        (hasattr(model_class, 'NEEDS_FULL_DATA') and model_class.NEEDS_FULL_DATA)
    )

    # Load data with appropriate phase filtering
    # Always filter out score=0 for training
    if full_data:
        df_train = load_turn_data(csv_path, filter_experiments, phase_filter=None, filter_zero_score=True)
    elif needs_full_data:
        # Use all data for phase-aware models
        df_train = load_turn_data(csv_path, filter_experiments, phase_filter=None, filter_zero_score=True)
    else:
        # For non-phased models, filter to late game (80%+)
        df_train = load_turn_data(csv_path, filter_experiments, phase_filter=(1, [0.8]), filter_zero_score=True)
    df_train = apply_city_adjustments(df_train)
    df_train = add_relative_features(df_train)
    df_train = add_competitive_features(df_train)
    df_train = drop_transformed_columns(df_train)

    # Prepare features for training
    X_train, y_train = prepare_features(df_train)
    clusters_train = df_train['game_id']

    if verbose:
        print(f"\nTraining Dataset (late game only): {len(df_train)} turns from {df_train['game_id'].nunique()} games")
        print(f"Features: {X_train.shape[1]}")
        print(f"Win rate: {y_train.mean():.2%}")

    # Instantiate and train model
    # Check if model_class is actually a factory function (for phase ensembles)
    if callable(model_class) and not isinstance(model_class, type):
        # It's a factory function, call it with kwargs
        model = model_class(random_state=random_state, **model_kwargs)
    else:
        # It's a class, instantiate normally
        model = model_class(random_state=random_state, **model_kwargs)

    # Apply resampling if requested (usually not recommended for full predictions)
    if resample_method is not None:
        if hasattr(model, 'DISABLE_RESAMPLING') and model.DISABLE_RESAMPLING:
            if verbose:
                print(f"  ⚠ Skipping resampling for {model.__class__.__name__} (model disables resampling)")
        else:
            from utils.data_utils import apply_resampling
            X_train, y_train, clusters_train = apply_resampling(
                X_train, y_train, clusters_train,
                method=resample_method,
                random_state=random_state
            )
            if verbose:
                print(f"After resampling: {len(y_train)} samples")

    if verbose:
        print("\nTraining model on late-game data...")

    # Strip ID columns if model doesn't need them
    X_train = _strip_id_columns_if_not_needed(X_train, model)

    model.fit(X_train, y_train, clusters=clusters_train)

    if verbose:
        print("Training complete!")

    # Generate predictions for ALL data (not just late game)
    # Load UNTRIMMED data for predictions - DO NOT filter score=0 here!
    # We want predictions for all turns including eliminated players
    df_all = load_turn_data(csv_path, filter_experiments, phase_filter=None, filter_zero_score=False)
    df_all = apply_city_adjustments(df_all)
    df_all = add_relative_features(df_all)
    df_all = add_competitive_features(df_all)
    df_all = drop_transformed_columns(df_all)
    X_all, _ = prepare_features(df_all)

    if verbose:
        print(f"\nGenerating predictions for ALL {len(df_all)} turns (including early/mid game)...")

    # Strip ID columns if model doesn't need them
    X_all = _strip_id_columns_if_not_needed(X_all, model)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_all)[:, 1]  # P(win)

    # Add predictions to dataframe
    df_all['predicted_win_probability'] = y_pred_proba

    if verbose:
        print(f"Predictions generated: {len(y_pred_proba)} turns")
        print(f"  Mean predicted probability: {y_pred_proba.mean():.4f}")
        print(f"  Std predicted probability: {y_pred_proba.std():.4f}")
        print(f"  Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}")

    # Save predictions
    if save_predictions_path is None:
        model_name = model_class.__name__.lower().replace('predictor', '').replace('victory', '')
        save_predictions_path = f"{model_name}_predictions.csv"

    df_all.to_csv(save_predictions_path, index=False)
    if verbose:
        print(f"\nPredictions saved to: {save_predictions_path}")
        print("=" * 80)

    # Display feature importance if available
    if verbose:
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            print("\nFEATURE IMPORTANCE (Top 15)")
            print("=" * 80)

            if 'robust_se' in feature_importance.columns:
                display_cols = ['feature', 'coefficient', 'robust_se', 'z_statistic', 'significant_95']
                display_df = feature_importance[display_cols].head(15)
            else:
                display_df = feature_importance.head(15)

            if HAS_TABULATE:
                print("\n" + tabulate(display_df, headers='keys', tablefmt='simple', showindex=False, floatfmt='.4f'))
            else:
                print("\n" + display_df.to_string(index=False))

    return model, df_all
