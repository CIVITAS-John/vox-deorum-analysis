#!/usr/bin/env python3
"""
Data utilities for loading, processing, and splitting Civilization V game data.
Handles city-based cost adjustments per Vox Populi mechanics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from pathlib import Path
from typing import Optional, List, Tuple, Literal

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    SMOTE = None
    RandomUnderSampler = None


def get_phase_data(
    df: pd.DataFrame,
    phase_index: Optional[int] = None,
    phase_boundaries: List[float] = [0.33, 0.66]
) -> pd.DataFrame:
    """
    Filter data to specific game phase based on turn_progress.

    Args:
        df: DataFrame with turn_progress column
        phase_index: None for all data, or 0-based phase index
                    (0 = before first boundary, 1 = between first and second, etc.)
        phase_boundaries: List of turn_progress values defining phase splits
                         e.g., [0.33, 0.66] creates 3 phases

    Returns:
        Filtered DataFrame
    """
    if phase_index is None:
        return df

    # Ensure turn_progress exists
    if 'turn_progress' not in df.columns:
        # Calculate if missing
        df = df.copy()
        df['turn_progress'] = df['turn'] / df['max_turn']

    # Create phase bins: [0.0, boundary1, boundary2, ..., 1.0]
    boundaries = [0.0] + list(phase_boundaries) + [1.0]

    if phase_index < 0 or phase_index >= len(boundaries) - 1:
        raise ValueError(f"Invalid phase_index {phase_index}. Must be 0 to {len(boundaries) - 2}")

    lower = boundaries[phase_index]
    upper = boundaries[phase_index + 1]

    return df[(df['turn_progress'] >= lower) & (df['turn_progress'] < upper)]


def load_turn_data(
    csv_path: str = "../turn_data.csv",
    filter_experiments: Optional[List[str]] = None,
    phase_filter: Optional[Tuple[int, List[float]]] = None,
    filter_zero_score: bool = True
) -> pd.DataFrame:
    """
    Load turn data CSV with optional experiment and phase filtering.

    Args:
        csv_path: Path to turn_data.csv
        filter_experiments: List of experiment names to include (None = all)
        phase_filter: Tuple of (phase_index, phase_boundaries) for phase filtering
                     Default None keeps all data
                     Example: (2, [0.33, 0.8]) would keep phase 2 (80%+ of game)
        filter_zero_score: If True, filters out records where score = 0 (eliminated players)
                          Default True for training/evaluation

    Returns:
        DataFrame with turn data
    """
    df = pd.read_csv(csv_path)

    # Apply experiment filter first
    if filter_experiments is not None:
        df = df[df['experiment'].isin(filter_experiments)]
        print(f"Filtered to {len(filter_experiments)} experiments: {filter_experiments}")

    # Filter out score = 0 cases (eliminated/inactive players)
    if filter_zero_score:
        original_len = len(df)
        df = df[df['score'] != 0]
        filtered_count = original_len - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} records with score = 0")

    # Ensure turn_progress is calculated
    if 'turn_progress' not in df.columns:
        df['turn_progress'] = df['turn'] / df['max_turn']

    # Apply phase filtering
    if phase_filter is not None:
        phase_index, phase_boundaries = phase_filter
        df = get_phase_data(df, phase_index, phase_boundaries)

    print(f"Loaded {len(df)} records from {df['game_id'].nunique()} games")
    return df


def apply_city_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Vox Populi city scaling formula to per-turn metrics.
    Formula: adjusted_value = value / (1.05 * (cities - 1))

    Applied to: science_per_turn, culture_per_turn, tourism_per_turn, gold_per_turn
    NOT applied to: policies, technologies (these are cumulative, not per-turn rates)

    Args:
        df: DataFrame with raw game data

    Returns:
        DataFrame with additional adjusted columns
    """
    df = df.copy()

    # City cost multiplier: 1.05 * (cities - 1), minimum 1 to avoid division issues
    df['city_multiplier'] = np.maximum(1.05 * (df['cities'] - 1), 1.0)

    # Apply adjustments to per-turn metrics
    df['science_adj'] = df['science_per_turn'] / df['city_multiplier']
    df['culture_adj'] = df['culture_per_turn'] / df['city_multiplier']
    df['tourism_adj'] = df['tourism_per_turn'] / df['city_multiplier']
    df['gold_adj'] = df['gold_per_turn'] / df['city_multiplier']
    df['faith_adj'] = df['faith_per_turn'] / df['city_multiplier']
    df['production_adj'] = df['production_per_turn'] / df['city_multiplier']
    df['food_adj'] = df['food_per_turn'] / df['city_multiplier']
    df['military_adj'] = df['military_strength'] / df['city_multiplier']
    df['population_per_city'] = df['population'] / df['cities'].replace(0, 1)
    df['territory_per_city'] = df['territory'] / df['cities'].replace(0, 1)

    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that express player's position relative to others in the same turn.

    Features added:
    - score_ratio: player's score / max score in turn
    - rank_normalized: (max_players + 1 - rank) / max_players (1.0 = 1st place, 0.25 = 4th in 4-player)
    - turn_progress: turn / max_turn

    Args:
        df: DataFrame with game data

    Returns:
        DataFrame with additional relative features
    """
    df = df.copy()

    # Auto-detect max players per game
    max_players_per_game = df.groupby('game_id')['player_id'].nunique()
    df['max_players'] = df['game_id'].map(max_players_per_game)

    # Score ratio (avoid division by zero)
    df['score_ratio'] = df['score'] / df['max_score'].replace(0, 1)

    # Normalized rank (1.0 = first place, closer to 0 = last place)
    df['rank_normalized'] = (df['max_players'] + 1 - df['rank']) / df['max_players']

    # Turn progress (0 = start, 1 = end)
    df['turn_progress'] = df['turn'] / df['max_turn']

    return df


def add_competitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add competitive position features that measure standing relative to opponents within each turn.
    This removes temporal confounding by using relative shares and gaps instead of absolute values.

    Args:
        df: DataFrame with city-adjusted features already computed

    Returns:
        DataFrame with additional competitive features
    """
    df = df.copy()

    # Define grouping key: within same game and turn
    group_key = ['game_id', 'turn']

    # Metrics for relative share calculation (per-turn rates, already city-adjusted)
    share_metrics = {
        'science_adj': 'science_share',
        'culture_adj': 'culture_share',
        'tourism_adj': 'tourism_share',
        'gold_adj': 'gold_share',
        'faith_adj': 'faith_share',
        'production_adj': 'production_share',
        'food_adj': 'food_share',
        'military_adj': 'military_share',
        'cities': 'cities_share',
        'population': 'population_share',
        'votes': 'votes_share',
        'minor_allies': 'minor_allies_share'
    }

    # Compute relative shares: (my_value / sum(all_values_in_turn)) * num_players
    # This normalizes so that an average player has 100% share regardless of player count
    for source_col, target_col in share_metrics.items():
        # Sum within each (game_id, turn) group
        group_sum = df.groupby(group_key)[source_col].transform('sum')
        # Avoid division by zero
        group_sum = group_sum.replace(0, 1)
        # Use max_players (computed per game in add_relative_features)
        df[target_col] = (df[source_col] / group_sum) * df['max_players'] * 100

    # Metrics for gap from leader calculation (cumulative metrics)
    gap_metrics = {
        'technologies': 'technologies_gap',
        'policies': 'policies_gap'
    }

    # Compute gap from leader: max_value - my_value
    for source_col, target_col in gap_metrics.items():
        # Maximum within each (game_id, turn) group
        group_max = df.groupby(group_key)[source_col].transform('max')
        df[target_col] = (group_max - df[source_col])

    return df


def drop_transformed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original/intermediate columns that have been superseded by transformations."""
    cols_to_drop = [
        # Raw per-turn rates (-> adjusted -> shares)
        'science_per_turn', 'culture_per_turn', 'tourism_per_turn',
        'gold_per_turn', 'faith_per_turn', 'production_per_turn', 'food_per_turn',
        # Raw absolute (-> adjusted -> share)
        'military_strength',
        # Intermediate adjusted columns (-> shares)
        'science_adj', 'culture_adj', 'tourism_adj', 'gold_adj',
        'faith_adj', 'production_adj', 'food_adj', 'military_adj',
        # Intermediate computation columns
        'city_multiplier', 'max_players',
        # Raw columns (-> shares/gaps)
        'cities', 'population', 'territory', 'votes', 'minor_allies',
        'technologies', 'policies',
        # Raw columns (-> relative features)
        'score', 'max_score', 'rank',
        # Text column (space savings)
        'rationale',
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# Feature group definitions
FEATURE_GROUPS = {
    'shares': [
        'science_share', 'culture_share', 'tourism_share', 'gold_share',
        'faith_share', 'production_share', 'food_share', 'military_share',
        'cities_share', 'population_share', 'votes_share', 'minor_allies_share'
    ],
    'gaps': [
        'technologies_gap', 'policies_gap'
    ],
    'percentages': [
        'happiness_percentage', 'religion_percentage'
    ],
    'progress': [
        'turn_progress'
    ]
}


def get_all_available_features(df: pd.DataFrame, include_civs: bool = False) -> List[str]:
    """
    Get list of all available feature columns.

    Args:
        df: DataFrame with processed data
        include_civs: Whether to include civilization dummy variables

    Returns:
        List of all feature column names
    """
    # Combine all numeric feature groups
    features = []
    for group in FEATURE_GROUPS.values():
        features.extend(group)

    if include_civs:
        # Get civilization dummies (drop_first=True)
        civ_dummies = pd.get_dummies(
            df['civilization'],
            prefix='civ',
            drop_first=True,
            dtype=int
        )
        features.extend(civ_dummies.columns.tolist())

    return features


def get_feature_group(group_name: str) -> List[str]:
    """
    Get features belonging to a specific group.

    Args:
        group_name: Name of feature group ('shares', 'gaps', 'percentages', 'progress')

    Returns:
        List of feature names in that group
    """
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {group_name}. "
                        f"Available groups: {list(FEATURE_GROUPS.keys())}")
    return FEATURE_GROUPS[group_name].copy()


def prepare_features(df: pd.DataFrame, keep_ids: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y for modeling.

    Args:
        df: DataFrame with all preprocessing applied (including add_competitive_features)
        keep_ids: If True, include game_id, turn, player_id in X (default: True).
                  Models that don't need these columns will strip them automatically.

    Returns:
        Tuple of (X: features DataFrame, y: target Series)
    """
    feature_cols = [
        # Relative shares
        # 'science_share', 'culture_share',
        'tourism_share', 'gold_share',
        'production_share', 'military_share',
        'population_share', 'votes_share', 'minor_allies_share',
        # 'faith_share', 'cities_share', , 'food_share'
        # Gaps from leader (cumulative metrics)
        'technologies_gap', 'policies_gap',
        # Percentage metrics
        'happiness_percentage', 'religion_percentage', 'turn_progress'
    ]

    if keep_ids:
        # Include ID columns along with features (NEW BEHAVIOR - DEFAULT)
        # This allows models like PlackettLucePredictor that need grouping columns
        # to receive them directly, avoiding a strip-then-inject inefficiency
        id_cols = ['game_id', 'turn', 'player_id']
        X = df[id_cols + feature_cols].copy()
    else:
        # Old behavior: features only
        X = df[feature_cols].copy()

    # Add one-hot encoded civilization
    # Drop first category to avoid multicollinearity (dummy variable trap)
    civilization_dummies = pd.get_dummies(
        df['civilization'],
        prefix='civ',
        drop_first=True,
        dtype=int
    )

    # Combine numerical and categorical features
    # X = pd.concat([X, civilization_dummies], axis=1)

    y = df['is_winner'].copy()

    return X, y


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: Optional[pd.Series] = None,
    method: Optional[Literal['oversample', 'undersample', 'combined']] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Apply resampling to address class imbalance.

    Uses imbalanced-learn library for SMOTE (oversampling) and RandomUnderSampler.
    Note: This should only be applied to training data, never to validation/test data.

    Args:
        X: Feature matrix
        y: Target vector
        clusters: Cluster IDs (e.g., game_id) - will be preserved if provided
        method: Resampling method
            - None: No resampling (returns original data)
            - 'oversample': SMOTE oversampling of minority class
            - 'undersample': Random undersampling of majority class
            - 'combined': SMOTE + RandomUnderSampler (recommended in original SMOTE paper)
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled, clusters_resampled)
    """
    if method is None:
        return X, y, clusters

    if not HAS_IMBLEARN:
        raise ImportError(
            "imbalanced-learn library is required for resampling. "
            "Install with: pip install imbalanced-learn"
        )

    # print(f"\nApplying {method} resampling...")
    # print(f"  Before: {len(y)} samples, class distribution: {y.value_counts().to_dict()}")

    # Store cluster information if provided
    has_clusters = clusters is not None
    if has_clusters:
        # Encode cluster IDs as integers (SMOTE requires numeric features)
        # Map each unique cluster ID to an integer
        unique_clusters = clusters.unique()
        cluster_to_int = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}

        # Add encoded clusters as a numeric feature temporarily
        X_with_clusters = X.copy()
        X_with_clusters['__cluster_id__'] = clusters.map(cluster_to_int).values
    else:
        X_with_clusters = X.copy()

    # Apply resampling based on method
    if method == 'oversample':
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_with_clusters, y)

    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_with_clusters, y)

    elif method == 'combined':
        # Apply SMOTE first, then undersample (as recommended in original paper)
        smote = SMOTE(random_state=random_state)
        X_temp, y_temp = smote.fit_resample(X_with_clusters, y)

        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)

    else:
        raise ValueError(f"Unknown resampling method: {method}. "
                        f"Choose from: 'oversample', 'undersample', 'combined'")

    # Extract cluster information if it was present
    if has_clusters:
        # Round cluster IDs to nearest integer (for synthetic samples created by SMOTE)
        # and clip to valid range
        cluster_ints = X_resampled['__cluster_id__'].values.round().astype(int)
        cluster_ints = cluster_ints.clip(0, len(unique_clusters) - 1)

        # Keep as integers - downstream code only needs them for grouping/counting
        clusters_resampled = pd.Series(cluster_ints, name=clusters.name)
        X_resampled = X_resampled.drop(columns=['__cluster_id__'])
    else:
        clusters_resampled = None

    # Convert back to DataFrame/Series with proper indices
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    # print(f"  After: {len(y_resampled)} samples, class distribution: {y_resampled.value_counts().to_dict()}")

    return X_resampled, y_resampled, clusters_resampled


def get_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold cross-validation splits grouped by game_id.
    Ensures entire games stay together in train or validation.

    Args:
        df: DataFrame with 'game_id' column
        n_splits: Number of folds
        random_state: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['game_id']

    splits = []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        splits.append((train_idx, val_idx))

    # Print split statistics
    print(f"\nK-Fold CV Splits (k={n_splits}):")
    for i, (train_idx, val_idx) in enumerate(splits):
        train_games = df.iloc[train_idx]['game_id'].nunique()
        val_games = df.iloc[val_idx]['game_id'].nunique()
        train_win_rate = df.iloc[train_idx]['is_winner'].mean()
        val_win_rate = df.iloc[val_idx]['is_winner'].mean()
        print(f"  Fold {i+1}: {len(train_idx)} train ({train_games} games, {train_win_rate:.2%} wins) | "
              f"{len(val_idx)} val ({val_games} games, {val_win_rate:.2%} wins)")

    return splits


def load_and_prepare_data(
    csv_path: str = "../turn_data.csv",
    filter_experiments: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    phase_filter: Optional[Tuple[int, List[float]]] = None,
    filter_zero_score: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Full pipeline: load, adjust, engineer features, and create CV splits.

    Args:
        csv_path: Path to turn_data.csv
        filter_experiments: Experiment filter list
        n_splits: Number of CV folds. If 0, creates single split with
                  train='observe-vanilla-standard', val='all other experiments'
        random_state: Random seed
        phase_filter: Tuple of (phase_index, phase_boundaries) for phase filtering (None = all data)
        filter_zero_score: If True, filters out records where score = 0 (eliminated players)

    Returns:
        Tuple of (df_processed, X, y, cv_splits)
    """
    # Load data
    df = load_turn_data(csv_path, filter_experiments, phase_filter, filter_zero_score)

    # Apply transformations
    df = apply_city_adjustments(df)
    df = add_relative_features(df)
    df = add_competitive_features(df)

    # Drop original/intermediate columns superseded by transformations
    df = drop_transformed_columns(df)

    # Prepare features
    X, y = prepare_features(df)

    # Generate CV splits
    if n_splits == 0:
        # Custom split: train on observe-vanilla-standard, validate on all others
        train_mask = df['experiment'] == 'observe-vanilla-standard'
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(~train_mask)[0]

        cv_splits = [(train_idx, val_idx)]

        # Print split statistics (matching format of get_kfold_splits)
        train_games = df.iloc[train_idx]['game_id'].nunique()
        val_games = df.iloc[val_idx]['game_id'].nunique()
        train_wins = y.iloc[train_idx].mean() * 100
        val_wins = y.iloc[val_idx].mean() * 100

        print(f"\n=== Custom Experiment-Based Split ===")
        print(f"Train (observe-vanilla-standard): {len(train_idx)} samples "
              f"({train_games} games, {train_wins:.1f}% wins)")
        print(f"Val (all other experiments): {len(val_idx)} samples "
              f"({val_games} games, {val_wins:.1f}% wins)")
    else:
        cv_splits = get_kfold_splits(df, n_splits, random_state)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Win rate: {y.mean():.2%}")

    return df, X, y, cv_splits
