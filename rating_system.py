"""
Rating system for player type evaluation using Plackett-Luce MLE.

This module provides functions to calculate and analyze player ratings
using the Plackett-Luce model via R's PlackettLuce package (called
via subprocess).
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os


def _find_rscript():
    """Find Rscript executable on the system."""
    # Check common Windows locations
    r_paths = []
    for drive in ['C', 'D']:
        r_base = f"{drive}:/Program Files/R"
        if os.path.isdir(r_base):
            for entry in sorted(os.listdir(r_base), reverse=True):
                candidate = os.path.join(r_base, entry, 'bin', 'Rscript.exe')
                if os.path.isfile(candidate):
                    r_paths.append(candidate)

    if r_paths:
        return r_paths[0]

    # Fallback: try PATH
    return 'Rscript'


def calculate_ratings(strength_df, verbose=True, **kwargs):
    """
    Calculate player ratings using R's PlackettLuce (batch MLE).

    Fits a Plackett-Luce model to all game rankings simultaneously via
    maximum likelihood estimation. Deterministic — no random ordering needed.

    Args:
        strength_df: DataFrame with columns: game_id, player_id, player_type,
                     adjusted_strength, civilization
        verbose: Print progress and results

    Returns:
        DataFrame with columns: player_type, worth, log_worth, se_log_worth,
                                z_value, p_value, elo, se_elo, mu, sigma, mu_std
    """
    # Prepare input data for R script
    # Assign unique slot IDs for duplicate player types within each game
    input_cols = ['game_id', 'player_type', 'adjusted_strength']
    r_input = strength_df[input_cols].copy()
    slot_ids = []
    for _, game in r_input.groupby('game_id'):
        type_counters = {}
        for idx in game.index:
            pt = game.at[idx, 'player_type']
            count = type_counters.get(pt, 0)
            slot_ids.append(f"{pt}_{count}")
            type_counters[pt] = count + 1
    r_input['slot_id'] = slot_ids

    # Write to temp CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                     newline='') as f_in:
        input_path = f_in.name
        r_input.to_csv(f_in, index=False)

    output_path = input_path.replace('.csv', '_output.csv')

    try:
        # Find R script path (relative to this file)
        r_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'plackett_luce.R')
        rscript_exe = _find_rscript()

        if verbose:
            print("=" * 76)
            print("CALCULATING RATINGS USING PLACKETT-LUCE MLE (R)")
            print("=" * 76)
            print(f"\nTotal games: {strength_df['game_id'].nunique()}")
            print(f"Player types: {sorted(strength_df['player_type'].unique())}")
            print(f"\nFitting model...")

        # Call R
        result = subprocess.run(
            [rscript_exe, r_script, input_path, output_path],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}")

        if verbose and result.stdout:
            print(result.stdout.strip())

        # Read results (already re-centered on Vanilla by R script)
        results_df = pd.read_csv(output_path)

        # Read diagnostics
        diagnostics = {}
        diag_path = output_path.replace('.csv', '_diagnostics.csv')
        if os.path.exists(diag_path):
            diag_df = pd.read_csv(diag_path)
            diagnostics = dict(zip(diag_df['metric'], diag_df['value']))

        # Elo conversion: 1500 + 400 * log10(worth)
        # Vanilla has worth=1, so Elo=1500
        results_df['elo'] = 1500 + 400 * np.log10(results_df['worth'])
        results_df['se_elo'] = 400 / np.log(10) * results_df['se_log_worth']

        # Backward compatibility aliases
        results_df['mu'] = results_df['log_worth']
        results_df['sigma'] = results_df['se_log_worth']
        results_df['mu_std'] = 0.0

        results_df = results_df.sort_values('elo', ascending=False)

        if verbose:
            print("\n" + "=" * 76)
            print("RATING SUMMARY (PLACKETT-LUCE MLE)")
            print("=" * 76)
            print(f"{'Rank':<6} {'Player Type':<25} {'Worth':<10} {'Log-Worth':<12} "
                  f"{'SE':<10} {'Elo':<8} {'p-value':<10}")
            print("-" * 81)

            for rank, row in enumerate(results_df.itertuples(), 1):
                if row.player_type == 'Vanilla':
                    p_str = "ref"
                elif not np.isnan(row.p_value):
                    p_str = f"{row.p_value:.4f}"
                else:
                    p_str = "N/A"
                print(f"{rank:<6} {row.player_type:<25} {row.worth:>8.4f} "
                      f"{row.log_worth:>10.4f} {row.se_log_worth:>10.4f} "
                      f"{row.elo:>8.0f} {p_str:>10}")

            print("\n" + "=" * 76)
            print(f"Elo Range: {results_df['elo'].max() - results_df['elo'].min():.0f} points")
            if diagnostics:
                print(f"Deviance: {diagnostics.get('deviance', 'N/A'):.1f}")
                print(f"AIC: {diagnostics.get('AIC', 'N/A'):.1f}")
                print(f"Iterations: {int(diagnostics.get('n_iterations', 0))}")

        return results_df

    finally:
        # Clean up temp files
        for path in [input_path, output_path,
                     output_path.replace('.csv', '_diagnostics.csv'),
                     output_path.replace('.csv', '_slots.csv')]:
            if os.path.exists(path):
                os.unlink(path)


def compare_with_ols(summary_df, ols_model, baseline_type='Vanilla',
                     baseline_rating=0.0, verbose=True):
    """
    Compare Plackett-Luce ratings with OLS regression coefficients for validation.

    Args:
        summary_df: DataFrame from calculate_ratings()
        ols_model: Fitted statsmodels OLS model
        baseline_type: Reference player type (default: 'Vanilla')
        baseline_rating: Baseline rating value (default: 0.0)
        verbose: Print comparison table

    Returns:
        DataFrame with correlation analysis
    """
    # Extract OLS coefficients
    params = ols_model.params
    condition_vars = [col for col in params.index if 'player_type' in col and col != 'Intercept']

    ols_effects = {}
    ols_effects[baseline_type] = 0.0  # Baseline

    for var in condition_vars:
        # Extract player type name
        player_name = var.split('[T.')[-1].rstrip(']')
        ols_effects[player_name] = params[var]

    # Create comparison dataframe
    comparison_data = []

    for _, row in summary_df.iterrows():
        player_type = row['player_type']
        rating = row['log_worth']
        rating_deviation = rating - baseline_rating
        ols_coef = ols_effects.get(player_type, np.nan)

        comparison_data.append({
            'player_type': player_type,
            'rating': rating,
            'rating_deviation': rating_deviation,
            'ols_coefficient': ols_coef
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('rating', ascending=False)

    if verbose:
        print("=" * 60)
        print("VALIDATION: PLACKETT-LUCE MLE vs OLS COEFFICIENTS")
        print("=" * 60)
        print("\nComparison of ranking methods:\n")
        print(f"{'Player Type':<25} {'Rank':<8} {'Log-Worth':<12} {'OLS Coef':<12}")
        print("-" * 57)

        for idx, row in enumerate(comparison_df.itertuples(), 1):
            print(f"{row.player_type:<25} {idx:<8} {row.rating_deviation:>10.4f}   {row.ols_coefficient:>10.4f}")

        # Calculate correlation
        valid_comparison = comparison_df.dropna()
        if len(valid_comparison) > 1:
            correlation = valid_comparison['rating_deviation'].corr(valid_comparison['ols_coefficient'])
            print(f"\n{'=' * 57}")
            print(f"Correlation: {correlation:.3f}")

    return comparison_df


def create_matchup_matrix(strength_df, verbose=True):
    """
    Create empirical pairwise outperform probability matrix between all player types.

    Analyzes actual game data to calculate how often each player type had higher
    adjusted strength than another when they played in the same game.

    Args:
        strength_df: DataFrame with columns: game_id, player_id, player_type, adjusted_strength
        verbose: Print summary statistics

    Returns:
        tuple: (matchup_df, count_df, pvalue_df) where:
            - matchup_df: NxN matrix of empirical probabilities (0-1)
            - count_df: NxN matrix of sample sizes (number of matchups)
            - pvalue_df: NxN matrix of p-values from one-way ANOVA
    """
    from scipy.stats import f_oneway

    # Get unique player types
    player_types = sorted(strength_df['player_type'].unique())
    n_players = len(player_types)

    # Initialize matrices
    matchup_matrix = np.zeros((n_players, n_players))
    count_matrix = np.zeros((n_players, n_players))

    # Store strength values for each matchup pair for ANOVA
    # strength_data[i][j] will contain list of (strength_A, strength_B) tuples
    strength_data = [[{'player_a': [], 'player_b': []} for _ in range(n_players)] for _ in range(n_players)]

    # For each game, compare all pairs of players
    for game_id in strength_df['game_id'].unique():
        game_data = strength_df[strength_df['game_id'] == game_id]

        # Compare all pairs of players in this game
        for _, player_a in game_data.iterrows():
            for _, player_b in game_data.iterrows():
                if player_a['player_type'] == player_b['player_type']:
                    continue  # Skip same player type

                # Get matrix indices
                i = player_types.index(player_a['player_type'])
                j = player_types.index(player_b['player_type'])

                # Store strength values for ANOVA
                strength_data[i][j]['player_a'].append(player_a['adjusted_strength'])
                strength_data[i][j]['player_b'].append(player_b['adjusted_strength'])

                # Compare adjusted strengths
                if player_a['adjusted_strength'] > player_b['adjusted_strength']:
                    matchup_matrix[i, j] += 1
                count_matrix[i, j] += 1

    # Convert counts to probabilities and calculate p-values using ANOVA
    matchup_probs = np.zeros((n_players, n_players))
    pvalue_matrix = np.ones((n_players, n_players))  # Default to 1.0 (not significant)

    for i in range(n_players):
        for j in range(n_players):
            if i == j:
                # Player vs themselves - set to NaN
                matchup_probs[i, j] = np.nan
                pvalue_matrix[i, j] = np.nan
            elif count_matrix[i, j] > 0:
                # Calculate probability
                matchup_probs[i, j] = matchup_matrix[i, j] / count_matrix[i, j]

                # Calculate p-value using one-way ANOVA
                # H0: mean strength of player A = mean strength of player B in their matchups
                group_a = strength_data[i][j]['player_a']
                group_b = strength_data[i][j]['player_b']

                if len(group_a) > 1 and len(group_b) > 1:
                    # Need at least 2 samples in each group for ANOVA
                    _, p_value = f_oneway(group_a, group_b)
                    pvalue_matrix[i, j] = p_value
                else:
                    # Not enough data for ANOVA
                    pvalue_matrix[i, j] = np.nan
            else:
                # No data for this matchup
                matchup_probs[i, j] = np.nan
                pvalue_matrix[i, j] = np.nan

    # Create DataFrames
    matchup_df = pd.DataFrame(
        matchup_probs,
        index=player_types,
        columns=player_types
    )

    count_df = pd.DataFrame(
        count_matrix,
        index=player_types,
        columns=player_types
    )

    pvalue_df = pd.DataFrame(
        pvalue_matrix,
        index=player_types,
        columns=player_types
    )

    if verbose:
        print("=" * 60)
        print("HEAD-TO-HEAD MATCHUP MATRIX (EMPIRICAL)")
        print("=" * 60)
        print(f"\nMatrix dimensions: {n_players}x{n_players}")
        print(f"Player types: {n_players}")
        print(f"Total games analyzed: {strength_df['game_id'].nunique()}")
        print(f"\nInterpretation:")
        print(f"  - Rows: Player A")
        print(f"  - Columns: Player B")
        print(f"  - Value: Empirical P(A has higher adjusted strength than B)")
        print(f"  - P-values: One-way ANOVA testing if mean strengths differ")
        print(f"\nSample sizes:")

        # Show sample sizes for each matchup
        for i, player_a in enumerate(player_types):
            if player_a == 'Vanilla':
                continue
            for j, player_b in enumerate(player_types):
                if player_b == 'Vanilla' and count_matrix[i, j] > 0:
                    print(f"  {player_a} vs {player_b}: {int(count_matrix[i, j])} matchups")

    return matchup_df, count_df, pvalue_df


def create_mean_matchup_matrix(strength_df, verbose=True):
    """
    Create pairwise mean strength difference matrix between all player types.

    Same iteration as create_matchup_matrix, but reports mean(A_strength - B_strength)
    instead of win rate. This captures whether A is stronger on average, even if A
    wins fewer than 50% of individual comparisons.

    Args:
        strength_df: DataFrame with columns: game_id, player_id, player_type, adjusted_strength
        verbose: Print summary statistics

    Returns:
        tuple: (mean_diff_df, count_df, pvalue_df) where:
            - mean_diff_df: NxN matrix of mean strength differences (A - B)
            - count_df: NxN matrix of sample sizes
            - pvalue_df: NxN matrix of p-values from one-sample t-test (H0: diff=0)
    """
    from scipy.stats import ttest_1samp

    player_types = sorted(strength_df['player_type'].unique())
    n_players = len(player_types)

    # Store all pairwise differences for each (i, j) pair
    diff_data = [[[] for _ in range(n_players)] for _ in range(n_players)]
    count_matrix = np.zeros((n_players, n_players))

    for game_id in strength_df['game_id'].unique():
        game_data = strength_df[strength_df['game_id'] == game_id]

        for _, player_a in game_data.iterrows():
            for _, player_b in game_data.iterrows():
                if player_a['player_type'] == player_b['player_type']:
                    continue

                i = player_types.index(player_a['player_type'])
                j = player_types.index(player_b['player_type'])

                diff_data[i][j].append(
                    player_a['adjusted_strength'] - player_b['adjusted_strength']
                )
                count_matrix[i, j] += 1

    # Calculate means and p-values
    mean_matrix = np.full((n_players, n_players), np.nan)
    pvalue_matrix = np.full((n_players, n_players), np.nan)

    for i in range(n_players):
        for j in range(n_players):
            if i == j:
                continue
            diffs = diff_data[i][j]
            if len(diffs) > 1:
                mean_matrix[i, j] = np.mean(diffs)
                _, p_value = ttest_1samp(diffs, 0)
                pvalue_matrix[i, j] = p_value
            elif len(diffs) == 1:
                mean_matrix[i, j] = diffs[0]

    mean_diff_df = pd.DataFrame(mean_matrix, index=player_types, columns=player_types)
    count_df = pd.DataFrame(count_matrix, index=player_types, columns=player_types)
    pvalue_df = pd.DataFrame(pvalue_matrix, index=player_types, columns=player_types)

    if verbose:
        print("=" * 60)
        print("HEAD-TO-HEAD MATCHUP MATRIX (MEAN STRENGTH DIFFERENCE)")
        print("=" * 60)
        print(f"\nMatrix dimensions: {n_players}x{n_players}")
        print(f"Player types: {n_players}")
        print(f"Total games analyzed: {strength_df['game_id'].nunique()}")
        print(f"\nInterpretation:")
        print(f"  - Rows: Player A")
        print(f"  - Columns: Player B")
        print(f"  - Value: Mean(A_adjusted_strength - B_adjusted_strength)")
        print(f"  - Positive = A is stronger on average")
        print(f"  - P-values: One-sample t-test (H0: mean diff = 0)")

    return mean_diff_df, count_df, pvalue_df
