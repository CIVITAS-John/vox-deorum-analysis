"""
Rating system for player type evaluation using OpenSkill.

This module provides functions to calculate and analyze player ratings
using the Plackett-Luce model via the OpenSkill library, replacing
the custom ELO implementation.
"""

import pandas as pd
import numpy as np
from openskill.models import PlackettLuce, BradleyTerryFull
from collections import defaultdict

def prepare_game_level_data(strength_df):
    """
    Keep all players as individual entries for fair rating comparison.

    Args:
        strength_df: DataFrame with columns:
            - game_id: Unique game identifier
            - player_id: Player identifier within game
            - player_type: Type of player (Vanilla, GPT-OSS-120B-Simple, etc.)
            - adjusted_strength: Civilization-adjusted strength metric
            - civilization: Player's civilization

    Returns:
        DataFrame with game-level player records
    """
    game_level_players = []

    for game_id in strength_df['game_id'].unique():
        game = strength_df[strength_df['game_id'] == game_id].copy()

        # Keep all players (both Vanilla and LLM) as individual entries
        for _, player in game.iterrows():
            game_level_players.append({
                'game_id': game_id,
                'player_type': player['player_type'],
                'adjusted_strength': player['adjusted_strength'],
                'civilization': player['civilization'],
                'n_players': 1
            })

    return pd.DataFrame(game_level_players)


def process_single_game_openskill(game_players, ratings, model):
    """
    Process a single game using OpenSkill's model.

    When multiple players of the same type compete in one game, they each
    start with the same rating, get rated individually, then their updated
    ratings are averaged back to update the shared player_type rating.

    Args:
        game_players: DataFrame of players in one game
        ratings: Dict mapping player_type to Rating objects
        model: OpenSkill model instance

    Returns:
        Updated ratings dict
    """
    from collections import defaultdict
    from scipy.stats import rankdata

    players = game_players.to_dict('records')

    # Build teams: each player is independent with a copy of their type's rating
    teams = []
    strengths = []
    for player in players:
        r = ratings[player['player_type']]
        teams.append([type(r)(mu=r.mu, sigma=r.sigma)])  # Independent copy
        strengths.append(player['adjusted_strength'])

    # Convert adjusted_strength to ranks
    # Higher strength = better = lower rank (1 = best)
    # Use 'ordinal' method to handle ties (give different ranks to tied values)
    # Negate strengths so higher strength gets lower rank
    ranks = rankdata([-s for s in strengths], method='ordinal')

    # Rate all players individually using ranks
    # OpenSkill expects rank where 1 = winner, 2 = second, etc.
    updated_teams = model.rate(teams, scores=strengths) # ranks=ranks.tolist()
    # updated_teams = model.rate(teams, ranks=ranks.tolist()) # ranks=ranks.tolist()

    # Collect updated ratings by player type
    updates_by_type = defaultdict(list)
    for i, player in enumerate(players):
        updates_by_type[player['player_type']].append(updated_teams[i][0])

    # Average ratings for each player type
    new_ratings = ratings.copy()
    for player_type, rating_list in updates_by_type.items():
        if player_type == 'Vanilla':
            # Keep Vanilla's rating fixed (don't update it)
            continue
        avg_mu = sum(r.mu for r in rating_list) / len(rating_list)
        avg_sigma = sum(r.sigma for r in rating_list) / len(rating_list)
        # Keep original sigma to avoid it decreasing over iterations
        new_ratings[player_type] = type(rating_list[0])(mu=avg_mu, sigma=avg_sigma)

    return new_ratings


def calculate_ratings(strength_df, initial_mu=25.0, initial_sigma=8.33,
                     tau=0.083, n_runs=100, verbose=True):
    """
    Calculate player ratings using OpenSkill with random-order averaging.

    Processes all games multiple times with randomized game order, then
    averages the results to ensure order-independent ratings.

    Args:
        strength_df: DataFrame with player strength data
        initial_mu: Initial mean skill rating (default: 25.0)
        initial_sigma: Initial uncertainty (default: 8.33)
        tau: Dynamic factor for rating updates (default: 0.083)
        n_runs: Number of random orderings to average (default: 30)
        verbose: Print progress messages

    Returns:
        DataFrame with columns: player_type, mu, sigma, mu_std
    """
    # Prepare game-level data
    game_level_df = prepare_game_level_data(strength_df)

    if verbose:
        print("=" * 76)
        print("CALCULATING RATINGS USING OPENSKILL (RANDOM-ORDER AVERAGING)")
        print("=" * 76)
        print(f"\nModel: PlackettLuce (rank-based)")
        print(f"Initial rating: μ={initial_mu}, σ={initial_sigma}")
        print(f"Number of runs: {n_runs}")
        print(f"Total games: {game_level_df['game_id'].nunique()}")
        print(f"\nProcessing {n_runs} random orderings...")

    # Initialize model - Use PlackettLuce for rank-based rating
    model = PlackettLuce(mu=initial_mu, sigma=initial_sigma, tau=tau)

    # Get unique player types and games
    player_types = sorted(game_level_df['player_type'].unique())
    game_ids = game_level_df['game_id'].unique()

    # Store ratings from each run
    all_run_ratings = defaultdict(lambda: {'mu': [], 'sigma': []})

    # Run multiple times with different random orderings
    for run in range(n_runs):
        # Initialize ratings for this run
        ratings = {}
        for pt in player_types:
            ratings[pt] = model.rating()
        ratings["Vanilla"].sigma = 0.001

        # Shuffle game order
        shuffled_game_ids = np.random.permutation(game_ids)

        # Process all games once in this random order
        for game_id in shuffled_game_ids:
            game_players = game_level_df[game_level_df['game_id'] == game_id]
            ratings = process_single_game_openskill(game_players, ratings, model)

        # Store results from this run
        for pt in player_types:
            all_run_ratings[pt]['mu'].append(ratings[pt].mu)
            all_run_ratings[pt]['sigma'].append(ratings[pt].sigma)

        if verbose and (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs...")

    if verbose:
        print(f"\n✓ Random-order averaging complete!")

    # Average ratings across all runs and calculate uncertainty
    # First, get vanilla_mu for Elo calculations
    vanilla_mu = np.mean(all_run_ratings['Vanilla']['mu'])

    summary_data = []
    for player_type in player_types:
        mu_values = all_run_ratings[player_type]['mu']
        sigma_values = all_run_ratings[player_type]['sigma']

        avg_mu = np.mean(mu_values)
        avg_sigma = np.mean(sigma_values)
        mu_std = np.std(mu_values)  # Uncertainty across runs

        # Elo conversion using each player's own sigma for scaling
        # This makes sense since each player has different uncertainty
        elo_score = 1500 + (avg_mu - vanilla_mu) * (200 / avg_sigma)

        summary_data.append({
            'player_type': player_type,
            'mu': avg_mu,
            'sigma': avg_sigma,
            'mu_std': mu_std,
            'elo': elo_score,
        })

    summary_df = pd.DataFrame(summary_data).sort_values('elo', ascending=False)

    if verbose:
        print("\n" + "=" * 76)
        print("RATING SUMMARY")
        print("=" * 76)
        print(f"{'Rank':<6} {'Player Type':<25} {'Rating (μ)':<15} {'Sigma':<12} {'Elo':<10}")
        print("-" * 76)

        for rank, row in enumerate(summary_df.itertuples(), 1):
            print(f"{rank:<6} {row.player_type:<25} {row.mu:>15.2f} {row.sigma:>12.3f} {row.elo:>10.0f}")

        print("\n" + "=" * 76)
        print(f"μ Range: {summary_df['mu'].max() - summary_df['mu'].min():.2f} rating points")
        print(f"Elo Range: {summary_df['elo'].max() - summary_df['elo'].min():.0f} points")
        print(f"Avg Uncertainty (μ_std): {summary_df['mu_std'].mean():.3f}")

    return summary_df

def compare_with_ols(summary_df, ols_model, baseline_type='Vanilla', baseline_rating=25.0, verbose=True):
    """
    Compare OpenSkill ratings with OLS regression coefficients for validation.

    Args:
        summary_df: DataFrame from calculate_ratings()
        ols_model: Fitted statsmodels OLS model
        baseline_type: Reference player type (default: 'Vanilla')
        baseline_rating: Baseline rating value (default: 25.0)
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
        rating = row['mu']
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
        print("VALIDATION: OPENSKILL RATINGS vs OLS COEFFICIENTS")
        print("=" * 60)
        print("\nComparison of ranking methods:\n")
        print(f"{'Player Type':<25} {'Rank':<8} {'Rating Dev':<12} {'OLS Coef':<12}")
        print("-" * 57)

        for idx, row in enumerate(comparison_df.itertuples(), 1):
            print(f"{row.player_type:<25} {idx:<8} {row.rating_deviation:>10.2f}   {row.ols_coefficient:>10.4f}")

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
