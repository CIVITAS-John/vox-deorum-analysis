#!/usr/bin/env python3
"""
Iterative Bradley-Terry ratings: add games one-by-one in chronological order
and compute BT ratings at each step. Outputs a CSV dataset and per-player-type
Elo convergence charts.

Usage:
    python -m ratings.iterative_bt [--no-cache] [--output-dir DIR]
"""

import sys
import os
import argparse
import hashlib
import pickle

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ratings import calculate_ratings_bt
from plot_utilities import (
    load_turn_data,
    get_player_color,
    get_player_linestyle,
    get_player_marker,
)


# ---------------------------------------------------------------------------
# Data preprocessing (reproduces the turn_predicted.ipynb pipeline)
# ---------------------------------------------------------------------------

def prepare_strength_data(turn_data_path='models/output/grouped_mlp_predictions.csv'):
    """
    Reproduce the strength_df computation from turn_predicted.ipynb.

    Returns:
        strength_df with columns: game_id, player_id, player_type, civilization,
                                  experiment, adjusted_strength
    """
    print("Loading turn data...")
    turn_df = load_turn_data(turn_data_path, condition_exclude=[], print_metadata=False)
    print(f"  {len(turn_df)} rows, {turn_df['game_id'].nunique()} games")

    # Step 1: Compute weighted_strength per player per game
    print("Computing weighted strength per player...")
    filtered = turn_df[turn_df['turn_progress'] > 0.2].copy()
    turn_progress_avg = filtered.groupby(
        ['game_id', 'player_id', 'turn_progress']
    ).agg({
        'predicted_win_probability': 'mean',
        'player_type': 'first',
        'experiment': 'first',
        'is_winner': 'last',
        'civilization': 'first',
    }).reset_index()
    turn_progress_avg['weight'] = turn_progress_avg['turn_progress']

    records = []
    for (game_id, player_id), grp in turn_progress_avg.groupby(['game_id', 'player_id']):
        w = grp['weight']
        p = grp['predicted_win_probability']
        weighted_avg = (w * p).sum() / w.sum()
        records.append({
            'game_id': game_id,
            'player_id': player_id,
            'player_type': grp['player_type'].iloc[0],
            'civilization': grp['civilization'].iloc[0],
            'experiment': grp['experiment'].iloc[0],
            'weighted_strength': weighted_avg,
            'is_winner': grp['is_winner'].iloc[-1],
        })

    strength_df = pd.DataFrame(records)

    # Step 2: Relative strength
    game_max = strength_df.groupby('game_id')['weighted_strength'].max().reset_index()
    game_max.columns = ['game_id', 'max_weighted_strength']
    strength_df = strength_df.merge(game_max, on='game_id')
    strength_df['relative_strength'] = (
        strength_df['weighted_strength'] / strength_df['max_weighted_strength']
    )

    # Step 3: Adjust winners
    winner_mask = (strength_df['is_winner'] == 1) & (strength_df['relative_strength'] < 1.0)
    strength_df.loc[winner_mask, 'weighted_strength'] = (
        strength_df.loc[winner_mask, 'max_weighted_strength'] + 0.001
    )
    strength_df.loc[winner_mask, 'relative_strength'] = 1.0

    # Step 4: OLS for civilization effects (on ALL data)
    print("Fitting OLS for civilization effects...")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from patsy.contrasts import Sum

    formula = 'relative_strength ~ C(civilization, Sum) + C(player_type, Treatment(reference="Vanilla"))'
    model = ols(formula, data=strength_df).fit()

    params = model.params
    civ_vars = [col for col in params.index if 'civilization' in col]
    civ_effects = {}
    for var in civ_vars:
        civ_name = var.split('[S.')[-1].rstrip(']')
        civ_effects[civ_name] = params[var]
    # Reference category (Sum coding: all effects sum to zero)
    civ_effects['Venice'] = -sum(civ_effects.values())

    # Step 5: Compute adjusted_strength
    strength_df['adjusted_strength'] = strength_df.apply(
        lambda row: row['relative_strength'] - civ_effects.get(row['civilization'], 0),
        axis=1,
    )

    # Step 6: Exclude observe-vanilla-standard
    strength_df = strength_df[
        ~strength_df['experiment'].isin(['observe-vanilla-standard'])
    ].copy()

    # Step 7: Exclude games where no LLM players are present
    games_with_llm = strength_df[
        strength_df['player_type'] != 'Vanilla'
    ]['game_id'].unique()
    strength_df = strength_df[strength_df['game_id'].isin(games_with_llm)].copy()

    print(f"  {strength_df['game_id'].nunique()} games after excluding vanilla-only games")
    return strength_df


# ---------------------------------------------------------------------------
# Game ordering
# ---------------------------------------------------------------------------

def load_game_order(strength_df, timestamps_path='game_timestamps.csv'):
    """
    Return a list of game_ids sorted chronologically using game_timestamps.csv.
    Falls back to alphabetical game_id ordering if timestamps unavailable.
    """
    valid_game_ids = set(strength_df['game_id'].unique())

    if os.path.exists(timestamps_path):
        ts_df = pd.read_csv(timestamps_path)
        ts_df = ts_df[ts_df['game_id'].isin(valid_game_ids)]
        ts_df = ts_df.sort_values('timestamp')
        ordered = ts_df['game_id'].tolist()

        # Append any games missing from timestamps file
        missing = valid_game_ids - set(ordered)
        if missing:
            ordered.extend(sorted(missing))
            print(f"  Warning: {len(missing)} games not found in timestamps file, appended at end")

        return ordered
    else:
        print(f"  Warning: {timestamps_path} not found, using alphabetical game_id order")
        return sorted(valid_game_ids)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


def cache_key(game_ids):
    """MD5 hash of sorted game_id list."""
    return hashlib.md5(','.join(sorted(game_ids)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main iteration
# ---------------------------------------------------------------------------

def run_iterative_bt(strength_df, game_order, cache, min_games=3):
    """
    Iteratively add games and compute BT ratings.

    Returns:
        list of result dicts, updated cache, new_opponent_events
    """
    results = []
    n_total = len(game_order)
    # Build experiment lookup and per-game player types
    exp_lookup = strength_df.drop_duplicates('game_id').set_index('game_id')['experiment'].to_dict()
    game_player_types = strength_df.groupby('game_id')['player_type'].apply(set).to_dict()

    # Track cumulative game count per player type
    player_game_counts = {}
    # Track which opponent types each player type has faced
    opponents_seen = {}  # player_type -> set of opponent player_types
    # Record events: list of (player_type, n_player_games, [new_opponent_names])
    new_opponent_events = []

    for i in range(1, n_total + 1):
        game_id = game_order[i - 1]

        # Update per-player-type game counts
        for pt in game_player_types.get(game_id, set()):
            player_game_counts[pt] = player_game_counts.get(pt, 0) + 1

        # Detect new opponent introductions
        pts_in_game = game_player_types.get(game_id, set())
        for pt in pts_in_game:
            if pt not in opponents_seen:
                opponents_seen[pt] = set()
            new_opps = pts_in_game - {pt} - opponents_seen[pt]
            if new_opps:
                opponents_seen[pt].update(new_opps)
                new_opponent_events.append((pt, player_game_counts[pt], sorted(new_opps)))

        games_so_far = game_order[:i]
        subset = strength_df[strength_df['game_id'].isin(games_so_far)]

        n_types = subset['player_type'].nunique()
        if n_types < 2 or i < min_games:
            continue

        key = cache_key(games_so_far)
        if key in cache:
            rating_df = cache[key]
        else:
            try:
                rating_df = calculate_ratings_bt(subset, verbose=False)
                cache[key] = rating_df
            except Exception as e:
                print(f"  [{i}/{n_total}] BT failed: {e}")
                continue

        for _, row in rating_df.iterrows():
            results.append({
                'game_index': i,
                'game_id': game_id,
                'experiment': exp_lookup.get(game_id, ''),
                'n_games': i,
                'n_player_games': player_game_counts.get(row['player_type'], 0),
                'player_type': row['player_type'],
                'elo': row['elo'],
                'se_elo': row['se_elo'],
                'log_worth': row['log_worth'],
                'se_log_worth': row['se_log_worth'],
            })

        if i % 10 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] {n_types} player types, {len(results)} total rows")

    return results, cache, new_opponent_events


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(results_df, output_dir, new_opponent_events=None):
    """Generate per-player-type and combined Elo convergence charts."""
    sns.set_style('whitegrid')

    player_types = sorted(
        results_df['player_type'].unique(),
        key=lambda pt: results_df[results_df['player_type'] == pt]['elo'].iloc[-1],
        reverse=True,
    )
    non_vanilla = [pt for pt in player_types if pt != 'Vanilla']

    # Per-player-type charts
    # Filter out iterations where ANY player type in the model has < 5 game
    # (causes near-separation and unstable BT estimates for everyone)
    min_games_per_iter = results_df.groupby('game_index')['n_player_games'].min()
    stable_iters = min_games_per_iter[min_games_per_iter > 4].index
    plot_df = results_df[results_df['game_index'].isin(stable_iters)].copy()

    for pt in non_vanilla:
        pt_data = plot_df[plot_df['player_type'] == pt].copy()
        if pt_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))

        color = get_player_color(pt)
        ls = get_player_linestyle(pt)

        ax.plot(pt_data['n_player_games'], pt_data['elo'],
                color=color, linestyle=ls, linewidth=2, label=pt)
        ax.fill_between(
            pt_data['n_player_games'],
            pt_data['elo'] - 1.96 * pt_data['se_elo'],
            pt_data['elo'] + 1.96 * pt_data['se_elo'],
            color=color, alpha=0.15,
        )
        ax.axhline(1500, color='gray', linestyle='--', linewidth=1, label='Vanilla (1500)')

        # Draw vertical lines where new opponents were introduced
        if new_opponent_events:
            # Collect events for this player type, dedup by x-position
            pt_events = {}
            for ev_pt, ev_x, ev_names in new_opponent_events:
                if ev_pt == pt:
                    if ev_x not in pt_events:
                        pt_events[ev_x] = []
                    pt_events[ev_x].extend(ev_names)
            x_min = pt_data['n_player_games'].min()
            x_max = pt_data['n_player_games'].max()
            _, y_max = ax.get_ylim()
            for ev_x, names in sorted(pt_events.items()):
                if ev_x < x_min or ev_x > x_max:
                    continue
                unique_names = sorted(set(names))
                label = ' + '.join(unique_names)
                ax.axvline(ev_x, color='black', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(
                    ev_x, y_max, f' {label}',
                    rotation=90, va='top', ha='left',
                    fontsize=7, color='black', alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                )

        ax.set_xlabel('Number of Games (player participated)', fontsize=12)
        ax.set_ylabel('Elo Rating', fontsize=12)
        ax.set_title(f'Elo Rating Convergence: {pt}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = pt.replace(' ', '_').replace('/', '_')
        fig.savefig(os.path.join(output_dir, f'iterative_bt_{safe_name}.png'), dpi=150)
        plt.close(fig)

    # Combined chart
    fig, ax = plt.subplots(figsize=(14, 7))
    for pt in non_vanilla:
        pt_data = plot_df[plot_df['player_type'] == pt]
        if pt_data.empty:
            continue
        ax.plot(
            pt_data['n_player_games'], pt_data['elo'],
            color=get_player_color(pt),
            linestyle=get_player_linestyle(pt),
            linewidth=1.5,
            label=pt,
        )

    ax.axhline(1500, color='gray', linestyle='--', linewidth=1, label='Vanilla (1500)')
    ax.set_xlabel('Number of Games (player participated)', fontsize=12)
    ax.set_ylabel('Elo Rating', fontsize=12)
    ax.set_title('Elo Rating Convergence: All Player Types', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'iterative_bt_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved {len(non_vanilla) + 1} charts to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Iterative Bradley-Terry ratings')
    parser.add_argument('--no-cache', action='store_true', help='Ignore existing cache')
    parser.add_argument('--output-dir', default='ratings/output', help='Output directory')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, 'iterative_bt_cache.pkl')

    # Prepare data
    strength_df = prepare_strength_data()

    # Game ordering
    print("Determining game order...")
    game_order = load_game_order(strength_df)
    print(f"  {len(game_order)} games to process")

    # Load cache
    cache = {} if args.no_cache else load_cache(cache_path)
    cached_count = len(cache)
    if cached_count:
        print(f"  Loaded {cached_count} cached results")

    # Run iterative BT
    print("Running iterative Bradley-Terry...")
    results, cache, new_opponent_events = run_iterative_bt(strength_df, game_order, cache)

    # Save cache
    if len(cache) > cached_count:
        save_cache(cache, cache_path)
        print(f"  Cache saved ({len(cache)} entries)")

    if not results:
        print("No results produced. Exiting.")
        return

    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'iterative_bt_ratings.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved {len(results_df)} rows to {csv_path}")

    # Generate charts
    print("Generating charts...")
    generate_charts(results_df, output_dir, new_opponent_events)

    print("Done.")


if __name__ == '__main__':
    main()
