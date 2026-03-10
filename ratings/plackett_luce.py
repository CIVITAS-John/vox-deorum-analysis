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
                                z_value, p_value, elo, se_elo, mu, sigma
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
