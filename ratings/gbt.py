"""
Player ratings via Uniform-GBT (Generalized Bradley-Terry).

Uses continuous pairwise strength comparisons (not ordinal rankings) to
estimate latent player scores.  The Uniform-GBT variant assumes comparison
noise follows a uniform distribution on [-1, 1].

Reference implementation
------------------------
Farhadkhani et al., "GBT: Generalized Bradley-Terry Model"
https://github.com/sadeghfarhadkhani/GBT/blob/main/code.ipynb

Every helper below is annotated with the exact function / section of the
reference notebook it corresponds to.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# GBT core functions
# (Ref: code.ipynb — "Loss Functions (phi functions)" section)
# ---------------------------------------------------------------------------

def _phi_uni(theta):
    """Conjugate function for the uniform distribution on [-1, 1].

    Ref: ``phi_uni`` in code.ipynb, "Loss Functions (phi functions)" section.
        phi_uni(theta) = log(sinh(theta) / theta)

    A Taylor expansion (theta^2 / 6) is used near zero to avoid 0/0.
    """
    # Guard against 0/0 in sinh(t)/t: torch.where evaluates both branches,
    # so autograd sees NaN gradients from the "unused" branch when theta ≈ 0.
    # Using a safe substitute (1.0) in that case avoids the NaN gradient.
    safe_theta = torch.where(theta.abs() < 1e-6, torch.ones_like(theta), theta)
    return torch.where(
        theta.abs() < 1e-6,
        theta * theta / 6,                         # Taylor: log(sinh(t)/t) ≈ t²/6
        torch.log(torch.sinh(safe_theta) / safe_theta),
    )


# ---------------------------------------------------------------------------
# GBT loss
# (Ref: code.ipynb — "GBT Loss and Optimization" section, ``loss_gbt``)
# ---------------------------------------------------------------------------

def _loss_gbt(scores, comparisons, alpha):
    """Compute the Uniform-GBT loss over all observed comparisons.

    Ref: ``loss_gbt`` in code.ipynb, "GBT Loss and Optimization" section.

        L = (alpha / 2) * ||scores||^2
            + sum_{(i,j): j>i, c_ij observed} [ phi(s_i - s_j) - c_ij * (s_i - s_j) ]

    Parameters
    ----------
    scores : Tensor (n,)
        Current latent scores for each item (player type).
    comparisons : Tensor (n, n)
        Pairwise comparison matrix.  ``comparisons[i, j]`` is the (normalised)
        strength advantage of item *i* over item *j*.  NaN = unobserved pair.
    alpha : float
        L2 regularisation strength.
    """
    # Ref: ``reg = alpha * torch.sum(scores**2) / 2`` in loss_gbt
    reg = alpha * (scores ** 2).sum() / 2

    # Ref: vectorised index selection in loss_gbt
    #   ``i, j = torch.where(~torch.isnan(comparisons))``
    #   ``valid_indices = j > i``
    i, j = torch.where(~torch.isnan(comparisons))
    upper = j > i
    i, j = i[upper], j[upper]

    # Ref: ``theta_ij = scores[i] - scores[j]``
    theta_ij = scores[i] - scores[j]

    # Ref: ``fit = torch.sum(phi(theta_ij, ...) - comparisons[i, j] * theta_ij)``
    fit = (_phi_uni(theta_ij) - comparisons[i, j] * theta_ij).sum()

    return reg + fit


# ---------------------------------------------------------------------------
# L-BFGS optimiser
# (Ref: code.ipynb — "GBT Loss and Optimization" section, ``compute_scores``)
# ---------------------------------------------------------------------------

def _compute_scores(comparisons, alpha, n_steps=50, lr=0.1):
    """Optimise GBT scores via L-BFGS.

    Ref: ``compute_scores`` in code.ipynb, "GBT Loss and Optimization" section.

    Parameters
    ----------
    comparisons : Tensor (n, n)
        Pairwise comparison matrix (NaN = missing).
    alpha : float
        Regularisation strength.
    n_steps : int
        Number of outer L-BFGS iterations.
    lr : float
        L-BFGS learning rate.

    Returns
    -------
    numpy array (n,) of optimised scores.
    """
    n = comparisons.shape[0]

    # Ref: ``scores = torch.normal(0, 1, (nb_items,), requires_grad=True, ...)``
    # We initialise at zero for determinism (small matrix, converges fast).
    scores = torch.zeros(n, requires_grad=True, dtype=torch.float64,
                         device=_DEVICE)

    # Ref: ``lbfgs = torch.optim.LBFGS((scores,))``
    optimizer = torch.optim.LBFGS([scores], lr=lr, max_iter=20)

    for _ in range(n_steps):
        # Ref: closure pattern used identically in compute_scores
        def closure():
            optimizer.zero_grad()
            loss = _loss_gbt(scores, comparisons, alpha)
            loss.backward()
            return loss
        optimizer.step(closure)

    return scores.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Comparison-matrix builder  (project-specific; no GBT-reference counterpart)
# ---------------------------------------------------------------------------

def _build_comparison_matrix(strength_df, _precomputed=None):
    """Construct an (n_types x n_types) pairwise comparison matrix from game data.

    For every pair of player types that co-occur in a game, the comparison
    value is the difference of their *adjusted_strength* values.  When a type
    appears more than once in a game (duplicate slots), its strengths are
    averaged first.  All per-game differences are then averaged across games
    and normalised to [-1, 1] (the support assumed by Uniform-GBT).

    Parameters
    ----------
    _precomputed : dict, optional
        Internal cache from ``_precompute_game_arrays`` to skip redundant
        pandas groupby work during bootstrap.

    Returns
    -------
    comparisons : Tensor (n, n)
    player_types : list[str]
    scale : float
        The divisor used to normalise into [-1, 1].  Needed to convert GBT
        scores back to the original strength scale.
    """
    if _precomputed is not None:
        player_types = _precomputed['player_types']
        n = len(player_types)
        # per_game_indices[g] and per_game_strengths[g] are arrays for game g
        per_game_indices = _precomputed['per_game_indices']
        per_game_strengths = _precomputed['per_game_strengths']
    else:
        player_types = sorted(strength_df['player_type'].unique())
        type_to_idx = {pt: i for i, pt in enumerate(player_types)}
        n = len(player_types)

        # Vectorised: average strength per (game_id, player_type)
        avg = (strength_df.groupby(['game_id', 'player_type'])['adjusted_strength']
               .mean().reset_index())
        avg['type_idx'] = avg['player_type'].map(type_to_idx)

        per_game_indices = []
        per_game_strengths = []
        for _, grp in avg.groupby('game_id'):
            per_game_indices.append(grp['type_idx'].values)
            per_game_strengths.append(grp['adjusted_strength'].values)

    diff_sum = np.zeros((n, n))
    diff_count = np.zeros((n, n))

    for idx_arr, str_arr in zip(per_game_indices, per_game_strengths):
        # Outer difference: diffs[a, b] = str_arr[a] - str_arr[b]
        diffs = str_arr[:, None] - str_arr[None, :]
        k = len(idx_arr)
        ix = np.ix_(idx_arr, idx_arr)
        diff_sum[ix] += diffs
        diff_count[ix] += 1

    # Zero out diagonal accumulations
    np.fill_diagonal(diff_sum, 0)
    np.fill_diagonal(diff_count, 0)

    # Mean difference per pair; NaN where no data
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_diff = np.where(diff_count > 0, diff_sum / diff_count, np.nan)

    # Normalise to [-1, 1] for the uniform-distribution assumption
    abs_max = np.nanmax(np.abs(mean_diff))
    scale = abs_max if abs_max > 0 else 1.0
    normalised = mean_diff / scale

    # Diagonal is self-comparison → NaN
    np.fill_diagonal(normalised, np.nan)

    comparisons = torch.tensor(normalised, dtype=torch.float64, device=_DEVICE)
    return comparisons, player_types, scale


def _precompute_game_arrays(strength_df, player_types):
    """Pre-index per-game type indices and strengths for fast bootstrap resampling."""
    type_to_idx = {pt: i for i, pt in enumerate(player_types)}
    avg = (strength_df.groupby(['game_id', 'player_type'])['adjusted_strength']
           .mean().reset_index())
    avg['type_idx'] = avg['player_type'].map(type_to_idx)

    game_ids = strength_df['game_id'].unique()
    game_id_to_pos = {gid: i for i, gid in enumerate(game_ids)}

    per_game_indices = [None] * len(game_ids)
    per_game_strengths = [None] * len(game_ids)
    for gid, grp in avg.groupby('game_id'):
        pos = game_id_to_pos[gid]
        per_game_indices[pos] = grp['type_idx'].values
        per_game_strengths[pos] = grp['adjusted_strength'].values

    return {
        'player_types': player_types,
        'game_ids': game_ids,
        'per_game_indices': per_game_indices,
        'per_game_strengths': per_game_strengths,
    }


# ---------------------------------------------------------------------------
# Bootstrap standard errors  (project-specific)
# ---------------------------------------------------------------------------

def _bootstrap_se(strength_df, player_types, alpha, n_bootstrap=200,
                  n_steps=50, lr=0.1):
    """Estimate standard errors by resampling games with replacement.

    For each bootstrap iteration the full pipeline (build matrix → fit GBT →
    re-centre on Vanilla) is repeated.  The SE is the std-dev of the centred
    scores across iterations.
    """
    precomputed = _precompute_game_arrays(strength_df, player_types)
    game_ids = precomputed['game_ids']
    n_games = len(game_ids)
    vanilla_idx = player_types.index('Vanilla')
    n_types = len(player_types)

    boot_scores = np.zeros((n_bootstrap, n_types))

    for b in range(n_bootstrap):
        sample_pos = np.random.randint(0, n_games, size=n_games)
        resampled = {
            'player_types': player_types,
            'per_game_indices': [precomputed['per_game_indices'][p] for p in sample_pos],
            'per_game_strengths': [precomputed['per_game_strengths'][p] for p in sample_pos],
        }

        comparisons, _, _ = _build_comparison_matrix(None, _precomputed=resampled)
        scores = _compute_scores(comparisons, alpha, n_steps=n_steps, lr=lr)
        scores -= scores[vanilla_idx]
        boot_scores[b] = scores

    return np.std(boot_scores, axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_ratings(strength_df, verbose=True, **kwargs):
    """Calculate player ratings using Uniform-GBT.

    Fits a Generalized Bradley-Terry model with the uniform-distribution
    conjugate (phi_uni) to continuous pairwise strength comparisons.  Uses
    raw adjusted_strength values — not ordinal rankings — so magnitude
    information is preserved.

    Ref: The overall pipeline mirrors the "Experiment 1" block in code.ipynb
    where ``compute_scores(comp_mat_c, alpha, phi_uni, ...)`` is called on a
    comparison matrix and scores are evaluated with ``error_metric``.

    Args:
        strength_df: DataFrame with columns: game_id, player_id, player_type,
                     adjusted_strength, civilization
        verbose: Print progress and results
        **kwargs:
            alpha (float): L2 regularisation, default 0.01
            n_bootstrap (int): bootstrap iterations for SEs, default 200
            n_steps (int): L-BFGS outer iterations, default 50
            lr (float): L-BFGS learning rate, default 0.1

    Returns:
        DataFrame with columns: player_type, worth, log_worth, se_log_worth,
                                z_value, p_value, elo, se_elo, mu, sigma
    """
    alpha = kwargs.get('alpha', 0.01)
    n_bootstrap = kwargs.get('n_bootstrap', 200)
    n_steps = kwargs.get('n_steps', 50)
    lr = kwargs.get('lr', 0.1)

    if verbose:
        print("=" * 76)
        print("CALCULATING RATINGS USING UNIFORM-GBT")
        print("=" * 76)
        print(f"\nTotal games: {strength_df['game_id'].nunique()}")
        print(f"Player types: {sorted(strength_df['player_type'].unique())}")
        print(f"Hyperparameters: alpha={alpha}, n_steps={n_steps}, "
              f"n_bootstrap={n_bootstrap}")
        print(f"\nBuilding comparison matrix...")

    # --- Step 1: Build comparison matrix ---
    # (Project-specific: aggregate per-game strength differences into a
    #  single n×n matrix, then normalise to [-1, 1] for Uniform-GBT.)
    comparisons, player_types, scale = _build_comparison_matrix(strength_df)

    if verbose:
        print(f"Comparison matrix: {len(player_types)}x{len(player_types)}, "
              f"scale={scale:.4f}")
        print("Fitting Uniform-GBT...")

    # --- Step 2: Compute scores ---
    # Ref: ``compute_scores(comp_mat_c, alpha, phi_uni, nb_steps=nb_steps)``
    #      in code.ipynb, "Experiment 1" section.
    scores = _compute_scores(comparisons, alpha, n_steps=n_steps, lr=lr)

    # --- Step 3: Re-centre on Vanilla ---
    vanilla_idx = player_types.index('Vanilla')
    scores_centred = scores - scores[vanilla_idx]

    if verbose:
        print("Bootstrapping standard errors...")

    # --- Step 4: Bootstrap SEs ---
    se_scores = _bootstrap_se(
        strength_df, player_types, alpha, n_bootstrap, n_steps, lr
    )

    # --- Step 5: Build output DataFrame ---
    results_df = pd.DataFrame({
        'player_type': player_types,
        'log_worth': scores_centred,
        'se_log_worth': se_scores,
        'worth': np.exp(scores_centred),
    })

    # z-value and p-value (H0: score = Vanilla)
    results_df['z_value'] = np.where(
        results_df['se_log_worth'] > 1e-12,
        results_df['log_worth'] / results_df['se_log_worth'],
        np.nan,
    )
    results_df['p_value'] = np.where(
        ~np.isnan(results_df['z_value']),
        2 * (1 - norm.cdf(np.abs(results_df['z_value']))),
        np.nan,
    )

    # Elo conversion: linear mapping centred at 1500 for Vanilla
    # 400 is the conventional Elo scale constant
    results_df['elo'] = 1500 + 400 * results_df['log_worth']
    results_df['se_elo'] = 400 * results_df['se_log_worth']

    # Backward-compatibility aliases
    results_df['mu'] = results_df['log_worth']
    results_df['sigma'] = results_df['se_log_worth']

    results_df = results_df.sort_values('elo', ascending=False)

    if verbose:
        print("\n" + "=" * 76)
        print("RATING SUMMARY (UNIFORM-GBT)")
        print("=" * 76)
        print(f"{'Rank':<6} {'Player Type':<25} {'Worth':<10} {'Score':<12} "
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

    return results_df
