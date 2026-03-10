"""Rating systems for player type evaluation.

Submodules:
    gbt         - Uniform-GBT rating via continuous pairwise comparisons
    matchups    - Empirical head-to-head matchup matrices and OLS validation
"""

from .gbt import calculate_ratings
from .matchups import compare_with_ols, create_matchup_matrix, create_mean_matchup_matrix

__all__ = [
    'calculate_ratings',
    'compare_with_ols',
    'create_matchup_matrix',
    'create_mean_matchup_matrix',
]
