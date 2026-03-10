"""Rating systems for player type evaluation.

Submodules:
    plackett_luce - Plackett-Luce MLE rating via R's PlackettLuce package
    matchups      - Empirical head-to-head matchup matrices and OLS validation
"""

from .plackett_luce import calculate_ratings
from .matchups import compare_with_ols, create_matchup_matrix, create_mean_matchup_matrix

__all__ = [
    'calculate_ratings',
    'compare_with_ols',
    'create_matchup_matrix',
    'create_mean_matchup_matrix',
]
