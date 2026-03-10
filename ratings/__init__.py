"""Rating systems for player type evaluation.

Submodules:
    plackett_luce  - Plackett-Luce MLE rating via R's PlackettLuce package
    bradley_terry  - Bradley-Terry MLE rating with pairwise score weights via R's BradleyTerry2
    matchups       - Empirical head-to-head matchup matrices and OLS validation
"""

from .plackett_luce import calculate_ratings
from .bradley_terry import calculate_ratings as calculate_ratings_bt
from .matchups import compare_with_ols, create_matchup_matrix, create_mean_matchup_matrix

__all__ = [
    'calculate_ratings',
    'calculate_ratings_bt',
    'compare_with_ols',
    'create_matchup_matrix',
    'create_mean_matchup_matrix',
]
