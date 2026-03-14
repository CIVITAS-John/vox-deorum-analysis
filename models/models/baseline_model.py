#!/usr/bin/env python3
"""
Baseline logistic regression model for victory probability prediction.
Includes cluster-robust standard errors to account for within-game correlation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from .base_predictor import BasePredictor
from utils.data_utils import get_selected_feature_names


class BaselineVictoryPredictor(BasePredictor):
    """
    Simple logistic regression model with standardized features.
    Uses statsmodels Logit for proper statistical inference.
    Computes cluster-robust standard errors to account for within-game correlation.
    """

    # All numeric features are supported (no specific restrictions)
    SUPPORTED_FEATURES = None  # None = all features supported
    # Baseline uses all features EXCEPT turn_progress (to avoid temporal confounding)
    DEFAULT_FEATURES = [f for f in get_selected_feature_names() if f != 'turn_progress']
    REQUIRED_FEATURES = None   # None = no required features

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize baseline logistic regression model.

        Args:
            include_features: Explicit list of features to include (None = all)
            exclude_features: List of features to exclude
            random_state: Random seed for reproducibility
        """
        super().__init__(include_features, exclude_features, random_state)
        self.scaler = StandardScaler()
        self.model_results = None  # Will store statsmodels LogitResults
        self.feature_names = None  # Will store feature names after filtering

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> 'BaselineVictoryPredictor':
        """
        Fit the model on training data.

        Args:
            X: Feature matrix
            y: Target vector (is_winner)
            clusters: Cluster IDs (e.g., game_id) for computing robust SEs

        Returns:
            Self for chaining
        """
        # Apply feature filtering
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X_filtered)

        # Add constant term (intercept)
        X_scaled_const = sm.add_constant(X_scaled)

        # Convert to arrays
        y_array = y.values if isinstance(y, pd.Series) else y

        # Fit logistic regression with statsmodels
        # Note: Unlike sklearn, statsmodels doesn't have a class_weight='balanced' parameter
        # For class balancing, we'd need to use freq_weights, but this complicates
        # cluster-robust SE calculation. Since we have a large sample, we proceed without
        # explicit class weighting - the cluster-robust SEs are more critical for inference
        logit_model = sm.Logit(y_array, X_scaled_const)

        # Fit with appropriate covariance type
        if clusters is not None:
            cluster_array = clusters.values if isinstance(clusters, pd.Series) else clusters
            try:
                # Try cluster-robust standard errors
                self.model_results = logit_model.fit(
                    disp=False,
                    method='bfgs',
                    maxiter=1000,
                    cov_type='cluster',
                    cov_kwds={'groups': cluster_array, 'use_correction': True}
                )
            except (np.linalg.LinAlgError, ValueError, AttributeError):
                # If cluster-robust fails, try HC1
                print("Warning: Cluster-robust SE calculation failed. "
                      "Falling back to HC1 robust standard errors.")
                try:
                    self.model_results = logit_model.fit(
                        disp=False,
                        method='bfgs',
                        maxiter=1000,
                        cov_type='HC1'
                    )
                except (np.linalg.LinAlgError, ValueError, AttributeError):
                    # If HC1 fails, use non-robust
                    print("Warning: All robust SE methods failed. Using non-robust standard errors.")
                    self.model_results = logit_model.fit(
                        disp=False,
                        method='bfgs',
                        maxiter=1000
                    )
        else:
            # No clusters, just fit normally
            self.model_results = logit_model.fit(disp=False, method='bfgs', maxiter=1000)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict victory probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with [P(loss), P(win)]
        """
        if self.model_results is None:
            raise ValueError("Model must be fitted before making predictions")

        # Use selected features from training
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")
        X_filtered = X[self.selected_features_]

        X_scaled = self.scaler.transform(X_filtered)
        # Add constant, ensuring it matches training shape
        X_scaled_const = sm.add_constant(X_scaled, has_constant='add')

        # Get predicted probabilities (P(y=1))
        probs_win = self.model_results.predict(X_scaled_const)

        # Return in sklearn format: [P(loss), P(win)]
        probs_loss = 1 - probs_win
        return np.column_stack([probs_loss, probs_win])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcomes (0 or 1).

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self, use_robust_se: bool = True) -> pd.DataFrame:
        """
        Get feature importance based on absolute coefficients.
        Larger absolute values = more influence on prediction.

        Args:
            use_robust_se: If True, include standard errors and CIs from statsmodels

        Returns:
            DataFrame with features ranked by importance
        """
        if self.model_results is None:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get coefficients (excluding intercept which is at index 0)
        coefficients = self.model_results.params[1:]  # Skip intercept

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # Add standard errors and confidence intervals from statsmodels
        if use_robust_se:
            # Standard errors (excluding intercept)
            importance_df['robust_se'] = self.model_results.bse[1:]

            # Confidence intervals (excluding intercept)
            conf_int = self.model_results.conf_int()
            importance_df['robust_ci_lower'] = conf_int[1:, 0]
            importance_df['robust_ci_upper'] = conf_int[1:, 1]

            # Z-statistics and p-values
            importance_df['z_statistic'] = self.model_results.tvalues[1:]
            importance_df['p_value'] = self.model_results.pvalues[1:]

            # Significance at 95% confidence level
            importance_df['significant_95'] = importance_df['p_value'] < 0.05

        # Sort by absolute value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the fitted model.

        Returns:
            Dictionary with model metadata
        """
        if self.model_results is None:
            raise ValueError("Model must be fitted before getting summary")

        return {
            'model_type': 'Logit (statsmodels)',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'intercept': float(self.model_results.params[0]),
            'converged': self.model_results.mle_retvals['converged'],
            'n_iterations': self.model_results.mle_retvals.get('iterations', 'N/A'),
            'log_likelihood': float(self.model_results.llf),
            'pseudo_r_squared': float(self.model_results.prsquared),
            'aic': float(self.model_results.aic),
            'bic': float(self.model_results.bic)
        }
