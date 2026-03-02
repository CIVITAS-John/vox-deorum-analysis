#!/usr/bin/env python3
"""
LightGBM model with probability calibration for victory prediction.
Uses gradient boosting with leaf-wise growth for faster training.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import sys
from pathlib import Path

try:
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None 
    CalibratedClassifierCV = None

# Add parent directory for imports
from .base_predictor import BasePredictor


class LightGBMPredictor(BasePredictor):
    """
    LightGBM classifier with probability calibration for victory prediction.
    Faster than XGBoost with similar performance.
    """

    # All numeric features are supported
    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    REQUIRED_FEATURES = None

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = -1,  # -1 = no limit
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        calibrate: bool = True,
        calibration_method: str = 'isotonic'
    ):
        """
        Initialize LightGBM predictor.

        Args:
            include_features: Explicit list of features to include (None = all)
            exclude_features: List of features to exclude
            random_state: Random seed for reproducibility
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees (-1 = no limit)
            learning_rate: Step size shrinkage
            num_leaves: Maximum number of leaves per tree
            subsample: Fraction of samples for each tree (bagging_fraction)
            colsample_bytree: Fraction of features for each tree (feature_fraction)
            calibrate: Whether to apply probability calibration
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
        """
        super().__init__(include_features, exclude_features, random_state)

        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm library is required for LightGBMPredictor. "
                "Install with: pip install lightgbm"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None) -> 'LightGBMPredictor':
        """
        Fit LightGBM on training data.

        Note: LightGBM doesn't use cluster information directly,
        but the clusters parameter is kept for API compatibility.

        Args:
            X: Feature matrix
            y: Target vector (is_winner)
            clusters: Cluster IDs (not used by LightGBM)

        Returns:
            Self for chaining
        """
        # Apply feature filtering
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Initialize base LightGBM model
        base_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            subsample_freq=1,  # Enable bagging
            colsample_bytree=self.colsample_bytree,
            is_unbalance=True,  # Handle class imbalance automatically
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=-1  # Suppress warnings
        )

        # Apply calibration if requested
        if self.calibrate:
            # Use CalibratedClassifierCV with 5-fold CV for calibration
            self.model = CalibratedClassifierCV(
                base_model,
                method=self.calibration_method,
                cv=5,
                n_jobs=-1
            )
        else:
            self.model = base_model

        # Fit the model
        self.model.fit(X_filtered, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict victory probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 2) with [P(loss), P(win)]
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Use selected features from training
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")

        X_filtered = X[self.selected_features_]
        return self.model.predict_proba(X_filtered)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcomes (0 or 1).

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0 = loss, 1 = win)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        X_filtered = X[self.selected_features_]
        return self.model.predict(X_filtered)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from LightGBM.

        For calibrated models, returns importance from the base estimator.

        Returns:
            DataFrame with features ranked by importance
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get base estimator if calibrated
        if self.calibrate:
            # CalibratedClassifierCV has multiple base estimators (one per CV fold)
            # Average importance across all folds
            importances_list = []
            for calibrated_classifier in self.model.calibrated_classifiers_:
                base_estimator = calibrated_classifier.estimator
                importances_list.append(base_estimator.feature_importances_)

            # Average importance across folds
            importances = np.mean(importances_list, axis=0)
        else:
            importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': importances,
            'abs_coefficient': np.abs(importances)
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    def get_model_summary(self) -> dict:
        """
        Get summary statistics about the fitted model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")

        return {
            'model_type': 'LGBMClassifier' + (' (Calibrated)' if self.calibrate else ''),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'calibrated': self.calibrate,
            'calibration_method': self.calibration_method if self.calibrate else None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }
