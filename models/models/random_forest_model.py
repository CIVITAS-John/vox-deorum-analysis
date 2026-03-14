#!/usr/bin/env python3
"""
Random Forest model for victory probability prediction.
Example of how to extend the BasePredictor framework.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory for imports
from .base_predictor import BasePredictor


class RandomForestPredictor(BasePredictor):
    """
    Random Forest classifier for victory prediction.
    Demonstrates how to create a new model in the framework.
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
        max_depth: Optional[int] = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = 'sqrt',
        class_weight: str = 'balanced',
        calibrate: bool = False,
        calibration_method: str = 'isotonic'
    ):
        """
        Initialize Random Forest predictor.

        Args:
            include_features: Explicit list of features to include (None = all)
            exclude_features: List of features to exclude
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features per split ('sqrt', 'log2', float, or None for all)
            class_weight: Class balancing ('balanced' or None)
            calibrate: Whether to apply probability calibration (default: False)
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
        """
        super().__init__(include_features, exclude_features, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> 'RandomForestPredictor':
        """
        Fit Random Forest on training data.

        Note: Random Forest doesn't use cluster information directly,
        but the clusters parameter is kept for API compatibility.

        Args:
            X: Feature matrix
            y: Target vector (is_winner)
            clusters: Cluster IDs (not used by Random Forest)

        Returns:
            Self for chaining
        """
        # Apply feature filtering
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Initialize base Random Forest
        base_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
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
        Get feature importance based on Gini impurity reduction.

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
            'coefficient': importances,  # For consistency with baseline (though this is importance, not coefficient)
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

        # Get n_trees from base estimator if calibrated
        if self.calibrate:
            n_trees = len(self.model.calibrated_classifiers_[0].estimator.estimators_) if hasattr(
                self.model.calibrated_classifiers_[0].estimator, 'estimators_'
            ) else 0
        else:
            n_trees = len(self.model.estimators_) if hasattr(self.model, 'estimators_') else 0

        return {
            'model_type': 'RandomForestClassifier' + (' (Calibrated)' if self.calibrate else ''),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'class_weight': self.class_weight,
            'calibrated': self.calibrate,
            'calibration_method': self.calibration_method if self.calibrate else None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'n_trees': n_trees
        }
