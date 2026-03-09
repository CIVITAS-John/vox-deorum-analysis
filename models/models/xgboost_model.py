#!/usr/bin/env python3
"""
XGBoost model with probability calibration for victory prediction.
Uses gradient boosting with calibrated probabilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import sys
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None
    CalibratedClassifierCV = None
    train_test_split = None

# Add parent directory for imports
from .base_predictor import BasePredictor


class XGBoostPredictor(BasePredictor):
    """
    XGBoost classifier with probability calibration for victory prediction.
    Handles class imbalance and provides well-calibrated probabilities.
    """

    # All numeric features are supported
    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    REQUIRED_FEATURES = None

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = [],
        random_state: int = 42,
        n_estimators: int = 50,
        max_depth: int = 6,
        learning_rate: float = 0.09961,
        subsample: float = 0.7568,
        colsample_bytree: float = 0.6302,
        min_child_weight: int = 2,
        gamma: float = 0.01063,
        reg_alpha: float = 0,
        reg_lambda: float = 0.1285,
        calibrate: bool = False,
        calibration_method: str = 'isotonic',
        early_stopping_rounds: Optional[int] = 10,
        eval_fraction: float = 0.1
    ):
        """
        Initialize XGBoost predictor.

        Args:
            include_features: Explicit list of features to include (None = all)
            exclude_features: List of features to exclude
            random_state: Random seed for reproducibility
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Step size shrinkage (eta)
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            min_child_weight: Minimum sum of instance weight in a child (controls overfitting)
            gamma: Minimum loss reduction required to make a split (tree pruning)
            reg_alpha: L1 regularization on leaf weights
            reg_lambda: L2 regularization on leaf weights
            calibrate: Whether to apply probability calibration
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
            early_stopping_rounds: Number of rounds with no improvement on validation
                set before stopping. None = disabled (use all n_estimators rounds).
                When calibrate=True, early stopping is used in a preliminary fit to
                determine optimal n_estimators, then the model is retrained with
                calibration using that value.
            eval_fraction: Fraction of training data to hold out as validation set
                for early stopping (default: 0.1). Only used when early_stopping_rounds
                is not None.
        """
        super().__init__(include_features, exclude_features, random_state)

        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost library is required for XGBoostPredictor. "
                "Install with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.best_iteration_ = None
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None) -> 'XGBoostPredictor':
        """
        Fit XGBoost on training data.

        When early_stopping_rounds is set:
        - If calibrate=False: splits data into train/val, fits with eval_set
          and early stopping, stores best_iteration_.
        - If calibrate=True: performs a preliminary fit with early stopping to
          find optimal n_estimators, then retrains with CalibratedClassifierCV
          using that n_estimators on the full training data.

        Args:
            X: Feature matrix
            y: Target vector (is_winner)
            clusters: Cluster IDs (not used by XGBoost)

        Returns:
            Self for chaining
        """
        # Apply feature filtering
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Calculate scale_pos_weight for class imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Common XGBoost parameters
        xgb_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        if self.early_stopping_rounds is not None:
            # Split off a validation set for early stopping
            X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                X_filtered, y,
                test_size=self.eval_fraction,
                random_state=self.random_state,
                stratify=y
            )

            # Fit with early stopping to find optimal n_estimators
            es_model = xgb.XGBClassifier(
                early_stopping_rounds=self.early_stopping_rounds,
                **xgb_params
            )
            es_model.fit(
                X_train_es, y_train_es,
                eval_set=[(X_val_es, y_val_es)],
                verbose=False
            )
            self.best_iteration_ = es_model.best_iteration

            if self.calibrate:
                # Two-stage: retrain with optimal n_estimators + calibration on full data
                optimal_params = xgb_params.copy()
                optimal_params['n_estimators'] = self.best_iteration_ + 1

                base_model = xgb.XGBClassifier(**optimal_params)
                self.model = CalibratedClassifierCV(
                    base_model,
                    method=self.calibration_method,
                    cv=5,
                    n_jobs=-1
                )
                self.model.fit(X_filtered, y)
            else:
                # Use the early-stopped model directly
                self.model = es_model
        else:
            # No early stopping -- original behavior
            self.best_iteration_ = None

            base_model = xgb.XGBClassifier(**xgb_params)

            if self.calibrate:
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
        Get feature importance from XGBoost.

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
            'model_type': 'XGBoostClassifier' + (' (Calibrated)' if self.calibrate else ''),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'calibrated': self.calibrate,
            'calibration_method': self.calibration_method if self.calibrate else None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'early_stopping_rounds': self.early_stopping_rounds,
            'best_iteration': self.best_iteration_,
            'effective_n_estimators': (self.best_iteration_ + 1) if self.best_iteration_ is not None else self.n_estimators,
        }
