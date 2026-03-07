#!/usr/bin/env python3
"""
Multi-Layer Perceptron (MLP) neural network for victory prediction.
Uses sklearn's MLPClassifier with binary cross-entropy loss.
Supports optional post-hoc probability calibration via CalibratedClassifierCV.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory for imports
from .base_predictor import BasePredictor


class MLPPredictor(BasePredictor):
    """
    Multi-layer perceptron neural network for victory prediction.
    Supports optional probability calibration via CalibratedClassifierCV.
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
        hidden_layer_sizes: tuple = (73),
        activation: str = 'relu',
        alpha: float = 0.8368,  # L2 regularization
        learning_rate_init: float = 0.00526,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        solver: str = 'adam',
        batch_size: str = 1000,
        calibrate: bool = False,
        calibration_method: str = 'sigmoid'
    ):
        """
        Initialize MLP predictor.

        Args:
            include_features: Explicit list of features to include (None = all)
            exclude_features: List of features to exclude
            random_state: Random seed for reproducibility
            hidden_layer_sizes: Tuple of hidden layer sizes (e.g., (64, 32) = 2 layers)
            activation: Activation function ('relu', 'tanh', 'logistic')
            alpha: L2 regularization parameter
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of training iterations
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of training data for validation (if early_stopping=True)
            solver: Weight optimization solver ('adam', 'lbfgs', 'sgd')
            batch_size: Size of minibatches ('auto' or int)
            calibrate: Whether to apply post-hoc probability calibration
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
        """
        super().__init__(include_features, exclude_features, random_state)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.solver = solver
        self.batch_size = batch_size
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None) -> 'MLPPredictor':
        """
        Fit MLP on training data.

        Note: MLP doesn't use cluster information directly,
        but the clusters parameter is kept for API compatibility.

        Args:
            X: Feature matrix
            y: Target vector (is_winner)
            clusters: Cluster IDs (not used by MLP)

        Returns:
            Self for chaining
        """
        # Apply feature filtering
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        # Standardize features (critical for neural networks)
        X_scaled = self.scaler.fit_transform(X_filtered)

        # Initialize base MLP model
        # Note: MLPClassifier does NOT have a class_weight parameter.
        # For class imbalance handling, use resampling or post-hoc calibration.
        base_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            solver=self.solver,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        # Apply calibration if requested
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                base_model,
                method=self.calibration_method,
                cv=5,
                n_jobs=-1
            )
        else:
            self.model = base_model

        # Fit the model
        self.model.fit(X_scaled, y)

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
        X_scaled = self.scaler.transform(X_filtered)

        return self.model.predict_proba(X_scaled)

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
        X_scaled = self.scaler.transform(X_filtered)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance based on input layer weights.

        Returns average absolute weight from input to first hidden layer.
        This is a rough approximation - neural networks don't have clear feature importance.

        Returns:
            DataFrame with features ranked by importance (approximate)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if self.calibrate:
            # CalibratedClassifierCV has multiple base estimators (one per CV fold)
            # Average importance across all folds
            importances_list = []
            for calibrated_classifier in self.model.calibrated_classifiers_:
                base_estimator = calibrated_classifier.estimator
                input_weights = base_estimator.coefs_[0]
                importances_list.append(np.abs(input_weights).mean(axis=1))
            importances = np.mean(importances_list, axis=0)
        else:
            # Get weights from input layer to first hidden layer
            input_weights = self.model.coefs_[0]  # Shape: (n_features, first_hidden_size)
            # Average absolute weight across all neurons in first hidden layer
            importances = np.abs(input_weights).mean(axis=1)

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

        # Get n_iter and loss from base model (may be wrapped in calibrator)
        if self.calibrate:
            base = self.model.calibrated_classifiers_[0].estimator
        else:
            base = self.model

        return {
            'model_type': 'MLPClassifier' + (' (Calibrated)' if self.calibrate else ''),
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'solver': self.solver,
            'batch_size': self.batch_size,
            'calibrated': self.calibrate,
            'calibration_method': self.calibration_method if self.calibrate else None,
            'n_iter': base.n_iter_ if hasattr(base, 'n_iter_') else None,
            'loss': base.loss_ if hasattr(base, 'loss_') else None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'n_layers': base.n_layers_ if hasattr(base, 'n_layers_') else None,
            'n_outputs': base.n_outputs_ if hasattr(base, 'n_outputs_') else None
        }
