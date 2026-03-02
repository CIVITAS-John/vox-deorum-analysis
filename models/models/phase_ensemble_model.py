#!/usr/bin/env python3
"""
Phase-based ensemble predictor that trains separate models for each game phase.
Can work with any BasePredictor-compatible model as the base learner.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Type
from .base_predictor import BasePredictor


class PhaseEnsemblePredictor(BasePredictor):
    """
    Ensemble that trains separate models for different game phases.

    This allows the model to learn phase-specific patterns, as game dynamics
    often shift significantly between early, mid, and late game.
    """

    # Flag to indicate this model needs full game data
    NEEDS_FULL_DATA = True

    def __init__(
        self,
        base_model_class: Type[BasePredictor],
        phase_boundaries: List[float] = [0.66, 0.82],
        blend_width: float = 0,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        **base_model_kwargs
    ):
        """
        Initialize phase ensemble predictor.

        Args:
            base_model_class: Model class to use for each phase (e.g., BaselinePredictor)
            phase_boundaries: List of N boundaries creating N+1 phases
                             e.g., [0.33, 0.66] creates 3 phases: early, mid, late
            blend_width: Width for smooth blending at phase boundaries (0 = hard switch)
                        e.g., 0.05 means blend ±5% around each boundary
            include_features: Features to include (passed to base models)
            exclude_features: Features to exclude (passed to base models)
            random_state: Random seed for reproducibility
            **base_model_kwargs: Additional arguments passed to each phase model
        """
        super().__init__(include_features, exclude_features, random_state)

        self.base_model_class = base_model_class
        self.phase_boundaries = sorted(phase_boundaries)  # Ensure sorted
        self.blend_width = blend_width
        self.base_model_kwargs = base_model_kwargs
        self.n_phases = len(phase_boundaries) + 1
        self.phase_models = []
        self.phase_sample_counts = []

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None) -> 'PhaseEnsemblePredictor':
        """
        Fit separate models for each game phase.

        Args:
            X: Feature matrix (must include turn_progress)
            y: Target vector
            clusters: Cluster IDs for grouped operations

        Returns:
            Self for chaining
        """
        # Ensure turn_progress is available
        if 'turn_progress' not in X.columns:
            raise ValueError("turn_progress column is required for PhaseEnsemblePredictor")

        # Store selected features from first phase for consistency
        self.selected_features_ = None

        # Train model for each phase
        self.phase_models = []
        self.phase_sample_counts = []

        for phase_idx in range(self.n_phases):
            # Get phase data
            phase_mask = self._get_phase_mask(X['turn_progress'], phase_idx)
            X_phase = X[phase_mask]
            y_phase = y[phase_mask]
            # Handle clusters with proper indexing alignment
            if clusters is not None:
                # Ensure clusters has same index as X for proper masking
                clusters_aligned = clusters.copy()
                clusters_aligned.index = X.index
                clusters_phase = clusters_aligned[phase_mask]
            else:
                clusters_phase = None

            # Track sample counts for debugging/analysis
            self.phase_sample_counts.append(len(y_phase))

            if len(y_phase) == 0:
                print(f"Warning: No samples in phase {phase_idx} (boundaries: {self.phase_boundaries})")
                # Create a dummy model that always predicts the overall mean
                self.phase_models.append(None)
                continue

            # Create and train phase model
            model = self.base_model_class(
                include_features=self.include_features,
                exclude_features=self.exclude_features,
                random_state=self.random_state + phase_idx,  # Different seed per phase
                **self.base_model_kwargs
            )
            model.fit(X_phase, y_phase, clusters_phase)
            self.phase_models.append(model)

            # Store selected features from first successful model
            if self.selected_features_ is None and hasattr(model, 'selected_features_'):
                self.selected_features_ = model.selected_features_

        # Validate at least one model was trained
        if all(m is None for m in self.phase_models):
            raise ValueError("No phases had sufficient data for training")

        print(f"Phase ensemble trained with {self.n_phases} phases:")
        for i, count in enumerate(self.phase_sample_counts):
            phase_range = self._get_phase_range(i)
            print(f"  Phase {i} ({phase_range[0]:.0%}-{phase_range[1]:.0%}): {count} samples")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using appropriate phase model(s).

        Args:
            X: Feature matrix (must include turn_progress)

        Returns:
            Array of shape (n_samples, 2) with [P(loss), P(win)]
        """
        if not self.phase_models:
            raise ValueError("Model must be fitted before making predictions")

        if 'turn_progress' not in X.columns:
            raise ValueError("turn_progress column is required for prediction")

        n_samples = len(X)
        probas = np.zeros((n_samples, 2))

        if self.blend_width == 0:
            # Hard boundaries - process in batches by phase for efficiency
            for phase_idx in range(self.n_phases):
                if self.phase_models[phase_idx] is None:
                    continue

                # Get samples belonging to this phase
                phase_mask = self._get_phase_mask_hard(X['turn_progress'], phase_idx)
                if not phase_mask.any():
                    continue

                X_phase = X[phase_mask]
                phase_probas = self.phase_models[phase_idx].predict_proba(X_phase)
                probas[phase_mask] = phase_probas
        else:
            # Soft boundaries - need to handle blending
            # Group samples by their blending requirements
            boundaries = [0.0] + self.phase_boundaries + [1.0]

            for phase_idx in range(self.n_phases):
                if self.phase_models[phase_idx] is None:
                    continue

                # Process samples clearly in this phase (not near boundaries)
                phase_lower = boundaries[phase_idx]
                phase_upper = boundaries[phase_idx + 1]

                # Safe zone: not near any boundary
                safe_lower = phase_lower + (self.blend_width if phase_idx > 0 else 0)
                safe_upper = phase_upper - (self.blend_width if phase_idx < self.n_phases - 1 else 0)

                if safe_lower < safe_upper:
                    safe_mask = (X['turn_progress'] >= safe_lower) & (X['turn_progress'] < safe_upper)
                    if safe_mask.any():
                        X_safe = X[safe_mask]
                        probas[safe_mask] = self.phase_models[phase_idx].predict_proba(X_safe)

            # Process samples in blending zones
            for boundary_idx, boundary in enumerate(self.phase_boundaries):
                blend_mask = (X['turn_progress'] >= boundary - self.blend_width) & \
                            (X['turn_progress'] < boundary + self.blend_width)

                if not blend_mask.any():
                    continue

                X_blend = X[blend_mask]
                blend_indices = np.where(blend_mask)[0]

                for i, (idx, row) in enumerate(X_blend.iterrows()):
                    turn_prog = row['turn_progress']
                    dist_to_boundary = turn_prog - boundary

                    if dist_to_boundary < 0:
                        # Before boundary - blend phase_idx and phase_idx-1
                        blend_ratio = (turn_prog - (boundary - self.blend_width)) / self.blend_width
                        if self.phase_models[boundary_idx] is not None:
                            prob1 = self.phase_models[boundary_idx].predict_proba(X_blend.iloc[[i]])
                            probas[blend_indices[i]] += (1 - blend_ratio) * prob1[0]
                        if self.phase_models[boundary_idx + 1] is not None:
                            prob2 = self.phase_models[boundary_idx + 1].predict_proba(X_blend.iloc[[i]])
                            probas[blend_indices[i]] += blend_ratio * prob2[0]
                    else:
                        # After boundary - blend phase_idx+1 and phase_idx
                        blend_ratio = (boundary + self.blend_width - turn_prog) / self.blend_width
                        if self.phase_models[boundary_idx] is not None:
                            prob1 = self.phase_models[boundary_idx].predict_proba(X_blend.iloc[[i]])
                            probas[blend_indices[i]] += blend_ratio * prob1[0]
                        if self.phase_models[boundary_idx + 1] is not None:
                            prob2 = self.phase_models[boundary_idx + 1].predict_proba(X_blend.iloc[[i]])
                            probas[blend_indices[i]] += (1 - blend_ratio) * prob2[0]

        return probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcomes using phase models.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0 = loss, 1 = win)
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get aggregated feature importance across all phases.

        Returns:
            DataFrame with phase-specific and average importance
        """
        if not self.phase_models:
            return None

        importance_dfs = []

        for phase_idx, model in enumerate(self.phase_models):
            if model is None:
                continue

            imp_df = model.get_feature_importance()
            if imp_df is None:
                continue

            # Add phase information
            phase_range = self._get_phase_range(phase_idx)
            phase_label = f"phase_{phase_idx}_{phase_range[0]:.0%}_{phase_range[1]:.0%}"
            imp_df[f'importance_{phase_label}'] = imp_df['coefficient']
            imp_df['phase'] = phase_idx
            imp_df['n_samples'] = self.phase_sample_counts[phase_idx]

            importance_dfs.append(imp_df)

        if not importance_dfs:
            return None

        # Merge all phase importances
        combined = importance_dfs[0][['feature']].copy()

        for phase_idx, imp_df in enumerate(importance_dfs):
            phase_range = self._get_phase_range(imp_df['phase'].iloc[0])
            phase_label = f"phase_{imp_df['phase'].iloc[0]}_{phase_range[0]:.0%}_{phase_range[1]:.0%}"
            col_name = f'importance_{phase_label}'
            if col_name in imp_df.columns:
                combined = combined.merge(
                    imp_df[['feature', col_name]],
                    on='feature',
                    how='outer'
                )

        # Calculate weighted average importance
        total_samples = sum(self.phase_sample_counts)
        combined['coefficient'] = 0

        for phase_idx in range(self.n_phases):
            if self.phase_sample_counts[phase_idx] > 0:
                phase_range = self._get_phase_range(phase_idx)
                phase_label = f"phase_{phase_idx}_{phase_range[0]:.0%}_{phase_range[1]:.0%}"
                col_name = f'importance_{phase_label}'
                if col_name in combined.columns:
                    weight = self.phase_sample_counts[phase_idx] / total_samples
                    combined['coefficient'] += combined[col_name].fillna(0) * weight

        # Add absolute coefficient for sorting
        combined['abs_coefficient'] = combined['coefficient'].abs()
        combined = combined.sort_values('abs_coefficient', ascending=False)

        return combined

    def get_model_summary(self) -> dict:
        """
        Get summary statistics about the fitted ensemble.

        Returns:
            Dictionary with ensemble metadata
        """
        if not self.phase_models:
            return {'status': 'not_fitted'}

        # Get summary from first non-None model
        base_summary = {}
        for model in self.phase_models:
            if model is not None:
                base_summary = model.get_model_summary()
                break

        return {
            'model_type': f'PhaseEnsemble({self.base_model_class.__name__})',
            'n_phases': self.n_phases,
            'phase_boundaries': self.phase_boundaries,
            'blend_width': self.blend_width,
            'phase_sample_counts': self.phase_sample_counts,
            'base_model_type': self.base_model_class.__name__,
            'base_model_params': self.base_model_kwargs,
            **{f'base_{k}': v for k, v in base_summary.items() if k != 'model_type'}
        }

    def _get_phase_mask_hard(self, turn_progress: pd.Series, phase_idx: int) -> np.ndarray:
        """
        Get boolean mask for samples belonging to a phase (hard boundaries).

        Args:
            turn_progress: Series of turn progress values
            phase_idx: Phase index (0 to n_phases-1)

        Returns:
            Boolean mask
        """
        boundaries = [0.0] + self.phase_boundaries + [1.0]
        lower = boundaries[phase_idx]
        upper = boundaries[phase_idx + 1]
        return (turn_progress >= lower) & (turn_progress < upper)

    def _get_phase_mask(self, turn_progress: pd.Series, phase_idx: int) -> np.ndarray:
        """
        Get boolean mask for samples belonging to a phase.

        Args:
            turn_progress: Series of turn progress values
            phase_idx: Phase index (0 to n_phases-1)

        Returns:
            Boolean mask
        """
        boundaries = [0.0] + self.phase_boundaries + [1.0]
        lower = boundaries[phase_idx]
        upper = boundaries[phase_idx + 1]

        if self.blend_width > 0:
            # Extend phase boundaries for blending
            if phase_idx > 0:
                lower -= self.blend_width
            if phase_idx < self.n_phases - 1:
                upper += self.blend_width

        return (turn_progress >= lower) & (turn_progress < upper)

    def _get_phase_range(self, phase_idx: int) -> tuple:
        """Get the turn_progress range for a phase."""
        boundaries = [0.0] + self.phase_boundaries + [1.0]
        return (boundaries[phase_idx], boundaries[phase_idx + 1])

    def _get_phase_weights(self, turn_progress: float) -> np.ndarray:
        """
        Get blending weights for each phase model at a specific turn_progress.

        Args:
            turn_progress: Single turn_progress value

        Returns:
            Array of weights for each phase (sums to 1.0)
        """
        weights = np.zeros(self.n_phases)

        if self.blend_width == 0:
            # Hard boundaries - single phase gets weight 1.0
            phase_idx = self._get_phase_index(turn_progress)
            weights[phase_idx] = 1.0
            return weights

        # Find which phase(s) this point belongs to
        boundaries = [0.0] + self.phase_boundaries + [1.0]

        # Determine primary phase
        phase_idx = self._get_phase_index(turn_progress)

        # Check if near lower boundary
        if phase_idx > 0:
            lower_boundary = boundaries[phase_idx]
            dist_to_lower = turn_progress - lower_boundary
            if 0 <= dist_to_lower < self.blend_width:
                # Blend with previous phase
                blend_ratio = dist_to_lower / self.blend_width
                weights[phase_idx - 1] = 1.0 - blend_ratio
                weights[phase_idx] = blend_ratio
                return weights

        # Check if near upper boundary
        if phase_idx < self.n_phases - 1:
            upper_boundary = boundaries[phase_idx + 1]
            dist_to_upper = upper_boundary - turn_progress
            if 0 <= dist_to_upper < self.blend_width:
                # Blend with next phase
                blend_ratio = dist_to_upper / self.blend_width
                weights[phase_idx] = blend_ratio
                weights[phase_idx + 1] = 1.0 - blend_ratio
                return weights

        # Not near any boundary
        weights[phase_idx] = 1.0
        return weights

    def _get_sample_weights(self, turn_progress_values: np.ndarray, phase_idx: int) -> np.ndarray:
        """
        Get weights for samples when processing a specific phase.
        Used for smooth blending at boundaries.

        Args:
            turn_progress_values: Array of turn_progress values
            phase_idx: Current phase being processed

        Returns:
            Array of weights for each sample
        """
        if self.blend_width == 0:
            return np.ones(len(turn_progress_values))

        weights = np.ones(len(turn_progress_values))
        boundaries = [0.0] + self.phase_boundaries + [1.0]

        # Check lower boundary
        if phase_idx > 0:
            lower_boundary = boundaries[phase_idx]
            lower_dist = turn_progress_values - lower_boundary
            in_lower_blend = (lower_dist < 0) & (lower_dist > -self.blend_width)
            weights[in_lower_blend] = 1.0 + (lower_dist[in_lower_blend] / self.blend_width)

        # Check upper boundary
        if phase_idx < self.n_phases - 1:
            upper_boundary = boundaries[phase_idx + 1]
            upper_dist = upper_boundary - turn_progress_values
            in_upper_blend = (upper_dist < 0) & (upper_dist > -self.blend_width)
            weights[in_upper_blend] = 1.0 + (upper_dist[in_upper_blend] / self.blend_width)

        return weights

    def _get_phase_index(self, turn_progress: float) -> int:
        """
        Get the phase index for a single turn_progress value.

        Args:
            turn_progress: Turn progress value (0 to 1)

        Returns:
            Phase index (0 to n_phases-1)
        """
        for i, boundary in enumerate(self.phase_boundaries):
            if turn_progress < boundary:
                return i
        return self.n_phases - 1