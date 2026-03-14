#!/usr/bin/env python3
"""
Model registry for victory prediction models.
Provides factory functions to instantiate models by name.
"""

from typing import Dict, Type, List
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.base_predictor import BasePredictor
from models.baseline_model import BaselineVictoryPredictor
from models.random_forest_model import RandomForestPredictor
from models.grouped_mlp_model import GroupedMLPPredictor
from models.naive_model import NaivePredictor

# Try to import new models (may fail if dependencies not installed)
try:
    from models.xgboost_model import XGBoostPredictor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBoostPredictor = None

try:
    from models.lightgbm_model import LightGBMPredictor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LightGBMPredictor = None

from models.mlp_model import MLPPredictor


# Registry mapping model names to classes
MODEL_REGISTRY: Dict[str, Type[BasePredictor]] = {
    'baseline': BaselineVictoryPredictor,
    'random_forest': RandomForestPredictor,
    'grouped_mlp': GroupedMLPPredictor,
    'naive': NaivePredictor,
}

# Add optional models if dependencies are available
if HAS_XGBOOST:
    MODEL_REGISTRY['xgboost'] = XGBoostPredictor

if HAS_LIGHTGBM:
    MODEL_REGISTRY['lightgbm'] = LightGBMPredictor

MODEL_REGISTRY['mlp'] = MLPPredictor


def get_model(name: str, **kwargs) -> BasePredictor:
    """
    Get a model instance by name.

    Args:
        name: Model name (must be in MODEL_REGISTRY)
        **kwargs: Keyword arguments to pass to model constructor

    Returns:
        Instantiated BasePredictor model

    Raises:
        ValueError: If model name is not recognized

    Example:
        >>> model = get_model('baseline', random_state=42)
        >>> model = get_model('baseline', include_features=['science_share', 'gold_share'])
        >>> model = get_model('baseline', exclude_features=['civ_*'])
    """
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    model_class = MODEL_REGISTRY[name_lower]
    return model_class(**kwargs)


def list_models() -> List[str]:
    """
    Get list of all available model names.

    Returns:
        List of model name strings
    """
    return sorted(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BasePredictor]) -> None:
    """
    Register a new model in the registry.

    Args:
        name: Model name (will be converted to lowercase)
        model_class: BasePredictor subclass

    Raises:
        TypeError: If model_class is not a BasePredictor subclass
        ValueError: If name is already registered

    Example:
        >>> from my_models import MyCustomPredictor
        >>> register_model('custom', MyCustomPredictor)
    """
    if not issubclass(model_class, BasePredictor):
        raise TypeError(f"{model_class.__name__} must be a subclass of BasePredictor")

    name_lower = name.lower()
    if name_lower in MODEL_REGISTRY:
        raise ValueError(f"Model name '{name}' is already registered")

    MODEL_REGISTRY[name_lower] = model_class
