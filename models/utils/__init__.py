"""
Utility modules for model evaluation and data processing.
"""

from .data_utils import (
    load_turn_data,
    apply_city_adjustments,
    add_relative_features,
    add_competitive_features,
    prepare_features,
    get_kfold_splits,
    load_and_prepare_data,
    FEATURE_GROUPS,
    get_all_available_features,
    get_feature_group
)

from .model_evaluator import (
    evaluate_fold,
    evaluate_by_turn_phase,
    aggregate_feature_importance,
    run_kfold_evaluation
)

from .model_registry import (
    get_model,
    list_models,
    register_model,
    MODEL_REGISTRY
)

__all__ = [
    # data_utils
    'load_turn_data',
    'apply_city_adjustments',
    'add_relative_features',
    'add_competitive_features',
    'prepare_features',
    'get_kfold_splits',
    'load_and_prepare_data',
    'FEATURE_GROUPS',
    'get_all_available_features',
    'get_feature_group',

    # model_evaluator
    'evaluate_fold',
    'evaluate_by_turn_phase',
    'aggregate_feature_importance',
    'run_kfold_evaluation',

    # model_registry
    'get_model',
    'list_models',
    'register_model',
    'MODEL_REGISTRY',
]
