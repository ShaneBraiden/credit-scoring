"""
Credit Risk Assessment - Utility Functions

This module contains reusable functions for:
- Data preprocessing
- Feature engineering
- Model evaluation
- Visualization
"""

from .preprocessing import (
    load_and_preprocess_data,
    handle_missing_values,
    engineer_features,
    prepare_features_and_target,
)

from .evaluation import (
    evaluate_model,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    compare_models,
)

__all__ = [
    "load_and_preprocess_data",
    "handle_missing_values",
    "engineer_features",
    "prepare_features_and_target",
    "evaluate_model",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "compare_models",
]
