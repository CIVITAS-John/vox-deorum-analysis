#!/usr/bin/env python3
"""
Quick test to verify which features are being used by each model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.model_registry import get_model
from utils.data_utils import load_turn_data, apply_city_adjustments, add_relative_features, add_competitive_features, prepare_features

# Load and prepare data
print("Loading data...")
df = load_turn_data("../turn_data.csv")[:1000]  # Just use first 1000 rows for quick test
df = apply_city_adjustments(df)
df = add_relative_features(df)
df = add_competitive_features(df)
X, y = prepare_features(df)

print(f"\nTotal available features: {len(X.columns)}")
print(f"Features: {list(X.columns)}")

# Test baseline model
print("\n" + "="*50)
print("BASELINE MODEL")
print("="*50)
baseline = get_model('baseline')
X_filtered = baseline._filter_features(X)
print(f"Features used: {len(X_filtered.columns)}")
print(f"Has turn_progress: {'turn_progress' in X_filtered.columns}")
print(f"Features: {list(X_filtered.columns)}")

# Test random forest model
print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)
rf = get_model('random_forest')
X_filtered = rf._filter_features(X)
print(f"Features used: {len(X_filtered.columns)}")
print(f"Has turn_progress: {'turn_progress' in X_filtered.columns}")
print(f"Features: {list(X_filtered.columns)}")

# Test xgboost model (if available)
try:
    print("\n" + "="*50)
    print("XGBOOST MODEL")
    print("="*50)
    xgb = get_model('xgboost')
    X_filtered = xgb._filter_features(X)
    print(f"Features used: {len(X_filtered.columns)}")
    print(f"Has turn_progress: {'turn_progress' in X_filtered.columns}")
    print(f"Features: {list(X_filtered.columns)}")
except ValueError:
    print("XGBoost not available")

print("\n✅ Test complete!")