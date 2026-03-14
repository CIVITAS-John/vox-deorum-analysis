#!/usr/bin/env python3
"""
Compare multiple models side-by-side.

Usage:
    python compare_models.py --models baseline,random_forest
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from utils.model_registry import list_models, MODEL_REGISTRY
from utils.model_evaluator import run_kfold_evaluation
from utils.data_utils import load_and_prepare_base_data


def parse_comma_separated(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into list."""
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple victory prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help=f"Comma-separated model names. Available: {', '.join(list_models())}"
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default='../turn_data.csv',
        help="Path to turn data CSV (default: ../turn_data.csv)"
    )

    parser.add_argument(
        '--experiments',
        type=str,
        default=None,
        help="Comma-separated list of experiments to include (default: all)"
    )

    # CV arguments
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help="Number of k-fold CV splits (default: 5)"
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Resampling
    parser.add_argument(
        '--resample',
        type=str,
        default='none',
        choices=['none', 'oversample', 'undersample', 'combined'],
        help="Resampling method for class imbalance (default: none)"
    )

    # Data filtering
    parser.add_argument(
        '--full-data',
        action='store_true',
        help="Use all turn data for training (no late-game filtering)"
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help="Directory to save results (default: output/)"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse model names
    model_names = parse_comma_separated(args.models)
    if not model_names:
        print("Error: No models specified", file=sys.stderr)
        sys.exit(1)

    # Validate model names
    available = list_models()
    for name in model_names:
        if name.lower() not in available:
            print(f"Error: Unknown model '{name}'. Available: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)

    # Parse experiments
    filter_experiments = parse_comma_separated(args.experiments)

    # Convert 'none' to None for resample method
    resample_method = None if args.resample == 'none' else args.resample

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"Models:      {', '.join(model_names)}")
    print(f"Data:        {args.data}")
    print(f"Experiments: {filter_experiments if filter_experiments else 'all'}")
    print(f"CV Splits:   {args.n_splits}")
    print(f"Resampling:  {args.resample}")
    print(f"Full Data:   {args.full_data}")
    print(f"Output Dir:  {output_path}")
    print("=" * 80 + "\n")

    # Preload data once - shared across all model evaluations
    preloaded_df = load_and_prepare_base_data(
        args.data, filter_experiments=filter_experiments
    )

    # Run evaluations
    results = []

    for model_name in model_names:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'=' * 80}\n")

        model_class = MODEL_REGISTRY[model_name.lower()]

        try:
            summary, importance, models = run_kfold_evaluation(
                model_class=model_class,
                csv_path=args.data,
                filter_experiments=filter_experiments,
                n_splits=args.n_splits,
                random_state=args.random_state,
                verbose=True,
                resample_method=resample_method,
                full_data=args.full_data,
                preloaded_df=preloaded_df,
            )

            # Extract key metrics
            result = {
                'model': model_name,
                'roc_auc_mean': summary['roc_auc_mean'],
                'roc_auc_std': summary['roc_auc_std'],
                'brier_mean': summary['brier_score_mean'],
                'brier_std': summary['brier_score_std'],
                'log_loss_mean': summary['log_loss_mean'],
                'log_loss_std': summary['log_loss_std'],
                'balanced_accuracy_mean': summary['balanced_accuracy_mean'],
                'balanced_accuracy_std': summary['balanced_accuracy_std'],
                'n_features': len(models[0].get_selected_features()) if models else 0
            }
            results.append(result)

        except Exception as e:
            print(f"\nError evaluating {model_name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    # Create comparison table
    if not results:
        print("\nNo models successfully evaluated.", file=sys.stderr)
        sys.exit(1)

    comparison_df = pd.DataFrame(results)

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80 + "\n")

    # Format for display
    display_df = comparison_df.copy()
    display_df['ROC-AUC'] = display_df.apply(
        lambda x: f"{x['roc_auc_mean']:.4f} ± {x['roc_auc_std']:.4f}", axis=1
    )
    display_df['Brier (strat.)'] = display_df.apply(
        lambda x: f"{x['brier_mean']:.4f} ± {x['brier_std']:.4f}", axis=1
    )
    display_df['Log Loss (strat.)'] = display_df.apply(
        lambda x: f"{x['log_loss_mean']:.4f} ± {x['log_loss_std']:.4f}", axis=1
    )
    display_df['Balanced Accuracy'] = display_df.apply(
        lambda x: f"{x['balanced_accuracy_mean']:.4f} ± {x['balanced_accuracy_std']:.4f}", axis=1
    )

    # Use tabulate if available, otherwise fall back to pandas
    table_df = display_df[['model', 'ROC-AUC', 'Brier (strat.)', 'Log Loss (strat.)', 'Balanced Accuracy', 'n_features']]
    if HAS_TABULATE:
        print(tabulate(table_df, headers='keys', tablefmt='simple', showindex=False))
    else:
        print(table_df.to_string(index=False))

    # Save to CSV
    output_file = output_path / 'model_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComparison saved to: {output_file}")

    # Determine best model per metric
    print("\n" + "=" * 80)
    print("BEST MODELS BY METRIC")
    print("=" * 80)
    print(f"Highest ROC-AUC:            {comparison_df.loc[comparison_df['roc_auc_mean'].idxmax(), 'model']} "
          f"({comparison_df['roc_auc_mean'].max():.4f})")
    print(f"Lowest Brier (strat.):      {comparison_df.loc[comparison_df['brier_mean'].idxmin(), 'model']} "
          f"({comparison_df['brier_mean'].min():.4f})")
    print(f"Lowest Log Loss (strat.):   {comparison_df.loc[comparison_df['log_loss_mean'].idxmin(), 'model']} "
          f"({comparison_df['log_loss_mean'].min():.4f})")
    print(f"Highest Balanced Accuracy:  {comparison_df.loc[comparison_df['balanced_accuracy_mean'].idxmax(), 'model']} "
          f"({comparison_df['balanced_accuracy_mean'].max():.4f})")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
