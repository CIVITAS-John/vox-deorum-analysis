#!/usr/bin/env python3
"""
CLI script for evaluating victory prediction models with k-fold cross-validation.

Usage:
    python evaluate_model.py --model baseline
    python evaluate_model.py --model baseline --exclude-features civ_*
    python evaluate_model.py --model baseline --include-features science_share,gold_share
    python evaluate_model.py --model baseline --experiments "2026-staff-standard,2026-oss-v-sonnet-standard"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from utils.model_registry import get_model, list_models
from utils.model_evaluator import run_kfold_evaluation, run_full_prediction


def parse_comma_separated(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into list."""
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate victory prediction models with k-fold cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate baseline model with all features
  python evaluate_model.py --model baseline

  # Exclude civilization features
  python evaluate_model.py --model baseline --exclude-features "civ_*"

  # Include only economic features
  python evaluate_model.py --model baseline --include-features "science_share,gold_share,culture_share"

  # Filter to specific experiments
  python evaluate_model.py --model baseline --experiments "2026-staff-standard,2026-oss-v-sonnet-standard"

  # Save results to custom directory
  python evaluate_model.py --model baseline --output-dir results/baseline_full

  # Prediction mode: train on all data and output predictions CSV
  python evaluate_model.py --model baseline --predict
        """
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help=f"Model name. Available: {', '.join(list_models())}"
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

    # Cross-validation arguments
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

    # Feature filtering arguments
    parser.add_argument(
        '--include-features',
        type=str,
        default=None,
        help="Comma-separated list of features to include (supports wildcards like 'civ_*')"
    )

    parser.add_argument(
        '--exclude-features',
        type=str,
        default=None,
        help="Comma-separated list of features to exclude (supports wildcards)"
    )

    # Resampling
    parser.add_argument(
        '--resample',
        type=str,
        default='none',
        choices=['none', 'oversample', 'undersample', 'combined'],
        help="Resampling method for class imbalance (default: none)"
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help="Directory to save results (default: output/, feature importance always saved)"
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output"
    )

    # Prediction mode
    parser.add_argument(
        '--predict',
        action='store_true',
        help="Prediction mode: train on all data and output predictions CSV (no cross-validation)"
    )

    args = parser.parse_args()

    # Parse comma-separated arguments
    filter_experiments = parse_comma_separated(args.experiments)
    include_features = parse_comma_separated(args.include_features)
    exclude_features = parse_comma_separated(args.exclude_features)

    # Prepare model kwargs
    model_kwargs = {}
    if include_features is not None:
        model_kwargs['include_features'] = include_features
    if exclude_features is not None:
        model_kwargs['exclude_features'] = exclude_features

    # Create output directory if specified
    if args.output_dir is not None:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    # Convert 'none' to None for resample method
    resample_method = None if args.resample == 'none' else args.resample

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION CONFIGURATION")
        print("=" * 80)
        print(f"Model:            {args.model}")
        print(f"Data:             {args.data}")
        print(f"Experiments:      {filter_experiments if filter_experiments else 'all'}")
        print(f"CV Splits:        {args.n_splits}")
        print(f"Random State:     {args.random_state}")
        print(f"Resampling:       {args.resample}")
        print(f"Include Features: {include_features if include_features else 'all (default)'}")
        print(f"Exclude Features: {exclude_features if exclude_features else 'none'}")
        print(f"Output Directory: {output_path}")
        print("=" * 80 + "\n")

    # Get model class/factory from registry
    try:
        from utils.model_registry import MODEL_REGISTRY
        model_class = MODEL_REGISTRY[args.model.lower()]
    except (ValueError, KeyError) as e:
        print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
        sys.exit(1)

    # Check if prediction mode
    if args.predict:
        # PREDICTION MODE: Train on all data and output predictions
        predictions_save_path = str(output_path / f"{args.model}_predictions.csv")

        try:
            model, predictions_df = run_full_prediction(
                model_class=model_class,
                model_kwargs=model_kwargs,
                csv_path=args.data,
                filter_experiments=filter_experiments,
                random_state=args.random_state,
                verbose=not args.quiet,
                save_predictions_path=predictions_save_path,
                resample_method=resample_method
            )
        except Exception as e:
            print(f"\nError during prediction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Print final summary for prediction mode
        if not args.quiet:
            print("\n" + "=" * 80)
            print("PREDICTION COMPLETE")
            print("=" * 80)
            print(f"Model:              {args.model}")
            print(f"Predictions file:   {predictions_save_path}")
            print(f"Total turns:        {len(predictions_df)}")

            # Print selected features
            if model.get_selected_features() is not None:
                selected_features = model.get_selected_features()
                print(f"\nFeatures Used: {len(selected_features)}")
                if len(selected_features) <= 20:
                    print(f"  {', '.join(selected_features)}")

            print("=" * 80 + "\n")

    else:
        # EVALUATION MODE: K-fold cross-validation
        # Set up feature importance save path
        importance_save_path = str(output_path / f"{args.model}_feature_importance.csv")

        # Run evaluation
        try:
            summary, feature_importance, models = run_kfold_evaluation(
                model_class=model_class,
                model_kwargs=model_kwargs,
                csv_path=args.data,
                filter_experiments=filter_experiments,
                n_splits=args.n_splits,
                random_state=args.random_state,
                verbose=not args.quiet,
                save_importance_path=importance_save_path,
                resample_method=resample_method
            )
        except Exception as e:
            print(f"\nError during evaluation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Print final summary
        if not args.quiet:
            print("\n" + "=" * 80)
            print("EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Model:              {args.model}")
            print(f"ROC-AUC:            {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
            print(f"Brier Score:        {summary['brier_score_mean']:.4f} ± {summary['brier_score_std']:.4f}")
            print(f"Log Loss:           {summary['log_loss_mean']:.4f} ± {summary['log_loss_std']:.4f}")
            print(f"Balanced Accuracy:  {summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}")

            # Print selected features
            if models and models[0].get_selected_features() is not None:
                selected_features = models[0].get_selected_features()
                print(f"\nFeatures Used: {len(selected_features)}")
                if len(selected_features) <= 20:
                    print(f"  {', '.join(selected_features)}")

            print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
