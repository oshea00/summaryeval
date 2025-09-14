"""
Summary Evaluation Tool for CNN/DailyMail Dataset

This module provides functionality to evaluate reference summaries from the CNN/DailyMail dataset
using a comprehensive SummaryEvaluator with ROUGE-enhanced coherence scoring.

Key Features:
    - Loads and samples from CNN/DailyMail dataset (287k+ articles)
    - Evaluates summaries across three dimensions:
        * Accuracy: LLM-as-judge scoring using OpenAI models (GPT-4, GPT-4o)
        * Completeness: Topic coverage analysis using TF-IDF and semantic similarity
        * Coherence: ROUGE-enhanced scoring with discourse analysis and readability
    - Flexible output formats: pretty console display or CSV export
    - Configurable sampling with optional seeding for reproducibility
    - Command-line interface with comprehensive argument parsing

Usage Examples:
    python evalsummary.py                     # 5 samples, GPT-4o, pretty output
    python evalsummary.py -n 10 -m gpt-4     # 10 samples with GPT-4
    python evalsummary.py -f csv > results.csv # CSV export for analysis
    python evalsummary.py --seed 42 -n 20    # Reproducible results

Score Ranges:
    - Overall: 0.7-1.0 (Excellent), 0.5-0.7 (Good), 0.3-0.5 (Fair), 0.0-0.3 (Poor)
    - All individual metrics normalized to [0.0, 1.0] range
    - Weighted combination: 60% accuracy, 25% completeness, 15% coherence
"""

from datasets import load_dataset
import random
import argparse
import csv
import sys
from typing import List, Dict
from main import SummaryEvaluator


def load_dataset_splits(show_progress: bool = True):
    """
    Load the CNN/DailyMail dataset and return training split.

    This function downloads and loads the CNN/DailyMail dataset version 3.0.0,
    which contains approximately 287,000 news articles with reference summaries.

    Args:
        show_progress: Whether to display loading progress messages

    Returns:
        Dataset: The training split containing articles and reference summaries

    Note:
        First run may take time to download (~1.3GB). Subsequent runs use cached data.
        Each sample contains: article (full text), highlights (reference summary), id
    """
    if show_progress:
        print("📥 Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Access splits
    train_data = dataset["train"]

    if show_progress:
        print(f"✅ Dataset loaded - Train: {len(train_data)}")
    return train_data


def evaluate_reference_summaries(
    sample_size: int = 5, model: str = "gpt-4o", output_format: str = "pretty"
) -> None:
    """
    Evaluate reference summaries from CNN/DailyMail dataset using comprehensive metrics.

    This function randomly samples articles from the CNN/DailyMail dataset and evaluates
    their reference summaries using the SummaryEvaluator with ROUGE-enhanced coherence scoring.

    Evaluation Metrics:
        - Accuracy (60% weight): LLM-as-judge scoring using specified OpenAI model
        - Completeness (25% weight): Topic coverage using TF-IDF and semantic similarity
        - Coherence (15% weight): ROUGE-enhanced scoring with discourse and readability analysis

    Args:
        sample_size: Number of random articles to evaluate (default: 5)
        model: OpenAI model for accuracy evaluation (default: "gpt-4o")
               Supported: "gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-4o-mini"
        output_format: Output format (default: "pretty")
                      - "pretty": Formatted console display with progress indicators
                      - "csv": Machine-readable CSV with full article text

    Returns:
        None: Results printed to stdout (redirect for CSV storage)

    Output Fields (CSV mode):
        - article_index: Original dataset index
        - article_text: Full article content
        - reference_summary: CNN/DailyMail reference summary
        - accuracy, completeness, coherence, overall: Metric scores (0.000-1.000)
        - accuracy_rationale: LLM explanation of accuracy score
        - model: OpenAI model used for evaluation

    Requires:
        - OPENAI_API_KEY environment variable
        - Internet connection for first-time dataset download
        - ~2-10 seconds per evaluation depending on model and text length
    """
    # Load the dataset (suppress progress messages for CSV output)
    show_progress = output_format != "csv"
    train_data = load_dataset_splits(show_progress)

    # Initialize the evaluator
    evaluator = SummaryEvaluator(model)

    # Randomly select sample_size indices from train_data
    dataset_size = len(train_data)
    random_indices = random.sample(range(dataset_size), sample_size)

    if output_format == "csv":
        # CSV output
        fieldnames = [
            "article_index",
            "article_text",
            "reference_summary",
            "accuracy",
            "completeness",
            "coherence",
            "overall",
            "accuracy_rationale",
            "model",
        ]

        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, random_idx in enumerate(random_indices):
            article = train_data[random_idx]["article"]
            reference_summary = train_data[random_idx]["highlights"]

            # Use SummaryEvaluator to score reference summary
            score = evaluator.evaluate_summary(reference_summary, article, debug=True)

            # Write CSV row
            writer.writerow(
                {
                    "article_index": random_idx,
                    "article_text": article,  # Full article text for CSV
                    "reference_summary": reference_summary,
                    "accuracy": f"{score['accuracy']:.3f}",
                    "completeness": f"{score['completeness']:.3f}",
                    "coherence": f"{score['coherence']:.3f}",
                    "overall": f"{score['overall']:.3f}",
                    "accuracy_rationale": score.get("accuracy_rationale", ""),
                    "model": model,
                }
            )

    else:
        # Pretty console output
        print(f"🚀 Evaluating {sample_size} random reference summaries using {model}")
        print("=" * 80)

        for idx, random_idx in enumerate(random_indices):
            article = train_data[random_idx]["article"]
            reference_summary = train_data[random_idx]["highlights"]

            print(f"Article {idx+1} (index {random_idx}):")
            print(article[:500])  # first 500 chars of article for pretty output
            print("\nReference Summary:")
            print(reference_summary)
            print("-" * 80)

            # Use SummaryEvaluator to score reference summary
            score = evaluator.evaluate_summary(reference_summary, article, debug=True)

            # Format and display the score breakdown nicely
            print("\n📊 SUMMARY EVALUATION RESULTS:")
            print("-" * 40)
            print(
                f"🎯 Accuracy:     {score['accuracy']:.3f} ({score['accuracy']*100:.1f}%)"
            )
            print(
                f"📋 Completeness: {score['completeness']:.3f} ({score['completeness']*100:.1f}%)"
            )
            print(
                f"🔗 Coherence:    {score['coherence']:.3f} ({score['coherence']*100:.1f}%)"
            )
            print(
                f"⭐ Overall:      {score['overall']:.3f} ({score['overall']*100:.1f}%)"
            )

            # Include accuracy rationale if available (debug mode)
            if "accuracy_rationale" in score:
                print(f"\n💭 Accuracy Rationale: {score['accuracy_rationale']}")

            print("=" * 80)


def parse_arguments():
    """
    Parse command line arguments for the summary evaluation tool.

    Provides comprehensive argument parsing with validation and helpful examples.
    Supports configuration of sample size, model selection, output format, and random seeding.

    Returns:
        argparse.Namespace: Parsed command line arguments with:
            - sample_size: Number of articles to evaluate (1-1000)
            - model: OpenAI model name for accuracy scoring
            - format: Output format ("pretty" or "csv")
            - seed: Optional random seed for reproducibility

    Examples:
        Default usage: python evalsummary.py
        Custom sample: python evalsummary.py -n 20 -m gpt-4
        CSV export: python evalsummary.py -f csv > results.csv
        Reproducible: python evalsummary.py --seed 42 -n 10
    """
    parser = argparse.ArgumentParser(
        description="Evaluate reference summaries from CNN/DailyMail dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use defaults (5 samples, gpt-4o, pretty output)
  %(prog)s -n 10                    # Evaluate 10 random samples
  %(prog)s -m gpt-4                 # Use GPT-4 model
  %(prog)s -n 3 -m gpt-4            # 3 samples with GPT-4
  %(prog)s -f csv                   # CSV output with full article text
  %(prog)s -n 20 -f csv > results.csv  # Save 20 evaluations to CSV file
  %(prog)s --sample-size 10 --model gpt-4 --format pretty  # Explicit options
        """,
    )

    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=5,
        help="Number of random articles to evaluate (default: 5)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name to use for evaluation (default: gpt-4o). "
        "Common options: gpt-4o, gpt-4, gpt-4-turbo, gpt-4o-mini",
    )

    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible results (optional)"
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["pretty", "csv"],
        default="pretty",
        help="Output format (default: pretty). 'pretty' for console display, 'csv' for CSV output with full article text",
    )

    return parser.parse_args()


# Example usage - Generate summaries with different quality and length settings
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed if provided for reproducible results
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")

    # Run evaluation with parsed parameters
    evaluate_reference_summaries(
        sample_size=args.sample_size, model=args.model, output_format=args.format
    )
