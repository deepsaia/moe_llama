"""
Standalone benchmark runner for post-training evaluation.

This script loads a trained model checkpoint and runs comprehensive benchmarks,
generating a detailed markdown report.

Usage:
    python -m scripts.run_benchmarks --model-path ./model/model_wikitext_20250101_120000.pt
    python -m scripts.run_benchmarks --model-path ./model/model_wikitext_20250101_120000.pt --quick
    python -m scripts.run_benchmarks --config config.hocon --output-dir ./eval_results
"""

import argparse
from loguru import logger
from moellama.logging_setup import setup_logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from moellama import LLaMA4MoE, BPETokenizer, load_config, setup_device
from moellama.benchmarks import (
    BenchmarkEvaluator,
    get_default_benchmarks,
    get_comprehensive_benchmarks
)

# Configure logging
setup_logging()


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find the most recently modified file matching a pattern."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_model_and_tokenizer(
    config: dict,
    model_path: Optional[str] = None,
    vocab_path: Optional[str] = None
):
    """
    Load trained model and tokenizer.

    Args:
        config: Configuration dictionary
        model_path: Optional specific model checkpoint path
        vocab_path: Optional specific vocab file path

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Setup device
    device = setup_device(config)
    logger.info(f"Using device: {device}")

    # Determine paths
    model_dir = Path(config.get('paths', {}).get('model_path', './model'))

    # Load tokenizer
    if vocab_path:
        tokenizer_path = Path(vocab_path)
    else:
        # Try to find vocab file
        tokenizer_path = find_latest_file(model_dir, "vocab_*.txt")
        if tokenizer_path is None:
            tokenizer_path = model_dir / "vocab.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(str(tokenizer_path))
    vocab_size = len(tokenizer)
    logger.info(f"Loaded tokenizer (vocab size: {vocab_size})")

    # Load model checkpoint
    if model_path:
        checkpoint_path = Path(model_path)
    else:
        # Find latest model checkpoint
        checkpoint_path = find_latest_file(model_dir, "model_*.pt")
        if checkpoint_path is None:
            checkpoint_path = model_dir / "model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")

    # Create model with config
    model_config = config.get('model', {})
    model = LLaMA4MoE(
        vocab_size=vocab_size,
        dim=model_config.get('dim', 512),
        num_layers=model_config.get('num_layers', 8),
        num_heads=model_config.get('num_heads', 8),
        num_experts=model_config.get('num_experts', 8),
        top_k=model_config.get('top_k', 2),
        max_seq_len=model_config.get('max_seq_len', 512),
        dropout=model_config.get('dropout', 0.0),
        shared_expert=model_config.get('shared_expert', True),
        load_balancing_loss_coef=model_config.get('load_balancing_loss_coef', 0.01)
    )

    # Load weights
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, tokenizer, device


def generate_benchmark_report(
    model_name: str,
    config: dict,
    results: dict,
    output_path: Path
):
    """
    Generate comprehensive benchmark report in markdown format.

    Args:
        model_name: Name of the model
        config: Configuration dictionary
        results: Benchmark results dictionary
        output_path: Path to save the report
    """
    model_config = config.get('model', {})

    content = f"""# Benchmark Evaluation Report: {model_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration

- **Vocabulary Size:** {model_config.get('vocab_size', 'N/A')}
- **Dimensions:** {model_config.get('dim', 'N/A')}
- **Layers:** {model_config.get('num_layers', 'N/A')}
- **Attention Heads:** {model_config.get('num_heads', 'N/A')}
- **Experts:** {model_config.get('num_experts', 'N/A')}
- **Top-K:** {model_config.get('top_k', 'N/A')}
- **Max Sequence Length:** {model_config.get('max_seq_len', 'N/A')}
- **Shared Expert:** {model_config.get('shared_expert', 'N/A')}

## Benchmark Results

| Benchmark | Score | Metric | Samples |
|-----------|-------|--------|---------|
"""

    # Add each benchmark result
    for bench_name, bench_result in results['benchmarks'].items():
        score = bench_result['primary_metric']
        metric_name = list(bench_result['metrics'].keys())[0] if bench_result['metrics'] else 'score'
        samples = bench_result['num_samples']
        content += f"| {bench_name} | {score:.4f} | {metric_name} | {samples} |\n"

    # Add average score
    avg_score = results['average_score']
    content += f"\n**Overall Average Score:** {avg_score:.4f}\n"

    # Add detailed breakdown
    content += "\n## Detailed Results\n\n"

    for bench_name, bench_result in results['benchmarks'].items():
        content += f"### {bench_name}\n\n"
        content += f"**Description:** {bench_result.get('description', 'N/A')}\n\n"
        content += f"**Metrics:**\n"
        for metric_name, metric_value in bench_result['metrics'].items():
            content += f"- {metric_name}: {metric_value}\n"
        content += "\n"

    content += "\n---\n*Report generated by MoeLLaMA Benchmark Runner*\n"

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    logger.info(f"Report saved to: {output_path}")


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks on a trained MoE model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.hocon",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        help="Path to tokenizer vocab file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./report",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with limited samples (100 per benchmark)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per benchmark (overrides --quick)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 = greedy)"
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("MoeLLaMA Benchmark Runner")
    logger.info("="*80)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Load model and tokenizer
        logger.info("\n" + "="*80)
        logger.info("Loading Model and Tokenizer")
        logger.info("="*80)
        model, tokenizer, device = load_model_and_tokenizer(
            config,
            model_path=args.model_path,
            vocab_path=args.vocab_path
        )

        # Create benchmark evaluator
        logger.info("\n" + "="*80)
        logger.info("Initializing Benchmark Evaluator")
        logger.info("="*80)
        evaluator = BenchmarkEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

        # Determine max samples
        if args.max_samples is not None:
            max_samples = args.max_samples
        elif args.quick:
            max_samples = 100
        else:
            max_samples = None  # Use all samples

        logger.info(f"Max samples per benchmark: {max_samples if max_samples else 'All'}")

        # Get benchmarks
        logger.info("\n" + "="*80)
        logger.info("Running Benchmarks")
        logger.info("="*80)
        if max_samples:
            benchmarks = get_default_benchmarks(max_samples=max_samples)
            logger.info("Running quick evaluation with limited samples")
        else:
            benchmarks = get_comprehensive_benchmarks(max_samples=None)
            logger.info("Running comprehensive evaluation with all samples")

        # Run evaluation
        results = evaluator.evaluate_all(benchmarks, verbose=True)

        # Generate report
        logger.info("\n" + "="*80)
        logger.info("Generating Report")
        logger.info("="*80)
        model_name = config.get('model', {}).get('name', 'moellama')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"bm_report_{model_name}_{timestamp}.md"

        generate_benchmark_report(
            model_name=model_name,
            config=config,
            results=results,
            output_path=output_path
        )

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Benchmark Summary")
        logger.info("="*80)
        for bench_name, bench_result in results['benchmarks'].items():
            score = bench_result['primary_metric']
            logger.info(f"  {bench_name}: {score:.4f}")
        logger.info(f"\n  Overall Average: {results['average_score']:.4f}")
        logger.info("="*80)
        logger.info(f"✓ Benchmark evaluation complete!")
        logger.info(f"✓ Report saved to: {output_path}")
        logger.info("="*80)

    except Exception as e:
        logger.exception(f"Benchmark evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
