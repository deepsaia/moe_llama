"""
Evaluation script for the Mixture of Experts language model.

This script runs benchmarks and generates a comprehensive evaluation report.
All benchmarks are configurable via config.hocon.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --config custom_config.hocon
    python -m scripts.evaluate --model-file path/to/model.pt
"""

import argparse
from loguru import logger
from moellama.logging_setup import setup_logging
import os
from pathlib import Path

import torch

from moellama import (
    LLaMA4MoE,
    BPETokenizer,
    prepare_dataset,
    setup_device,
    load_config
)
from moellama.benchmarks import run_benchmarks
from moellama.report import generate_report

# Configure logging
setup_logging()


def find_latest_file(directory, pattern):
    """Find the most recently modified file matching a pattern."""
    directory = Path(directory)
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_model_and_tokenizer(config, model_file=None, vocab_file=None):
    """
    Load a trained model and tokenizer.

    Args:
        config: Configuration object
        model_file: Optional specific model checkpoint
        vocab_file: Optional specific vocab file

    Returns:
        Tuple of (model, tokenizer, device)
    """
    model_dir = config['paths']['model_path']
    logger.info(f"Loading model from {model_dir}")

    # Setup device
    device = setup_device(config)

    # Load tokenizer
    if vocab_file:
        tokenizer_path = Path(vocab_file)
    else:
        # Find latest vocab file
        tokenizer_path = find_latest_file(model_dir, "vocab_*.txt")
        if tokenizer_path is None:
            tokenizer_path = Path(model_dir) / "vocab.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer()
    tokenizer.load_vocab(str(tokenizer_path))
    vocab_size = len(tokenizer)
    logger.info(f"Loaded tokenizer (vocab size: {vocab_size})")

    # Load model
    if model_file:
        model_path = Path(model_file)
    else:
        model_path = find_latest_file(model_dir, "model_*.pt")
        if model_path is None:
            model_path = Path(model_dir) / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create model
    model = LLaMA4MoE(
        vocab_size=vocab_size,
        dim=config['model']['dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_experts=config['model']['num_experts'],
        top_k=config['model']['top_k'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        shared_expert=config['model']['shared_expert'],
        load_balancing_loss_coef=config['model']['load_balancing_loss_coef']
    )

    # Load weights
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    return model, tokenizer, device


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MoE model and generate report"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.hocon",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        help="Specific model checkpoint to evaluate"
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="Specific vocab file to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report.md",
        help="Output path for the report"
    )
    parser.add_argument(
        "--skip-eval-dataset",
        action="store_true",
        help="Skip loading evaluation dataset (faster, but no perplexity/accuracy)"
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)

    try:
        # Load configuration
        config = load_config(args.config)

        # Check if evaluation is enabled
        eval_config = config.get('evaluation', {})
        if not eval_config.get('enabled', True):
            logger.info("Evaluation is disabled in config. Set evaluation.enabled = true to enable.")
            return

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer, device = load_model_and_tokenizer(
            config,
            model_file=args.model_file,
            vocab_file=args.vocab_file
        )

        # Load evaluation dataset if needed
        eval_dataset = None
        if not args.skip_eval_dataset:
            enabled_benchmarks = eval_config.get('enabled_benchmarks', [])
            needs_dataset = any(b in enabled_benchmarks for b in ['perplexity', 'accuracy'])

            if needs_dataset:
                logger.info("Loading evaluation dataset...")
                _, eval_dataset, _ = prepare_dataset(config, tokenizer=tokenizer, device=device)
                logger.info(f"Evaluation dataset: {len(eval_dataset)} sequences")

        # Run benchmarks
        logger.info("\n" + "="*60)
        logger.info("Running Benchmarks")
        logger.info("="*60 + "\n")

        results = run_benchmarks(model, tokenizer, device, config, eval_dataset)

        # Generate report
        logger.info("\n" + "="*60)
        logger.info("Generating Report")
        logger.info("="*60 + "\n")

        generate_report(model, config, results, output_path=args.output)

        logger.info("="*60)
        logger.info(f"Evaluation complete! Report saved to: {args.output}")
        logger.info("="*60)

    except Exception as e:
        logger.exception(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
