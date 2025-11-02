"""
Batch inference script for the Mixture of Experts language model.

This script loads a trained model and generates text from prompts.
Supports single prompt or reading multiple prompts from a file.

Usage:
    # Single prompt
    python -m scripts.inference --prompt "The future of AI is"

    # Multiple prompts from file
    python -m scripts.inference --prompts-file prompts.txt

    # Custom parameters
    python -m scripts.inference --prompt "Hello" --temperature 0.9 --max-tokens 100
"""

import argparse
from loguru import logger
from moellama.logging_setup import setup_logging
import time
from pathlib import Path
import glob
import os

import torch
from torch.cuda.amp import autocast

from moellama import LLaMA4MoE, BPETokenizer, load_config, setup_device

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
            tokenizer_path = Path(model_dir) / "vocab.txt"  # Legacy fallback

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(str(tokenizer_path))
    vocab_size = len(tokenizer)
    logger.info(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Load model
    if model_file:
        model_path = Path(model_file)
    else:
        model_path = find_latest_file(model_dir, "model_*.pt")
        if model_path is None:
            model_path = Path(model_dir) / "model.pt"  # Legacy fallback

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

    # Load state dict
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully from {model_path}")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_new_tokens, temperature, top_k, top_p):
    """
    Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer
        device: Compute device
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter

    Returns:
        Tuple of (generated_text, generation_time, num_tokens)
    """
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    start_time = time.time()

    with torch.no_grad():
        if device.type == 'cuda':
            with autocast():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
        else:
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

    generation_time = time.time() - start_time

    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    num_tokens = len(generated_ids[0]) - len(input_ids[0])

    return generated_text, generation_time, num_tokens


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Generate text using a trained MoE model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.hocon",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum new tokens to generate (overrides config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (overrides config)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling (overrides config)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling threshold (overrides config)"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        help="Specific model checkpoint to load"
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="Specific vocab file to load"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get generation parameters (CLI args override config)
    max_new_tokens = args.max_tokens or config['inference'].get('max_new_tokens', 50)
    temperature = args.temperature or config['inference'].get('temperature', 0.7)
    top_k = args.top_k or config['inference'].get('top_k', None)
    top_p = args.top_p or config['inference'].get('top_p', None)

    logger.info(f"Generation parameters: max_tokens={max_new_tokens}, "
                f"temperature={temperature}, top_k={top_k}, top_p={top_p}")

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        config,
        model_file=args.model_file,
        vocab_file=args.vocab_file
    )

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        # Default prompts
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant land",
            "To be or not to be"
        ]
        logger.info("Using default prompts")

    # Generate for each prompt
    print("\n" + "="*80)
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("-"*80)

        generated_text, gen_time, num_tokens = generate_text(
            model, tokenizer, device, prompt,
            max_new_tokens, temperature, top_k, top_p
        )

        # Extract just the completion (remove prompt)
        completion = generated_text[len(prompt):]

        print(f"Generated: {generated_text}")
        print(f"\nStats: {num_tokens} tokens in {gen_time:.2f}s "
              f"({num_tokens/gen_time:.2f} tokens/sec)")
        print("="*80)


if __name__ == "__main__":
    main()
