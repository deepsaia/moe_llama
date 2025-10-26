"""
Interactive inference script for the Mixture of Experts language model.

This script provides an interactive chat interface where you can:
- Enter prompts and see generated text in real-time
- Adjust generation parameters on the fly
- See token statistics and generation speed

Usage:
    python -m scripts.interactive
    python -m scripts.interactive --config custom_config.hocon
"""

import argparse
import logging
import time
import sys
from pathlib import Path
import os

import torch
from torch.cuda.amp import autocast

from moellama import LLaMA4MoE, BPETokenizer, load_config, setup_device

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Less verbose for interactive use
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    print(f"Loading model from {model_dir}...")

    # Setup device
    device = setup_device(config)

    # Load tokenizer
    if vocab_file:
        tokenizer_path = Path(vocab_file)
    else:
        tokenizer_path = find_latest_file(model_dir, "vocab_*.txt")
        if tokenizer_path is None:
            tokenizer_path = Path(model_dir) / "vocab.txt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer()
    tokenizer.load_vocab(str(tokenizer_path))
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer (vocab size: {vocab_size})")

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

    print(f"Model loaded successfully!")
    print(f"Device: {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_new_tokens, temperature, top_k, top_p, verbose=False):
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
        verbose: Show detailed statistics

    Returns:
        Tuple of (generated_text, stats_dict)
    """
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    prompt_tokens = len(token_ids)
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
    total_tokens = len(generated_ids[0])
    new_tokens = total_tokens - prompt_tokens

    stats = {
        'prompt_tokens': prompt_tokens,
        'new_tokens': new_tokens,
        'total_tokens': total_tokens,
        'generation_time': generation_time,
        'tokens_per_sec': new_tokens / generation_time if generation_time > 0 else 0
    }

    return generated_text, stats


def get_input(prompt_text, default=None, input_type=str):
    """
    Get user input with a default value.

    Args:
        prompt_text: Prompt to display
        default: Default value if user presses enter
        input_type: Type to convert input to

    Returns:
        User input or default value
    """
    if default is not None:
        prompt_text = f"{prompt_text} (default: {default}): "
    else:
        prompt_text = f"{prompt_text}: "

    user_input = input(prompt_text).strip()

    if not user_input and default is not None:
        return default

    if user_input == "":
        return default

    try:
        if input_type == bool:
            return user_input.lower() in ['true', 'yes', 'y', '1']
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default


def main():
    """Main interactive inference function."""
    parser = argparse.ArgumentParser(
        description="Interactive text generation with MoE model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.hocon",
        help="Path to configuration file"
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed token statistics"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            config,
            model_file=args.model_file,
            vocab_file=args.vocab_file
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Default generation parameters
    default_max_tokens = config['inference'].get('max_new_tokens', 50)
    default_temperature = config['inference'].get('temperature', 0.7)
    default_top_k = config['inference'].get('top_k', 50)
    default_top_p = config['inference'].get('top_p', 0.95)

    print("\n" + "="*80)
    print("Interactive Text Generation")
    print("="*80)
    print("\nCommands:")
    print("  - Type your prompt and press Enter to generate")
    print("  - Type 'exit' or 'quit' to quit")
    print("  - Type 'params' to change generation parameters")
    print("  - Press Ctrl+C to interrupt generation")
    print("\n" + "="*80 + "\n")

    # Interactive loop
    while True:
        try:
            # Get prompt
            prompt = input("\nPrompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'params':
                print("\n--- Generation Parameters ---")
                default_max_tokens = get_input(
                    "Max new tokens",
                    default_max_tokens,
                    int
                )
                default_temperature = get_input(
                    "Temperature",
                    default_temperature,
                    float
                )
                default_top_k = get_input(
                    "Top-k",
                    default_top_k,
                    lambda x: None if x.lower() == 'none' else int(x)
                )
                default_top_p = get_input(
                    "Top-p",
                    default_top_p,
                    lambda x: None if x.lower() == 'none' else float(x)
                )
                print("Parameters updated!\n")
                continue

            # Generate
            print("\nGenerating...\n")

            generated_text, stats = generate_text(
                model, tokenizer, device, prompt,
                default_max_tokens, default_temperature,
                default_top_k, default_top_p,
                verbose=args.verbose
            )

            # Display results
            print("="*80)
            print("Generated Text:")
            print("-"*80)
            print(generated_text)
            print("-"*80)

            # Show statistics
            if args.verbose:
                print(f"\nDetailed Statistics:")
                print(f"  Prompt tokens: {stats['prompt_tokens']}")
                print(f"  Generated tokens: {stats['new_tokens']}")
                print(f"  Total tokens: {stats['total_tokens']}")
                print(f"  Generation time: {stats['generation_time']:.2f}s")
                print(f"  Speed: {stats['tokens_per_sec']:.2f} tokens/sec")
            else:
                print(f"\nStats: {stats['total_tokens']} tokens, "
                      f"{stats['tokens_per_sec']:.2f} tokens/sec")

            print("="*80)

        except KeyboardInterrupt:
            print("\n\n[Generation interrupted]")
            continue

        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Generation error")
            continue


if __name__ == "__main__":
    main()
