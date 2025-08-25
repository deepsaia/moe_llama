import torch
import logging
import os
import argparse
import time
from pyhocon import ConfigFactory, ConfigTree
from pathlib import Path

# FIRST: Configure PyTorch thread settings BEFORE any other PyTorch operations
# These settings must be set at the very beginning of the program
try:
    # Try to get basic thread settings from environment variables first
    num_threads = int(os.environ.get('LLAMA4MOE_NUM_THREADS', '4'))
    if num_threads == -1:
        num_threads = os.cpu_count() - 2
    num_interop_threads = int(os.environ.get('LLAMA4MOE_NUM_INTEROP_THREADS', '2'))
    
    # Set CPU thread configuration
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    logging.info(f"Initial thread configuration: {num_threads} intra-op, {num_interop_threads} inter-op")
except Exception as e:
    logging.warning(f"Could not set initial thread configuration: {str(e)}")
    # Fallback to defaults
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLaMA4MoE-Inference")

def load_config(config_path="config.hocon"):
    """
    Load configuration from a HOCON file.
    
    Args:
        config_path: Path to the HOCON configuration file
        
    Returns:
        config: Configuration object
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        config = ConfigFactory.parse_file(config_path)
        logger.info("Configuration loaded successfully")
        
        # Log the main configuration parameters
        logger.info("Model configuration:")
        logger.info(f"  dim: {config['model']['dim']}")
        logger.info(f"  num_layers: {config['model']['num_layers']}")
        logger.info(f"  num_heads: {config['model']['num_heads']}")
        logger.info(f"  num_experts: {config['model']['num_experts']}")
        logger.info(f"  top_k: {config['model']['top_k']}")
        logger.info(f"  max_seq_len: {config['model']['max_seq_len']}")
        logger.info(f"  shared_expert: {config['model']['shared_expert']}")
        
        logger.info("Device configuration:")
        logger.info(f"  type: {config.get('device', {}).get('type', 'auto')}")
        logger.info(f"  num_cpu_threads: {config.get('device', {}).get('num_cpu_threads', 4)}")
        logger.info(f"  num_cpu_interop_threads: {config.get('device', {}).get('num_cpu_interop_threads', 2)}")
        logger.info(f"  gpu_ids: {config.get('device', {}).get('gpu_ids', [0])}")
        logger.info(f"  use_mps: {config.get('device', {}).get('use_mps', True)}")
        
        logger.info("Inference configuration:")
        logger.info(f"  max_new_tokens: {config['inference']['max_new_tokens']}")
        logger.info(f"  temperature: {config['inference']['temperature']}")
        logger.info(f"  top_k: {config['inference']['top_k']}")
        logger.info(f"  top_p: {config['inference']['top_p']}")
        
        return config
    except Exception as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        raise

def setup_device(config):
    """Configure device settings based on configuration"""
    device_config = config.get('device', {})
    
    # Determine device type
    device_type = device_config.get('type', 'auto')
    use_mps = device_config.get('use_mps', True)
    
    # Check for MPS (Apple Silicon)
    if device_type == 'auto' and use_mps and hasattr(torch, 'mps') and torch.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders) for acceleration")
    # Check for CUDA
    elif device_type == 'auto' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Explicit device types
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device_type == 'mps' and hasattr(torch, 'mps') and torch.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders) for acceleration")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for computation")
    
    return device

def load_model_and_tokenizer(config):
    """
    Load a trained model and its tokenizer from disk using configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        model: The loaded LLaMA4MoE model
        tokenizer: The loaded CharacterTokenizer
        device: The device the model is on
    """
    model_path = config['paths']['model_path']
    logger.info(f"Loading model from {model_path}")
    
    # Set up device based on configuration
    device = setup_device(config)
    
    # First, load the tokenizer to get vocab size
    tokenizer_path = Path(model_path) / "vocab.txt"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer vocabulary file not found at {tokenizer_path}")
    
    # Dynamically import CharacterTokenizer from the main module
    try:
        from moellama import CharacterTokenizer
    except ImportError:
        # Try alternative import path
        try:
            from .moellama import CharacterTokenizer
        except ImportError:
            raise ImportError("Could not import CharacterTokenizer from moellama module. "
                            "Make sure moellama.py is in the Python path.")
    
    tokenizer = CharacterTokenizer(vocab_file=str(tokenizer_path))
    vocab_size = len(tokenizer)
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Create model with configuration parameters
    try:
        from moellama import LLaMA4MoE
    except ImportError:
        # Try alternative import path
        try:
            from .moellama import LLaMA4MoE
        except ImportError:
            raise ImportError("Could not import LLaMA4MoE from moellama module. "
                            "Make sure moellama.py is in the Python path.")
    
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
    
    # Load model weights
    model_path_file = Path(model_path) / "model.pt"
    if not model_path_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_path_file}")
    
    # Load the state dict
    state_dict = torch.load(str(model_path_file), map_location=device)
    
    # Verify the dimensions match
    embedding_shape = state_dict['token_embeddings.weight'].shape[0]
    if embedding_shape != vocab_size:
        logger.warning(f"Vocabulary size mismatch detected!")
        logger.warning(f"State dict expects vocab size: {embedding_shape}")
        logger.warning(f"Current tokenizer has vocab size: {vocab_size}")
        logger.warning(f"Using the state dict vocabulary size: {embedding_shape}")
        
        # Recreate the model with the correct vocabulary size from the state dict
        model = LLaMA4MoE(
            vocab_size=embedding_shape,
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
    
    # Now load the state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, tokenizer, device

def generate_text(
    prompt,
    config_path="config.hocon",
    model_path=None,
    device=None,
    return_details=False
):
    """
    Generate text from a prompt using the trained model with configuration.
    
    Args:
        prompt: Input text prompt
        config_path: Path to the HOCON configuration file
        model_path: Optional override for model path
        device: Device to use for inference
        return_details: If True, return additional generation details
        
    Returns:
        If return_details is False:
            generated_text: The generated text including the prompt
        If return_details is True:
            dict containing:
                - 'text': The generated text
                - 'prompt': The original prompt
                - 'completion': The newly generated part
                - 'prompt_tokens': Token IDs for the prompt
                - 'completion_tokens': Token IDs for the completion
                - 'total_tokens': Total token count
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override model path if provided
        if model_path:
            config['paths']['model_path'] = model_path
        
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(config)
        
        # Get inference parameters from config
        max_new_tokens = config['inference'].get('max_new_tokens', 50)
        temperature = config['inference'].get('temperature', 0.7)
        top_k = config['inference'].get('top_k', None)
        top_p = config['inference'].get('top_p', None)
        
        logger.info(f"Generating text with prompt: '{prompt}'")
        logger.info(f"Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, "
                   f"top_k={top_k}, top_p={top_p}")
        
        # Encode the prompt and convert to tensor with batch dimension
        token_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate text
        with torch.no_grad():
            start_time = time.time()
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            generation_time = time.time() - start_time
        
        # Decode the generated text
        all_tokens = generated_ids[0].tolist()
        generated_text = tokenizer.decode(all_tokens)
        
        # Extract just the newly generated part
        prompt_length = len(token_ids)
        completion_tokens = all_tokens[prompt_length:]
        completion = tokenizer.decode(completion_tokens)
        
        tokens_per_second = max_new_tokens / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {max_new_tokens} tokens in {generation_time:.2f}s "
                   f"({tokens_per_second:.2f} tokens/s)")
        
        if return_details:
            return {
                "text": generated_text,
                "prompt": prompt,
                "completion": completion,
                "prompt_token_ids": token_ids,
                "completion_token_ids": completion_tokens,
                "prompt_tokens": len(token_ids),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(all_tokens),
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            }
        else:
            return generated_text
            
    except Exception as e:
        logger.exception(f"Error during text generation: {str(e)}")
        raise

# def interactive_inference(config_path="config.hocon", model_path=None):
#     """
#     Start an interactive session for text generation using configuration.
    
#     Args:
#         config_path: Path to the HOCON configuration file
#         model_path: Optional override for model path
#     """
#     logger.info("Starting interactive inference session. Type 'exit' to quit.")
    
#     try:
#         # Load configuration
#         config = load_config(config_path)
        
#         # Override model path if provided
#         if model_path:
#             config['paths']['model_path'] = model_path
        
#         # Load model and tokenizer
#         model, tokenizer, device = load_model_and_tokenizer(config)
        
#         # Get default inference parameters
#         default_max_new_tokens = config['inference'].get('max_new_tokens', 50)
#         default_temperature = config['inference'].get('temperature', 0.7)
#         default_top_k = config['inference'].get('top_k', None)
#         default_top_p = config['inference'].get('top_p', None)
        
#         while True:
#             try:
#                 prompt = input("\nPrompt: ")
#                 if not prompt or prompt.lower() in ['exit', 'quit', 'bye']:
#                     break
                
#                 # Get user input for generation parameters with defaults from config
#                 max_new_tokens_input = input(f"Max new tokens (default {default_max_new_tokens}): ").strip()
#                 max_new_tokens = int(max_new_tokens_input) if max_new_tokens_input else default_max_new_tokens
                
#                 temperature_input = input(f"Temperature (default {default_temperature}): ").strip()
#                 temperature = float(temperature_input) if temperature_input else default_temperature
                
#                 top_k_input = input(f"Top-k (default {default_top_k}): ").strip()
#                 top_k = int(top_k_input) if top_k_input else default_top_k
                
#                 top_p_input = input(f"Top-p (default {default_top_p}): ").strip()
#                 top_p = float(top_p_input) if top_p_input else default_top_p
                
#                 print("\nGenerating...")
                
#                 result = generate_text(
#                     prompt,
#                     config_path=config_path,
#                     model_path=model_path,
#                     return_details=True
#                 )
                
#                 print("\n=== Generated Text ===")
#                 print(result["text"])
#                 print("\n=== New Completion ===")
#                 print(result["completion"])
#                 print(f"\nStats: {result['total_tokens']} total tokens, "
#                      f"{result['tokens_per_second']:.2f} tokens/sec")
                     
#             except KeyboardInterrupt:
#                 print("\nInterrupted by user. Type 'exit' to quit.")
#                 continue
#             except Exception as e:
#                 logger.error(f"Error generating text: {str(e)}")
#                 print(f"Error generating text: {str(e)}")
#                 continue
    
#     except KeyboardInterrupt:
#         logger.info("\nInteractive session terminated by user")
#     except Exception as e:
#         logger.error(f"Error in interactive session: {str(e)}")
#         print(f"Error in interactive session: {str(e)}")
#     finally:
#         logger.info("Interactive session ended")

def main():
    """Main function to run the inference script"""
    parser = argparse.ArgumentParser(description="LLaMA-4MoE Text Generation CLI")

    # === Shared Args ===
    parser.add_argument("--config", type=str, default="config.hocon",
                        help="Path to the configuration file (default: config.hocon)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the model directory (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose flag to stdout token accounting")

    # === Generation Mode Selection ===
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--prompt", type=str, nargs='+', 
                            help="One or more prompts to generate text from")
    mode_group.add_argument("--interactive", "-i", action="store_true",
                            help="Run in interactive mode (ask for input at runtime)")
    mode_group.add_argument("--stdin", action="store_true",
                            help="Read prompts from stdin (one per line)")

    # === Generation Parameters ===
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Number of new tokens to generate (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (overrides config)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling (overrides config)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-p (nucleus) sampling (overrides config)")

    args = parser.parse_args()

    # Load config early to access defaults
    try:
        config = load_config(args.config)
        inference_config = config['inference']
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Get generation params (CLI > Config > Defaults)
    max_new_tokens = args.max_new_tokens or inference_config.get('max_new_tokens', 50)
    temperature = args.temperature or inference_config.get('temperature', 0.7)
    top_k = args.top_k if args.top_k is not None else inference_config.get('top_k', None)
    top_p = args.top_p if args.top_p is not None else inference_config.get('top_p', None)
    verbose = args.verbose

    # === Prompt Input Handling ===
    if args.interactive:
        _run_interactive_loop(args.config, args.model_path, 
                              max_new_tokens, temperature, top_k, top_p, verbose)
    elif args.stdin:
        import sys
        for line in sys.stdin:
            prompt = line.strip()
            if prompt:
                _generate_and_print(prompt, args.config, args.model_path,
                                    max_new_tokens, temperature, top_k, top_p, verbose)
    elif args.prompt:
        for prompt in args.prompt:
            _generate_and_print(prompt, args.config, args.model_path,
                                max_new_tokens, temperature, top_k, top_p, verbose)
    else:
        # Default: run example
        print("=== Detailed Generation Example ===")
        result = generate_text(
            "To be or not to be",
            config_path=args.config,
            model_path=args.model_path,
            return_details=True
        )
        print(f"Prompt: '{result['prompt']}'")
        print(f"Completion: '{result['completion']}'")
        print(f"Stats: {result['total_tokens']} total tokens, "
              f"{result['tokens_per_second']:.2f} tokens/sec\n")

def _generate_and_print(prompt, config_path, model_path,
                        max_new_tokens, temperature, top_k, top_p, verbose):
    """Helper: Generate and print result in standard format"""
    print(f"\n=== Generating for prompt ===")
    print(f"Prompt: '{prompt}'")
    try:
        result = generate_text(
            prompt,
            config_path=config_path,
            model_path=model_path,
            return_details=True
        )
        print(f"Completion: '{result['completion']}'")
        print(f"Stats: {result['total_tokens']} total tokens, "
              f"{result['tokens_per_second']:.2f} tokens/sec")
        if verbose:
            print(f"\nVerbose result: {result}")
    except Exception as e:
        logger.error(f"Error generating for prompt '{prompt}': {e}")
        print(f"Error: {e}")


def _run_interactive_loop(config_path, model_path, max_new_tokens, temperature, top_k, top_p, verbose):
    """Run interactive mode with pre-defined parameters, no per-prompt input"""
    logger.info("Starting interactive inference session. Type 'exit' to quit.")
    try:
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if not prompt or prompt.lower() in ['exit', 'quit', 'bye']:
                    break

                print("\nUsing generation parameters:")
                print(f"  max_new_tokens: {max_new_tokens}")
                print(f"  temperature: {temperature}")
                print(f"  top_k: {top_k}")
                print(f"  top_p: {top_p}")
                print("Generating...\n")

                result = generate_text(
                    prompt,
                    config_path=config_path,
                    model_path=model_path,
                    return_details=True
                )

                print("\n=== Generated Text ===")
                print(result["text"])
                print("\n=== New Completion ===")
                print(result["completion"])
                print(f"\nStats: {result['total_tokens']} total tokens, "
                      f"{result['tokens_per_second']:.2f} tokens/sec")
                if verbose:
                    print(f"\nVerbose result: {result}")

            except KeyboardInterrupt:
                print("\nInterrupted by user. Type 'exit' to quit.")
                continue
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                print(f"Error generating text: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in interactive session: {str(e)}")
    finally:
        logger.info("Interactive session ended")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    main()