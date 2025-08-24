import torch
import logging
import os
import argparse
import time
from pyhocon import ConfigFactory, ConfigTree
from pathlib import Path

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
        
        logger.info("Inference configuration:")
        logger.info(f"  max_new_tokens: {config['inference']['max_new_tokens']}")
        logger.info(f"  temperature: {config['inference']['temperature']}")
        logger.info(f"  top_k: {config['inference']['top_k']}")
        logger.info(f"  top_p: {config['inference']['top_p']}")
        
        return config
    except Exception as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        raise

def load_model_and_tokenizer(config):
    """
    Load a trained model and its tokenizer from disk using configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        model: The loaded LLaMA4MoE model
        tokenizer: The loaded CharacterTokenizer
    """
    model_path = config['paths']['model_path']
    logger.info(f"Loading model from {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
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
        all_tokens = generated_ids[0].cpu().numpy().tolist()
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
                "prompt_tokens": token_ids,
                "completion_tokens": completion_tokens,
                "total_tokens": len(all_tokens),
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            }
        else:
            return generated_text
            
    except Exception as e:
        logger.exception(f"Error during text generation: {str(e)}")
        raise

def interactive_inference(config_path="config.hocon", model_path=None):
    """
    Start an interactive session for text generation using configuration.
    
    Args:
        config_path: Path to the HOCON configuration file
        model_path: Optional override for model path
    """
    logger.info("Starting interactive inference session. Type 'exit' to quit.")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override model path if provided
        if model_path:
            config['paths']['model_path'] = model_path
        
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(config)
        
        # Get default inference parameters
        default_max_new_tokens = config['inference'].get('max_new_tokens', 50)
        default_temperature = config['inference'].get('temperature', 0.7)
        default_top_k = config['inference'].get('top_k', None)
        default_top_p = config['inference'].get('top_p', None)
        
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() == 'exit':
                break
            
            # Get user input for generation parameters with defaults from config
            max_new_tokens = int(input(f"Max new tokens (default {default_max_new_tokens}): ") 
                               or default_max_new_tokens)
            temperature = float(input(f"Temperature (default {default_temperature}): ") 
                              or default_temperature)
            top_k_input = input(f"Top-k (default {default_top_k}): ").strip()
            top_k = int(top_k_input) if top_k_input else default_top_k
            top_p_input = input(f"Top-p (default {default_top_p}): ").strip()
            top_p = float(top_p_input) if top_p_input else default_top_p
            
            print("\nGenerating...")
            
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
    
    except KeyboardInterrupt:
        logger.info("\nInteractive session terminated by user")
    except Exception as e:
        logger.error(f"Error in interactive session: {str(e)}")
    finally:
        logger.info("Interactive session ended")

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Text generation demo")
    parser.add_argument(
        "--im",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Provide a one-off prompt for generation"
    )
    args = parser.parse_args()

    if args.im:
        interactive_inference()
    elif args.prompt:
        print(f"=== Prompt Mode ===")
        result = generate_text(args.prompt, return_details=True)
        print(f"Prompt: '{result['prompt']}'")
        print(f"Completion: '{result['completion']}'")
        print(f"Stats: {result['total_tokens']} total tokens, "
              f"{result['tokens_per_second']:.2f} tokens/sec\n")
    else:
        print("=== Basic Generation Example ===")
        generated_text = generate_text(
            "The future of AI is",
        )
        print(f"Prompt: 'The future of AI is'")
        print(f"Generated: {generated_text[len('The future of AI is'):]}\n")
        
        print("=== Detailed Generation Example ===")
        result = generate_text(
            "To be or not to be",
            return_details=True
        )
        print(f"Prompt: '{result['prompt']}'")
        print(f"Completion: '{result['completion']}'")
        print(f"Stats: {result['total_tokens']} total tokens, "
             f"{result['tokens_per_second']:.2f} tokens/sec\n")