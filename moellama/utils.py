"""
Utility functions for configuration and device management.

This module provides helper functions for:
- Loading HOCON configuration files
- Setting up compute devices (CPU, CUDA, MPS)
- Configuring thread settings for optimal performance
"""

import os
import logging

import torch
from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)


def load_config(config_path="config.hocon"):
    """
    Load configuration from a HOCON file.

    HOCON (Human-Optimized Config Object Notation) is a JSON superset
    that supports comments, includes, and other convenience features.

    The configuration file should contain sections for:
    - model: Architecture parameters
    - device: Hardware configuration
    - training: Training hyperparameters
    - inference: Generation parameters
    - paths: File paths for models and data

    Args:
        config_path: Path to the HOCON configuration file

    Returns:
        Configuration object (dictionary-like)

    Raises:
        FileNotFoundError: If config file doesn't exist
        Exception: If config file cannot be parsed
    """
    logger.info(f"Loading configuration from {config_path}")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        config = ConfigFactory.parse_file(config_path)
        logger.info("Configuration loaded successfully")

        # Log main configuration parameters
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
        logger.info(f"  gpu_ids: {config.get('device', {}).get('gpu_ids', [0])}")
        logger.info(f"  use_mps: {config.get('device', {}).get('use_mps', True)}")

        logger.info("Training configuration:")
        logger.info(f"  batch_size: {config['training']['batch_size']}")
        logger.info(f"  learning_rate: {config['training']['learning_rate']}")
        logger.info(f"  epochs: {config['training']['epochs']}")
        logger.info(f"  dataset: {config['training']['dataset']}")

        return config

    except Exception as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        raise


def setup_device(config):
    """
    Configure compute device based on configuration.

    This function:
    1. Sets CPU thread configuration for optimal performance
    2. Detects available devices (CUDA, MPS, CPU)
    3. Selects the appropriate device based on config and availability

    Device selection priority (when type='auto'):
    1. MPS (Apple Silicon) if available and use_mps=true
    2. CUDA (NVIDIA GPU) if available
    3. CPU as fallback

    CPU Thread Configuration:
        - num_cpu_threads: Intra-op parallelism (within operations)
            - -1 means use (total cores - 2) to keep system responsive
        - num_cpu_interop_threads: Inter-op parallelism (between operations)
            - Typically set to 2 for good balance

    Args:
        config: Configuration object with 'device' section

    Returns:
        PyTorch device object
    """
    device_config = config.get('device', {})

    # Configure CPU threads
    num_threads = device_config.get('num_cpu_threads', 4)
    if num_threads == -1:
        # Use all but 2 cores to keep system responsive
        num_threads = os.cpu_count() - 2

    num_interop_threads = device_config.get('num_cpu_interop_threads', 2)

    logger.info(
        f"Setting CPU threads: {num_threads} intra-op, "
        f"{num_interop_threads} inter-op"
    )
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    # Determine device type
    device_type = device_config.get('type', 'auto')
    use_mps = device_config.get('use_mps', True)

    # Check for MPS (Apple Silicon)
    if device_type == 'auto' and use_mps:
        if hasattr(torch, 'mps') and torch.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple MPS (Metal Performance Shaders)")
            return device

    # Check for CUDA (NVIDIA GPU)
    if device_type == 'auto' or device_type == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {gpu_name}")
            return device

    # Explicit device types
    if device_type == 'mps':
        if hasattr(torch, 'mps') and torch.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple MPS (Metal Performance Shaders)")
            return device
        else:
            logger.warning("MPS not available, falling back to CPU")

    # Fallback to CPU
    device = torch.device('cpu')
    logger.info("Using CPU for computation")
    return device


def get_model_size(model):
    """
    Calculate the total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """
    Format large numbers in a human-readable way.

    Examples:
        1234 -> "1.2K"
        1234567 -> "1.2M"
        1234567890 -> "1.2B"

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def log_model_info(model):
    """
    Log useful information about the model.

    Logs:
    - Total parameters
    - Trainable parameters
    - Model size in human-readable format

    Args:
        model: PyTorch model
    """
    total_params, trainable_params = get_model_size(model)

    logger.info("Model Information:")
    logger.info(f"  Total parameters: {format_number(total_params)} ({total_params:,})")
    logger.info(f"  Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")

    # Estimate model size in MB (assumes float32 = 4 bytes)
    size_mb = (total_params * 4) / (1024 * 1024)
    logger.info(f"  Estimated size (float32): {size_mb:.1f} MB")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")
