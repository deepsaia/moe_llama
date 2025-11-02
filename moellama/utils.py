"""
Utility functions for configuration and device management.

This module provides helper functions for:
- Loading HOCON configuration files
- Setting up compute devices (CPU, CUDA, MPS)
- Configuring thread settings for optimal performance
"""

import os
from loguru import logger

import torch
from pyhocon import ConfigFactory



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

        # Handle different dataset config formats
        training_config = config['training']
        if 'dataset_mixture' in training_config:
            num_datasets = len(training_config['dataset_mixture'])
            logger.info(f"  dataset: Multi-dataset mode ({num_datasets} datasets)")
            for i, ds in enumerate(training_config['dataset_mixture'], 1):
                logger.info(f"    {i}. {ds.get('name', 'unknown')} (ratio: {ds.get('ratio', 1.0)})")
        elif 'datasets' in training_config:
            datasets = training_config['datasets']
            logger.info(f"  dataset: Multi-dataset mode ({len(datasets)} datasets, equal ratios)")
            for ds_name in datasets:
                logger.info(f"    - {ds_name}")
        elif 'dataset' in training_config:
            logger.info(f"  dataset: {training_config['dataset']}")
        else:
            logger.warning("  dataset: No dataset configuration found (will need to be specified)")

        return config

    except Exception as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        raise


def setup_device(config):
    """
    Configure compute device based on configuration with robust validation.

    This function:
    1. Sets CPU thread configuration for optimal performance
    2. Validates user's preferred device from config
    3. Falls back gracefully if preferred device is unavailable
    4. Returns a validated PyTorch device object

    Device selection priority (when type='auto'):
    1. MPS (Apple Silicon) if available and use_mps=true
    2. CUDA (NVIDIA GPU) if available
    3. CPU as fallback

    Supported device types:
    - 'auto': Auto-detect best available device
    - 'cpu': Force CPU
    - 'cuda': Use default CUDA device (cuda:0)
    - 'cuda:N': Use specific CUDA device N
    - 'mps': Use Apple MPS

    CPU Thread Configuration:
        - num_cpu_threads: Intra-op parallelism (within operations)
            - -1 means use (total cores - 2) to keep system responsive
        - num_cpu_interop_threads: Inter-op parallelism (between operations)
            - Typically set to 2 for good balance

    Args:
        config: Configuration object with 'device' section

    Returns:
        PyTorch device object (validated and safe to use)
    """
    device_config = config.get('device', {})

    # Configure CPU threads
    num_threads = device_config.get('num_cpu_threads', 4)
    if num_threads == -1:
        # Use all but 2 cores to keep system responsive
        num_threads = max(1, os.cpu_count() - 2)

    num_interop_threads = device_config.get('num_cpu_interop_threads', 2)

    logger.info(
        f"Setting CPU threads: {num_threads} intra-op, "
        f"{num_interop_threads} inter-op"
    )
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)

    # Get user's device preference
    device_type = device_config.get('type', 'auto')
    use_mps = device_config.get('use_mps', True)

    logger.info(f"Device preference from config: '{device_type}'")

    # AUTO MODE: Select best available device
    if device_type == 'auto':
        # Priority 1: MPS (if enabled)
        if use_mps and hasattr(torch, 'mps') and torch.mps.is_available():
            device = torch.device('mps')
            logger.info("✓ Auto-selected: Apple MPS (Metal Performance Shaders)")
            return device

        # Priority 2: CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ Auto-selected: CUDA device 0 ({gpu_name})")
            return device

        # Priority 3: CPU fallback
        device = torch.device('cpu')
        logger.info("✓ Auto-selected: CPU (no GPU available)")
        return device

    # EXPLICIT DEVICE: Validate user's choice
    device_type_lower = device_type.lower()

    # CPU: Always available
    if device_type_lower == 'cpu':
        device = torch.device('cpu')
        logger.info("✓ Using CPU (as requested)")
        return device

    # MPS: Apple Silicon GPU
    if device_type_lower == 'mps':
        if hasattr(torch, 'mps') and torch.mps.is_available():
            device = torch.device('mps')
            logger.info("✓ Using Apple MPS (as requested)")
            return device
        else:
            logger.warning(
                "⚠ MPS requested but not available. "
                "Possible reasons: Not on Apple Silicon, PyTorch too old, or MPS not enabled."
            )
            logger.warning("→ Falling back to CPU")
            device = torch.device('cpu')
            return device

    # CUDA: NVIDIA GPU
    if device_type_lower.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning(
                "⚠ CUDA requested but not available. "
                "Possible reasons: No NVIDIA GPU, missing drivers, or PyTorch CPU-only build."
            )
            logger.warning("→ Falling back to CPU")
            device = torch.device('cpu')
            return device

        # Parse device ID if specified (e.g., 'cuda:1')
        if ':' in device_type_lower:
            try:
                device_id = int(device_type_lower.split(':')[1])
                num_gpus = torch.cuda.device_count()

                if device_id >= num_gpus:
                    logger.warning(
                        f"⚠ CUDA device {device_id} requested but only {num_gpus} GPU(s) available "
                        f"(valid IDs: 0-{num_gpus-1})"
                    )
                    logger.warning(f"→ Falling back to cuda:0")
                    device = torch.device('cuda:0')
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✓ Using CUDA device 0 ({gpu_name})")
                    return device
                else:
                    device = torch.device(f'cuda:{device_id}')
                    gpu_name = torch.cuda.get_device_name(device_id)
                    logger.info(f"✓ Using CUDA device {device_id} ({gpu_name})")
                    return device

            except (ValueError, IndexError) as e:
                logger.warning(f"⚠ Invalid CUDA device format: '{device_type}' ({e})")
                logger.warning("→ Falling back to cuda:0")
                device = torch.device('cuda:0')
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ Using CUDA device 0 ({gpu_name})")
                return device
        else:
            # Plain 'cuda' -> use cuda:0
            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            num_gpus = torch.cuda.device_count()
            logger.info(f"✓ Using CUDA device 0 ({gpu_name})")
            if num_gpus > 1:
                logger.info(f"   Note: {num_gpus} GPUs available. Use 'cuda:N' to select specific GPU.")
            return device

    # UNKNOWN DEVICE TYPE
    logger.warning(f"⚠ Unknown device type: '{device_type}'")
    logger.warning("   Supported types: 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'")
    logger.warning("→ Falling back to CPU")
    device = torch.device('cpu')
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


def is_ddp():
    """
    Check if running in Distributed Data Parallel mode.

    DDP is detected by checking if RANK environment variable is set
    (typically by torchrun launcher).

    Returns:
        Boolean indicating if DDP is active
    """
    return int(os.environ.get('RANK', -1)) != -1


def get_dist_info():
    """
    Get distributed training information from environment variables.

    When using torchrun, these environment variables are automatically set:
    - RANK: Global rank of this process (0 to world_size-1)
    - LOCAL_RANK: Local rank on this machine (0 to local_world_size-1)
    - WORLD_SIZE: Total number of processes across all machines

    Returns:
        Tuple of (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size)
        - is_ddp: Boolean indicating if DDP is active
        - ddp_rank: Global rank (0 if not DDP)
        - ddp_local_rank: Local rank (0 if not DDP)
        - ddp_world_size: World size (1 if not DDP)

    Example:
        >>> is_ddp, rank, local_rank, world_size = get_dist_info()
        >>> if rank == 0:
        ...     print("I'm the master process")
    """
    if is_ddp():
        # Verify all required environment variables are present
        required_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']
        if not all(var in os.environ for var in required_vars):
            logger.error(
                "DDP mode detected but missing environment variables. "
                f"Required: {required_vars}"
            )
            raise RuntimeError("Incomplete DDP environment variables")

        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def setup_distributed(device_type='cuda'):
    """
    Initialize distributed training (DDP).

    This should be called before creating the model. It:
    1. Gets distribution info from environment
    2. Initializes the process group
    3. Sets the correct device for this rank

    Args:
        device_type: Device type ('cuda', 'cpu', 'mps')

    Returns:
        Tuple of (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device)

    Example:
        Launch with torchrun:
        ```bash
        torchrun --nproc_per_node=4 train.py
        ```

        In your training script:
        ```python
        is_ddp, rank, local_rank, world_size, device = setup_distributed('cuda')
        model = Model().to(device)
        if is_ddp:
            model = DDP(model, device_ids=[local_rank])
        ```
    """
    is_distributed, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if is_distributed:
        logger.info(f"Initializing DDP: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")

        # DDP only supports CUDA currently
        if device_type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("DDP requires CUDA but torch.cuda.is_available() is False")

            # Set device for this rank
            device = torch.device(f'cuda:{ddp_local_rank}')
            torch.cuda.set_device(device)

            # Initialize process group with NCCL backend (optimized for NVIDIA GPUs)
            import torch.distributed as dist
            dist.init_process_group(backend='nccl')

            if ddp_rank == 0:
                logger.info(f"✓ DDP initialized with NCCL backend ({ddp_world_size} GPUs)")
        else:
            logger.warning(
                f"DDP requested with device_type='{device_type}', but DDP only supports CUDA. "
                "Falling back to single-device training."
            )
            is_distributed = False
            ddp_rank = 0
            ddp_local_rank = 0
            ddp_world_size = 1
            device = torch.device(device_type)
    else:
        # Not distributed - single device
        device = torch.device(device_type)
        logger.info(f"Single-device mode: {device}")

    return is_distributed, ddp_rank, ddp_local_rank, ddp_world_size, device


def cleanup_distributed():
    """
    Clean up distributed training resources.

    Call this at the end of training to properly shut down the process group.

    Example:
        ```python
        try:
            # ... training code ...
            pass
        finally:
            cleanup_distributed()
        ```
    """
    if is_ddp():
        import torch.distributed as dist
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")
