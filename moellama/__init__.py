"""
moellama: Mixture of Experts Language Model

A clean, educational implementation of the Mixture of Experts (MoE) architecture
inspired by LLaMA 4. This package provides modular components for building and
training MoE-based language models.

Main components:
- model: The complete LLaMA4MoE model
- moe: Mixture of Experts layer with routing
- attention: Multi-head self-attention with RoPE
- layers: Basic building blocks (RMSNorm, RoPE, TransformerBlock)
- trainer: Training loop and optimization
- dataset: Data loading and preprocessing
- tokenizer: BPE tokenization
- utils: Configuration and device management
"""

from moellama.model import LLaMA4MoE
from moellama.tokenizer import BPETokenizer
from moellama.trainer import LLaMA4Trainer
from moellama.dataset import TextDataset, prepare_dataset
from moellama.utils import setup_device, load_config
from moellama.logging_setup import setup_logging, get_logger

__version__ = "0.2.0"
__all__ = [
    "LLaMA4MoE",
    "BPETokenizer",
    "LLaMA4Trainer",
    "TextDataset",
    "prepare_dataset",
    "setup_device",
    "load_config",
    "setup_logging",
    "get_logger",
]
