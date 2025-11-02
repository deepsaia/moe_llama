"""
PEFT (Parameter-Efficient Fine-Tuning) Utilities.

This module provides integration with the HuggingFace PEFT library for
efficient fine-tuning using methods like LoRA, QLoRA, and prefix tuning.

Key features:
- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Quantized LoRA for even more efficiency
- Prefix Tuning: Learnable prompt prefixes
- Easy adapter management (save/load/merge)

Requires: pip install peft
"""

from loguru import logger
from typing import Dict, Optional, List

import torch
import torch.nn as nn


# Try to import PEFT
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning(
        "PEFT library not found. Install with: pip install peft\n"
        "LoRA/QLoRA fine-tuning will not be available."
    )


class PEFTManager:
    """
    Manager for PEFT (Parameter-Efficient Fine-Tuning) operations.

    Handles:
    - Creating PEFT models with LoRA/QLoRA
    - Saving and loading adapters
    - Merging adapters back into base model
    - Configuration management

    Args:
        method: PEFT method ('lora', 'qlora', 'prefix_tuning')
        config: PEFT configuration dictionary

    Example:
        >>> manager = PEFTManager(method='lora', config=peft_config)
        >>> peft_model = manager.prepare_model(base_model)
        >>> # Train peft_model
        >>> manager.save_adapter(peft_model, "path/to/adapter")
    """

    def __init__(self, method: str, config: Dict):
        """Initialize PEFT manager."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required but not installed. "
                "Install with: pip install peft"
            )

        self.method = method.lower()
        self.config = config

        # Validate method
        supported_methods = ['lora', 'qlora', 'prefix_tuning', 'full']
        if self.method not in supported_methods:
            raise ValueError(
                f"Unsupported PEFT method: {method}. "
                f"Supported: {supported_methods}"
            )

        logger.info(f"PEFTManager initialized with method: {self.method}")

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for PEFT training.

        Args:
            model: Base model to add adapters to

        Returns:
            PEFT model with adapters
        """
        if self.method == 'full':
            logger.info("Full fine-tuning mode, returning model unchanged")
            return model

        logger.info(f"Preparing model for {self.method} fine-tuning...")

        if self.method == 'lora':
            return self._prepare_lora_model(model)
        elif self.method == 'qlora':
            return self._prepare_qlora_model(model)
        elif self.method == 'prefix_tuning':
            return self._prepare_prefix_model(model)

    def _prepare_lora_model(self, model: nn.Module) -> PeftModel:
        """
        Prepare model with LoRA adapters.

        LoRA adds trainable low-rank matrices to attention layers,
        significantly reducing trainable parameters.

        Args:
            model: Base model

        Returns:
            LoRA model
        """
        # Extract LoRA config
        lora_r = self.config.get('lora_r', 8)
        lora_alpha = self.config.get('lora_alpha', 16)
        lora_dropout = self.config.get('lora_dropout', 0.1)
        target_modules = self.config.get('target_modules', ['q_proj', 'v_proj'])
        bias = self.config.get('bias', 'none')

        logger.info(f"LoRA configuration:")
        logger.info(f"  Rank (r): {lora_r}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {lora_dropout}")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Bias: {bias}")

        # Create LoRA config
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        peft_model = get_peft_model(model, peft_config)

        # Print trainable parameters
        self._print_trainable_parameters(peft_model)

        return peft_model

    def _prepare_qlora_model(self, model: nn.Module) -> PeftModel:
        """
        Prepare model with QLoRA (Quantized LoRA).

        QLoRA combines LoRA with quantization for even more memory efficiency.
        Requires bitsandbytes library.

        Args:
            model: Base model

        Returns:
            QLoRA model
        """
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "QLoRA requires bitsandbytes. Install with: pip install bitsandbytes"
            )

        logger.info("Preparing model for QLoRA (quantized training)...")

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Use same LoRA config but with quantization-aware training
        return self._prepare_lora_model(model)

    def _prepare_prefix_model(self, model: nn.Module) -> PeftModel:
        """
        Prepare model with prefix tuning.

        Prefix tuning prepends learnable vectors to the model's inputs.

        Args:
            model: Base model

        Returns:
            Prefix-tuned model
        """
        from peft import PrefixTuningConfig

        # Extract prefix tuning config
        num_virtual_tokens = self.config.get('num_virtual_tokens', 20)
        prefix_projection = self.config.get('prefix_projection', False)

        logger.info(f"Prefix tuning configuration:")
        logger.info(f"  Virtual tokens: {num_virtual_tokens}")
        logger.info(f"  Prefix projection: {prefix_projection}")

        # Create prefix tuning config
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=prefix_projection,
        )

        # Apply prefix tuning
        peft_model = get_peft_model(model, peft_config)

        # Print trainable parameters
        self._print_trainable_parameters(peft_model)

        return peft_model

    def _print_trainable_parameters(self, model: PeftModel):
        """
        Print number of trainable parameters.

        Shows the efficiency gain from using PEFT.
        """
        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_param

        logger.info(
            f"Trainable parameters: {trainable_params:,} / {all_param:,} "
            f"({percentage:.2f}%)"
        )

    def save_adapter(self, model: PeftModel, save_path: str):
        """
        Save PEFT adapter weights.

        Only saves the adapter weights, not the full model.
        This is much more storage-efficient.

        Args:
            model: PEFT model
            save_path: Path to save adapter
        """
        if self.method == 'full':
            logger.warning("Full fine-tuning mode, use regular model.save() instead")
            return

        logger.info(f"Saving {self.method} adapter to: {save_path}")
        model.save_pretrained(save_path)
        logger.info(f"✓ Adapter saved")

    def load_adapter(self, model: nn.Module, adapter_path: str) -> PeftModel:
        """
        Load PEFT adapter weights into model.

        Args:
            model: Base model
            adapter_path: Path to adapter weights

        Returns:
            Model with loaded adapter
        """
        if self.method == 'full':
            logger.warning("Full fine-tuning mode, no adapter to load")
            return model

        logger.info(f"Loading {self.method} adapter from: {adapter_path}")
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"✓ Adapter loaded")

        return peft_model

    def merge_adapter(self, model: PeftModel) -> nn.Module:
        """
        Merge adapter weights back into base model.

        After merging, the adapter is no longer separate and
        the model can be used like a regular model.

        Args:
            model: PEFT model with adapter

        Returns:
            Base model with merged weights
        """
        if self.method == 'full':
            logger.warning("Full fine-tuning mode, no adapter to merge")
            return model

        logger.info(f"Merging {self.method} adapter into base model...")
        merged_model = model.merge_and_unload()
        logger.info(f"✓ Adapter merged")

        return merged_model


def create_peft_manager(method: str, config: Dict) -> Optional[PEFTManager]:
    """
    Factory function to create PEFT manager.

    Args:
        method: PEFT method ('lora', 'qlora', 'prefix_tuning', 'full')
        config: PEFT configuration

    Returns:
        PEFTManager instance or None if full fine-tuning
    """
    if method.lower() == 'full':
        logger.info("Full fine-tuning mode, PEFT not needed")
        return None

    if not PEFT_AVAILABLE:
        logger.error(
            "PEFT library not available. Install with: pip install peft\n"
            "Falling back to full fine-tuning."
        )
        return None

    return PEFTManager(method, config)


def get_lora_target_modules(model_type: str = 'llama') -> List[str]:
    """
    Get recommended LoRA target modules for different model types.

    Args:
        model_type: Type of model ('llama', 'gpt', 'bert', etc.)

    Returns:
        List of module names to target
    """
    # Target modules for different architectures
    targets = {
        'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  # All attention projections
        'gpt': ['c_attn', 'c_proj'],  # GPT-2 style
        'bert': ['query', 'key', 'value'],  # BERT style
        'all_linear': ['Linear'],  # Target all linear layers (more parameters)
    }

    return targets.get(model_type, targets['llama'])
