"""
Supervised Fine-Tuning (SFT) Stage with PEFT Support.

This module implements the supervised fine-tuning stage where a pretrained/midtrained
model is fine-tuned on high-quality, task-specific data using full or parameter-efficient
methods (LoRA, QLoRA, etc.).

Key characteristics:
- Loads pretrained or midtrained model
- Very low learning rate (5e-5 typical)
- Supports full fine-tuning or PEFT (LoRA/QLoRA)
- Trains on high-quality instruction/task data
- Often single epoch
- Can fine-tune large models on single GPU with PEFT
"""

from loguru import logger
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from moellama.model import LLaMA4MoE
from moellama.trainer import LLaMA4Trainer
from moellama.dataset import prepare_dataset
from moellama.peft_utils import create_peft_manager, PEFT_AVAILABLE



class SFTTrainingStage:
    """
    Supervised Fine-Tuning (SFT) stage with PEFT support.

    This stage fine-tunes a pretrained model on high-quality data using:
    - Full fine-tuning: Train all parameters
    - LoRA: Train low-rank adapters (memory efficient)
    - QLoRA: Quantized LoRA (even more efficient)
    - Prefix Tuning: Learn prompt prefixes

    Args:
        config: Full configuration dictionary
        device: Device to train on

    Example:
        >>> # Load midtrained model
        >>> mid_model, tokenizer = load_checkpoint("mid_model")
        >>> # Run SFT with LoRA
        >>> stage = SFTTrainingStage(config, device='cuda')
        >>> model, tokenizer = stage.run(mid_model, tokenizer)
    """

    def __init__(self, config: Dict, device: str):
        """Initialize SFT stage."""
        self.config = config
        self.stage_config = config['training'].get('sft_stage', {})
        self.device = device

        # Check if stage is enabled
        if not self.stage_config.get('enabled', False):
            logger.info("SFT stage is disabled, skipping")
            self.enabled = False
        else:
            self.enabled = True

            # Get PEFT method
            self.method = self.stage_config.get('method', 'full').lower()

            logger.info("SFTTrainingStage initialized")
            logger.info(f"Device: {self.device}")
            logger.info(f"Method: {self.method}")

            # Create PEFT manager if needed
            if self.method != 'full':
                peft_config = self.stage_config.get('peft', {})
                self.peft_manager = create_peft_manager(self.method, peft_config)
            else:
                self.peft_manager = None

    def run(
        self,
        base_model: Optional[LLaMA4MoE] = None,
        tokenizer: Optional[any] = None
    ) -> Tuple[LLaMA4MoE, any]:
        """
        Run SFT stage.

        Process:
        1. Load model (from mid or base checkpoint)
        2. Apply PEFT if configured
        3. Prepare SFT datasets
        4. Fine-tune with very low LR
        5. Save checkpoint (merge adapter if PEFT)
        6. Return fine-tuned model

        Args:
            base_model: Model to fine-tune (loads from checkpoint if None)
            tokenizer: Tokenizer (loads from checkpoint if None)

        Returns:
            Tuple of (model, tokenizer)
        """
        if not self.enabled:
            logger.info("SFT stage disabled, returning model")
            return base_model, tokenizer

        logger.info("="*80)
        logger.info("STARTING SUPERVISED FINE-TUNING (SFT) STAGE")
        logger.info("="*80)

        # Step 1: Load model if not provided
        if base_model is None or tokenizer is None:
            logger.info("Step 1: Loading base/mid model...")
            base_model, tokenizer = self._load_model()
            logger.info(f"✓ Model loaded")
        else:
            logger.info("Step 1: Using provided model")

        # Step 2: Apply PEFT if configured
        if self.peft_manager is not None:
            logger.info(f"Step 2: Applying {self.method} adapters...")
            model = self.peft_manager.prepare_model(base_model)
            logger.info(f"✓ PEFT applied")
        else:
            logger.info("Step 2: Full fine-tuning mode (no PEFT)")
            model = base_model

        # Step 3: Prepare datasets
        logger.info("Step 3: Preparing SFT datasets...")
        train_data, eval_data, _ = self._prepare_datasets(tokenizer)
        logger.info(f"✓ Datasets prepared")

        # Step 4: Fine-tune
        logger.info("Step 4: Fine-tuning...")
        model = self._train_model(model, train_data, eval_data, tokenizer)
        logger.info(f"✓ Fine-tuning complete")

        # Step 5: Save checkpoint
        if self.stage_config.get('save_checkpoint', True):
            logger.info("Step 5: Saving checkpoint...")
            self._save_checkpoint(model, tokenizer, base_model)
            logger.info(f"✓ Checkpoint saved")

        logger.info("="*80)
        logger.info("SFT STAGE COMPLETE")
        logger.info("="*80)

        return model, tokenizer

    def _load_model(self) -> Tuple[LLaMA4MoE, any]:
        """
        Load model from checkpoint.

        Tries to load from mid checkpoint first, falls back to base.

        Returns:
            Tuple of (model, tokenizer)
        """
        checkpoint_name = self.stage_config.get('load_checkpoint', 'mid_model')
        model_path = Path(self.config['paths']['model_path'])
        checkpoint_dir = model_path / checkpoint_name

        # Try mid checkpoint first
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_dir}")
            # Try base checkpoint as fallback
            checkpoint_name = 'base_model'
            checkpoint_dir = model_path / checkpoint_name
            logger.info(f"Trying fallback checkpoint: {checkpoint_dir}")

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint found. Run base/mid training first or specify correct checkpoint."
            )

        # Load model
        model_file = checkpoint_dir / "model.pt"
        checkpoint = torch.load(model_file, map_location=self.device)

        # Create model with saved config
        model_config = checkpoint['model_config']
        model = LLaMA4MoE(
            vocab_size=model_config['vocab_size'],
            dim=model_config['dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            num_experts=model_config['num_experts'],
            top_k=model_config['top_k'],
            max_seq_len=model_config.get('max_seq_len', 512),
            dropout=model_config.get('dropout', 0.1),
            shared_expert=model_config.get('shared_expert', False),
            load_balancing_loss_coef=model_config.get('load_balancing_loss_coef', 0.01)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # Load tokenizer
        tokenizer_file = checkpoint_dir / "tokenizer.pkl"
        from moellama.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(str(tokenizer_file))

        logger.info(f"Loaded model from: {checkpoint_dir}")
        logger.info(f"  Stage: {checkpoint.get('stage', 'unknown')}")

        return model, tokenizer

    def _prepare_datasets(self, tokenizer):
        """
        Prepare SFT datasets.

        Args:
            tokenizer: Tokenizer to use

        Returns:
            Tuple of (train_data, eval_data, tokenizer)
        """
        # Create temporary config with SFT dataset configuration
        temp_config = {
            'training': self.stage_config.copy(),
            'model': self.config['model'],
        }

        # Ensure data_dir is set
        if 'data_dir' not in temp_config['training']:
            temp_config['training']['data_dir'] = self.config['training'].get('data_dir', 'dataset')

        # Prepare datasets with existing tokenizer
        train_data, eval_data, _ = prepare_dataset(temp_config, tokenizer=tokenizer, device=self.device)

        return train_data, eval_data, tokenizer

    def _train_model(self, model, train_data, eval_data, tokenizer):
        """
        Fine-tune model with SFT configuration.

        Uses very low learning rate for careful adaptation.

        Args:
            model: Model to fine-tune (possibly with PEFT)
            train_data: Training dataset/dataloader
            eval_data: Evaluation dataset/dataloader
            tokenizer: Tokenizer

        Returns:
            Fine-tuned model
        """
        # Get training hyperparameters (very low LR)
        batch_size = self.stage_config.get('batch_size', 8)  # Smaller for SFT
        learning_rate = self.stage_config.get('learning_rate', 5e-5)  # Very low
        epochs = self.stage_config.get('epochs', 1)  # Often just 1 epoch
        eval_steps = self.stage_config.get('eval_steps', 100)

        logger.info(f"SFT config:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate} (very low for fine-tuning)")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Eval steps: {eval_steps}")

        # Create trainer
        trainer = LLaMA4Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            device=self.device,
            eval_steps=eval_steps,
        )

        # Train
        trainer.train()

        return model

    def _save_checkpoint(self, model, tokenizer, base_model=None):
        """
        Save fine-tuned model checkpoint.

        If using PEFT:
        - Saves adapter separately
        - Optionally merges adapter into base model

        Args:
            model: Fine-tuned model (possibly with PEFT)
            tokenizer: Tokenizer
            base_model: Original base model (for PEFT merge)
        """
        checkpoint_name = self.stage_config.get('checkpoint_name', 'sft_model')
        model_path = Path(self.config['paths']['model_path'])

        # Create checkpoint directory
        checkpoint_dir = model_path / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Handle PEFT models differently
        if self.peft_manager is not None:
            logger.info(f"Saving {self.method} fine-tuned model...")

            # Save adapter
            adapter_dir = checkpoint_dir / "adapter"
            self.peft_manager.save_adapter(model, str(adapter_dir))

            # Option to merge adapter into base model
            if self.stage_config.get('merge_adapter', True):
                logger.info("Merging adapter into base model...")
                merged_model = self.peft_manager.merge_adapter(model)

                # Save merged model
                model_file = checkpoint_dir / "model.pt"
                torch.save({
                    'model_state_dict': merged_model.state_dict(),
                    'model_config': {
                        'vocab_size': merged_model.vocab_size,
                        'dim': merged_model.dim,
                        'num_layers': merged_model.num_layers,
                        'num_heads': self.config['model']['num_heads'],
                        'num_experts': merged_model.num_experts,
                        'top_k': self.config['model']['top_k'],
                        'max_seq_len': merged_model.max_seq_len,
                        'dropout': self.config['model'].get('dropout', 0.1),
                        'shared_expert': merged_model.shared_expert,
                        'load_balancing_loss_coef': merged_model.load_balancing_loss_coef,
                    },
                    'stage': 'sft',
                    'method': self.method,
                }, model_file)
                logger.info(f"✓ Merged model saved to: {model_file}")
        else:
            # Full fine-tuning: save normally
            model_file = checkpoint_dir / "model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'vocab_size': model.vocab_size,
                    'dim': model.dim,
                    'num_layers': model.num_layers,
                    'num_heads': self.config['model']['num_heads'],
                    'num_experts': model.num_experts,
                    'top_k': self.config['model']['top_k'],
                    'max_seq_len': model.max_seq_len,
                    'dropout': self.config['model'].get('dropout', 0.1),
                    'shared_expert': model.shared_expert,
                    'load_balancing_loss_coef': model.load_balancing_loss_coef,
                },
                'stage': 'sft',
                'method': 'full',
            }, model_file)

        # Save tokenizer
        tokenizer_file = checkpoint_dir / "tokenizer.pkl"
        tokenizer.save(str(tokenizer_file))

        logger.info(f"Checkpoint saved to: {checkpoint_dir}")


def run_sft_training(
    config: Dict,
    device: str,
    base_model: Optional[LLaMA4MoE] = None,
    tokenizer: Optional[any] = None
) -> Tuple[LLaMA4MoE, any]:
    """
    Convenience function to run SFT stage.

    Args:
        config: Configuration dictionary
        device: Device to train on
        base_model: Model to fine-tune (loads if None)
        tokenizer: Tokenizer (loads if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    stage = SFTTrainingStage(config, device)
    return stage.run(base_model, tokenizer)
