"""
Midtraining Stage.

This module implements the midtraining stage where a pretrained model is adapted
to conversations, task formats, and domain-specific data.

Key characteristics:
- Loads pretrained base model
- Lower learning rate than base (1e-4 typical)
- Trains on conversations, QA, and task-specific data
- Shorter training (often 1 epoch)
- Teaches special tokens and formats
"""

from loguru import logger
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from moellama.model import LLaMA4MoE
from moellama.trainer import LLaMA4Trainer
from moellama.dataset import prepare_dataset



class MidtrainingStage:
    """
    Midtraining stage for domain/task adaptation.

    This stage takes a pretrained base model and adapts it to:
    - Conversation formats
    - Task-specific patterns (QA, math, code)
    - Special tokens
    - Domain-specific knowledge

    Args:
        config: Full configuration dictionary
        device: Device to train on

    Example:
        >>> # Load base model first
        >>> base_model, tokenizer = load_checkpoint("base_model")
        >>> # Run midtraining
        >>> stage = MidtrainingStage(config, device='cuda')
        >>> model, tokenizer = stage.run(base_model, tokenizer)
    """

    def __init__(self, config: Dict, device: str):
        """Initialize midtraining stage."""
        self.config = config
        self.stage_config = config['training'].get('mid_stage', {})
        self.device = device

        # Check if stage is enabled
        if not self.stage_config.get('enabled', False):
            logger.info("Midtraining stage is disabled, skipping")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("MidtrainingStage initialized")
            logger.info(f"Device: {self.device}")

    def run(
        self,
        base_model: Optional[LLaMA4MoE] = None,
        tokenizer: Optional[any] = None
    ) -> Tuple[LLaMA4MoE, any]:
        """
        Run midtraining stage.

        Process:
        1. Load base model (if not provided)
        2. Prepare midtraining datasets
        3. Continue training with lower LR
        4. Save checkpoint
        5. Return adapted model

        Args:
            base_model: Pretrained base model (loads from checkpoint if None)
            tokenizer: Tokenizer (loads from checkpoint if None)

        Returns:
            Tuple of (model, tokenizer)
        """
        if not self.enabled:
            logger.info("Midtraining stage disabled, returning base model")
            return base_model, tokenizer

        logger.info("="*80)
        logger.info("STARTING MIDTRAINING STAGE")
        logger.info("="*80)

        # Step 1: Load model if not provided
        if base_model is None or tokenizer is None:
            logger.info("Step 1: Loading base model...")
            base_model, tokenizer = self._load_base_model()
            logger.info(f"✓ Base model loaded")
        else:
            logger.info("Step 1: Using provided base model")

        # Step 2: Prepare datasets
        logger.info("Step 2: Preparing midtraining datasets...")
        train_data, eval_data, _ = self._prepare_datasets(tokenizer)
        logger.info(f"✓ Datasets prepared")

        # Step 3: Continue training
        logger.info("Step 3: Midtraining...")
        model = self._train_model(base_model, train_data, eval_data, tokenizer)
        logger.info(f"✓ Midtraining complete")

        # Step 4: Save checkpoint
        if self.stage_config.get('save_checkpoint', True):
            logger.info("Step 4: Saving checkpoint...")
            self._save_checkpoint(model, tokenizer)
            logger.info(f"✓ Checkpoint saved")

        logger.info("="*80)
        logger.info("MIDTRAINING STAGE COMPLETE")
        logger.info("="*80)

        return model, tokenizer

    def _load_base_model(self) -> Tuple[LLaMA4MoE, any]:
        """
        Load pretrained base model from checkpoint.

        Returns:
            Tuple of (model, tokenizer)
        """
        checkpoint_name = self.stage_config.get('load_checkpoint', 'base_model')
        model_path = Path(self.config['paths']['model_path'])
        checkpoint_dir = model_path / checkpoint_name

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Base model checkpoint not found: {checkpoint_dir}. "
                f"Run base training first or specify correct checkpoint."
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

        logger.info(f"Loaded base model from: {checkpoint_dir}")

        return model, tokenizer

    def _prepare_datasets(self, tokenizer):
        """
        Prepare midtraining datasets.

        Args:
            tokenizer: Tokenizer to use

        Returns:
            Tuple of (train_data, eval_data, tokenizer)
        """
        # Create temporary config with mid stage dataset configuration
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
        Continue training model with midtraining configuration.

        Uses lower learning rate and shorter training than base.

        Args:
            model: Base model to continue training
            train_data: Training dataset/dataloader
            eval_data: Evaluation dataset/dataloader
            tokenizer: Tokenizer

        Returns:
            Trained model
        """
        # Get training hyperparameters (typically lower LR)
        batch_size = self.stage_config.get('batch_size', 16)
        learning_rate = self.stage_config.get('learning_rate', 1e-4)  # Lower than base
        epochs = self.stage_config.get('epochs', 1)  # Often just 1 epoch
        eval_steps = self.stage_config.get('eval_steps', 200)

        logger.info(f"Midtraining config:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate} (lower for adaptation)")
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

    def _save_checkpoint(self, model, tokenizer):
        """
        Save midtrained model checkpoint.

        Args:
            model: Midtrained model
            tokenizer: Tokenizer
        """
        checkpoint_name = self.stage_config.get('checkpoint_name', 'mid_model')
        model_path = Path(self.config['paths']['model_path'])

        # Create checkpoint directory
        checkpoint_dir = model_path / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
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
            'stage': 'mid',
        }, model_file)

        # Save tokenizer
        tokenizer_file = checkpoint_dir / "tokenizer.pkl"
        tokenizer.save(str(tokenizer_file))

        logger.info(f"Checkpoint saved to: {checkpoint_dir}")
        logger.info(f"  Model: {model_file}")
        logger.info(f"  Tokenizer: {tokenizer_file}")


def run_midtraining(
    config: Dict,
    device: str,
    base_model: Optional[LLaMA4MoE] = None,
    tokenizer: Optional[any] = None
) -> Tuple[LLaMA4MoE, any]:
    """
    Convenience function to run midtraining stage.

    Args:
        config: Configuration dictionary
        device: Device to train on
        base_model: Pretrained base model (loads if None)
        tokenizer: Tokenizer (loads if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    stage = MidtrainingStage(config, device)
    return stage.run(base_model, tokenizer)
