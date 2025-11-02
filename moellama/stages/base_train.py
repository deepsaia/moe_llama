"""
Base Training Stage (Pretraining).

This module implements the base pretraining stage where the model learns
general language understanding from large text corpora.

Key characteristics:
- Trains on large, diverse text datasets
- High learning rate (3e-4 typical)
- Multiple epochs over data
- Focus on next-token prediction
- Saves base model checkpoint
"""

from loguru import logger
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from moellama.model import LLaMA4MoE
from moellama.trainer import LLaMA4Trainer
from moellama.dataset import prepare_dataset
from moellama.utils import setup_device



class BaseTrainingStage:
    """
    Base training (pretraining) stage.

    This stage trains a model from scratch on large text corpora to learn
    general language understanding.

    Args:
        config: Full configuration dictionary
        device: Device to train on (auto-detected if None)

    Example:
        >>> stage = BaseTrainingStage(config)
        >>> model, tokenizer = stage.run()
        >>> # Model is now pretrained and saved
    """

    def __init__(self, config: Dict, device: Optional[str] = None):
        """Initialize base training stage."""
        self.config = config
        self.stage_config = config['training'].get('base_stage', {})

        # Check if stage is enabled
        if not self.stage_config.get('enabled', True):
            raise ValueError("Base training stage is not enabled in config")

        # Setup device
        self.device = device or setup_device(config)

        logger.info("BaseTrainingStage initialized")
        logger.info(f"Device: {self.device}")

    def run(self) -> Tuple[LLaMA4MoE, any]:
        """
        Run base training stage.

        Process:
        1. Load/prepare datasets
        2. Create model
        3. Train model
        4. Save checkpoint
        5. Return trained model and tokenizer

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info("="*80)
        logger.info("STARTING BASE TRAINING STAGE (PRETRAINING)")
        logger.info("="*80)

        # Step 1: Prepare datasets
        logger.info("Step 1: Preparing datasets...")
        train_data, eval_data, tokenizer = self._prepare_datasets()
        logger.info(f"✓ Datasets prepared")
        logger.info(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

        # Step 2: Create model
        logger.info("Step 2: Creating model...")
        model = self._create_model(tokenizer)
        logger.info(f"✓ Model created")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Step 3: Train model
        logger.info("Step 3: Training model...")
        model = self._train_model(model, train_data, eval_data, tokenizer)
        logger.info(f"✓ Training complete")

        # Step 4: Save checkpoint
        if self.stage_config.get('save_checkpoint', True):
            logger.info("Step 4: Saving checkpoint...")
            self._save_checkpoint(model, tokenizer)
            logger.info(f"✓ Checkpoint saved")

        logger.info("="*80)
        logger.info("BASE TRAINING STAGE COMPLETE")
        logger.info("="*80)

        return model, tokenizer

    def _prepare_datasets(self):
        """
        Prepare training and evaluation datasets.

        Creates a temporary config with base stage dataset configuration.
        """
        # Create temporary config for dataset preparation
        temp_config = {
            'training': self.stage_config.copy(),
            'model': self.config['model'],
        }

        # Ensure data_dir is set
        if 'data_dir' not in temp_config['training']:
            temp_config['training']['data_dir'] = self.config['training'].get('data_dir', 'dataset')

        # Prepare datasets
        train_data, eval_data, tokenizer = prepare_dataset(temp_config, device=self.device)

        return train_data, eval_data, tokenizer

    def _create_model(self, tokenizer) -> LLaMA4MoE:
        """
        Create model for base training.

        Args:
            tokenizer: Trained tokenizer

        Returns:
            LLaMA4MoE model
        """
        model_config = self.config['model'].copy()

        # Update vocab size from tokenizer
        model_config['vocab_size'] = tokenizer.vocab_size

        # Create model
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

        # Move to device
        model = model.to(self.device)

        return model

    def _train_model(self, model, train_data, eval_data, tokenizer):
        """
        Train model with base training configuration.

        Args:
            model: Model to train
            train_data: Training dataset/dataloader
            eval_data: Evaluation dataset/dataloader
            tokenizer: Tokenizer

        Returns:
            Trained model
        """
        # Get training hyperparameters
        batch_size = self.stage_config.get('batch_size', self.config['training']['batch_size'])
        learning_rate = self.stage_config.get('learning_rate', self.config['training']['learning_rate'])
        epochs = self.stage_config.get('epochs', self.config['training']['epochs'])
        eval_steps = self.stage_config.get('eval_steps', self.config['training'].get('eval_steps', 100))

        logger.info(f"Training config:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
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
        Save model checkpoint.

        Args:
            model: Trained model
            tokenizer: Tokenizer
        """
        checkpoint_name = self.stage_config.get('checkpoint_name', 'base_model')
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
            'stage': 'base',
        }, model_file)

        # Save tokenizer
        tokenizer_file = checkpoint_dir / "tokenizer.pkl"
        tokenizer.save(str(tokenizer_file))

        logger.info(f"Checkpoint saved to: {checkpoint_dir}")
        logger.info(f"  Model: {model_file}")
        logger.info(f"  Tokenizer: {tokenizer_file}")


def run_base_training(config: Dict, device: Optional[str] = None) -> Tuple[LLaMA4MoE, any]:
    """
    Convenience function to run base training stage.

    Args:
        config: Configuration dictionary
        device: Device to train on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    stage = BaseTrainingStage(config, device)
    return stage.run()
