"""
Multi-Stage Training Script.

This script runs the complete training pipeline:
Base Training → Midtraining → Fine-Tuning (SFT)

Usage:
    python -m scripts.train_multistage
    python -m scripts.train_multistage --config config_multistage.hocon
    python -m scripts.train_multistage --config my_config.hocon --validate-only

The script automatically handles all stage transitions and checkpointing.
"""

import argparse
from loguru import logger
from moellama.logging_setup import setup_logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moellama.utils import load_config
from moellama.stages import run_multistage_training
from moellama.stages.pipeline import validate_multistage_config

# Setup logging
setup_logging()



def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Multi-stage training script for moe_llama'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_multistage.hocon',
        help='Path to configuration file (default: config_multistage.hocon)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Multi-Stage Training Script")
    logger.info("="*80)
    logger.info(f"Config file: {args.config}")

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Validate configuration
    is_valid, error_msg = validate_multistage_config(config)
    if not is_valid:
        logger.error(f"Configuration validation failed: {error_msg}")
        return 1
    else:
        logger.info("✓ Configuration is valid")

    # If validate-only, stop here
    if args.validate_only:
        logger.info("Validation complete. Exiting (--validate-only mode)")
        return 0

    # Print pipeline summary
    logger.info("\nPipeline Summary:")
    training_config = config['training']
    logger.info(f"  Multi-stage: {training_config.get('multi_stage', False)}")
    logger.info(f"  Base Training: {training_config.get('base_stage', {}).get('enabled', True)}")
    logger.info(f"  Midtraining: {training_config.get('mid_stage', {}).get('enabled', False)}")
    logger.info(f"  Fine-Tuning (SFT): {training_config.get('sft_stage', {}).get('enabled', False)}")

    # Run training pipeline
    try:
        logger.info("\nStarting training pipeline...")
        model, tokenizer = run_multistage_training(config)

        logger.info("="*80)
        logger.info("TRAINING COMPLETE ✓")
        logger.info("="*80)
        logger.info(f"Final model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # Print checkpoint locations
        model_path = Path(config['paths']['model_path'])
        logger.info("\nSaved Checkpoints:")
        if training_config.get('base_stage', {}).get('enabled', True):
            base_checkpoint = training_config.get('base_stage', {}).get('checkpoint_name', 'base_model')
            logger.info(f"  Base: {model_path / base_checkpoint}")
        if training_config.get('mid_stage', {}).get('enabled', False):
            mid_checkpoint = training_config.get('mid_stage', {}).get('checkpoint_name', 'mid_model')
            logger.info(f"  Mid: {model_path / mid_checkpoint}")
        if training_config.get('sft_stage', {}).get('enabled', False):
            sft_checkpoint = training_config.get('sft_stage', {}).get('checkpoint_name', 'sft_model')
            logger.info(f"  SFT: {model_path / sft_checkpoint}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
