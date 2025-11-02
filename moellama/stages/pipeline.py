"""
Multi-Stage Training Pipeline Orchestrator.

This module orchestrates the complete training pipeline:
Base Training → Midtraining → Fine-Tuning (SFT)

Each stage can be enabled/disabled independently in the configuration.
The pipeline handles checkpointing and stage transitions automatically.
"""

from loguru import logger
import time
from datetime import timedelta
from typing import Dict, Tuple, Optional

from moellama.model import LLaMA4MoE
from moellama.stages.base_train import BaseTrainingStage
from moellama.stages.mid_train import MidtrainingStage
from moellama.stages.sft_train import SFTTrainingStage
from moellama.utils import setup_device



class MultiStageTrainingPipeline:
    """
    Orchestrates multi-stage training pipeline.

    This pipeline runs up to 3 stages sequentially:
    1. Base Training: Pretraining on large text corpora
    2. Midtraining: Adaptation to conversations/tasks
    3. Fine-Tuning: Supervised fine-tuning with PEFT

    Each stage can be enabled/disabled in the configuration.
    Models and checkpoints are passed between stages automatically.

    Args:
        config: Full configuration dictionary

    Example:
        >>> pipeline = MultiStageTrainingPipeline(config)
        >>> final_model, tokenizer = pipeline.run()
        >>> # final_model is the result of all enabled stages
    """

    def __init__(self, config: Dict):
        """Initialize pipeline."""
        self.config = config

        # Check if multi-stage is enabled
        training_config = config.get('training', {})
        self.multi_stage = training_config.get('multi_stage', False)

        if not self.multi_stage:
            raise ValueError(
                "Multi-stage training not enabled in config. "
                "Set training.multi_stage = true"
            )

        # Setup device
        self.device = setup_device(config)

        # Check which stages are enabled
        self.base_enabled = training_config.get('base_stage', {}).get('enabled', True)
        self.mid_enabled = training_config.get('mid_stage', {}).get('enabled', False)
        self.sft_enabled = training_config.get('sft_stage', {}).get('enabled', False)

        logger.info("MultiStageTrainingPipeline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Stages enabled:")
        logger.info(f"  Base Training: {self.base_enabled}")
        logger.info(f"  Midtraining: {self.mid_enabled}")
        logger.info(f"  Fine-Tuning (SFT): {self.sft_enabled}")

        if not any([self.base_enabled, self.mid_enabled, self.sft_enabled]):
            raise ValueError("At least one training stage must be enabled")

    def run(self) -> Tuple[LLaMA4MoE, any]:
        """
        Run the complete training pipeline.

        Executes all enabled stages in sequence:
        1. Base training (if enabled)
        2. Midtraining (if enabled)
        3. Fine-tuning (if enabled)

        Returns:
            Tuple of (final_model, tokenizer)
        """
        logger.info("="*100)
        logger.info("STARTING MULTI-STAGE TRAINING PIPELINE")
        logger.info("="*100)

        start_time = time.time()
        model = None
        tokenizer = None

        # Stage 1: Base Training
        if self.base_enabled:
            logger.info("\n" + "🚀 "*20)
            logger.info("STAGE 1/3: BASE TRAINING (PRETRAINING)")
            logger.info("🚀 "*20 + "\n")
            stage_start = time.time()

            stage = BaseTrainingStage(self.config, self.device)
            model, tokenizer = stage.run()

            stage_time = time.time() - stage_start
            logger.info(f"✓ Base training completed in {timedelta(seconds=int(stage_time))}")
        else:
            logger.info("⏭️  Skipping base training (disabled)")

        # Stage 2: Midtraining
        if self.mid_enabled:
            logger.info("\n" + "🎯 "*20)
            logger.info("STAGE 2/3: MIDTRAINING")
            logger.info("🎯 "*20 + "\n")
            stage_start = time.time()

            stage = MidtrainingStage(self.config, self.device)
            model, tokenizer = stage.run(model, tokenizer)

            stage_time = time.time() - stage_start
            logger.info(f"✓ Midtraining completed in {timedelta(seconds=int(stage_time))}")
        else:
            logger.info("⏭️  Skipping midtraining (disabled)")

        # Stage 3: Fine-Tuning (SFT)
        if self.sft_enabled:
            logger.info("\n" + "✨ "*20)
            logger.info("STAGE 3/3: SUPERVISED FINE-TUNING (SFT)")
            logger.info("✨ "*20 + "\n")
            stage_start = time.time()

            stage = SFTTrainingStage(self.config, self.device)
            model, tokenizer = stage.run(model, tokenizer)

            stage_time = time.time() - stage_start
            logger.info(f"✓ Fine-tuning completed in {timedelta(seconds=int(stage_time))}")
        else:
            logger.info("⏭️  Skipping fine-tuning (disabled)")

        # Summary
        total_time = time.time() - start_time
        logger.info("\n" + "="*100)
        logger.info("MULTI-STAGE TRAINING PIPELINE COMPLETE")
        logger.info("="*100)
        logger.info(f"Total training time: {timedelta(seconds=int(total_time))}")
        logger.info(f"Final model stage: {self._get_final_stage()}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("="*100)

        return model, tokenizer

    def _get_final_stage(self) -> str:
        """Get the name of the final enabled stage."""
        if self.sft_enabled:
            return "SFT"
        elif self.mid_enabled:
            return "Midtraining"
        elif self.base_enabled:
            return "Base"
        else:
            return "Unknown"

    def get_pipeline_summary(self) -> Dict:
        """
        Get summary of pipeline configuration.

        Returns:
            Dictionary with pipeline info
        """
        return {
            'multi_stage': self.multi_stage,
            'device': self.device,
            'stages': {
                'base': {
                    'enabled': self.base_enabled,
                    'config': self.config.get('training', {}).get('base_stage', {})
                },
                'mid': {
                    'enabled': self.mid_enabled,
                    'config': self.config.get('training', {}).get('mid_stage', {})
                },
                'sft': {
                    'enabled': self.sft_enabled,
                    'config': self.config.get('training', {}).get('sft_stage', {})
                },
            }
        }


def run_multistage_training(config: Dict) -> Tuple[LLaMA4MoE, any]:
    """
    Convenience function to run multi-stage training pipeline.

    Args:
        config: Configuration dictionary with multi_stage enabled

    Returns:
        Tuple of (final_model, tokenizer)

    Example:
        >>> config = load_config("config_multistage.hocon")
        >>> model, tokenizer = run_multistage_training(config)
    """
    pipeline = MultiStageTrainingPipeline(config)
    return pipeline.run()


def validate_multistage_config(config: Dict) -> Tuple[bool, str]:
    """
    Validate multi-stage training configuration.

    Checks:
    - multi_stage is enabled
    - At least one stage is enabled
    - Stage configurations are valid
    - Checkpoints are properly configured

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    training_config = config.get('training', {})

    # Check if multi-stage is enabled
    if not training_config.get('multi_stage', False):
        return False, "multi_stage must be set to true in training config"

    # Check if at least one stage is enabled
    base_enabled = training_config.get('base_stage', {}).get('enabled', True)
    mid_enabled = training_config.get('mid_stage', {}).get('enabled', False)
    sft_enabled = training_config.get('sft_stage', {}).get('enabled', False)

    if not any([base_enabled, mid_enabled, sft_enabled]):
        return False, "At least one training stage must be enabled"

    # Check stage order constraints
    if sft_enabled and not (base_enabled or mid_enabled):
        # SFT requires a checkpoint to load from
        load_checkpoint = training_config.get('sft_stage', {}).get('load_checkpoint')
        if not load_checkpoint:
            return False, "SFT stage requires load_checkpoint or a previous stage to be enabled"

    if mid_enabled and not base_enabled:
        # Mid requires a checkpoint to load from
        load_checkpoint = training_config.get('mid_stage', {}).get('load_checkpoint')
        if not load_checkpoint:
            return False, "Midtraining requires load_checkpoint or base stage to be enabled"

    # Validate each enabled stage has dataset_mixture
    for stage_name, stage_key in [
        ('base', 'base_stage'),
        ('mid', 'mid_stage'),
        ('sft', 'sft_stage')
    ]:
        stage_config = training_config.get(stage_key, {})
        if stage_config.get('enabled', False):
            if 'dataset_mixture' not in stage_config and 'dataset' not in stage_config:
                return False, f"{stage_name} stage is enabled but has no dataset configured"

    return True, "Configuration is valid"
