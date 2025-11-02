"""
Multi-Stage Training System for moe_llama.

This module provides a complete training pipeline with multiple stages:
- Base Training: Pretraining on large text corpora
- Midtraining: Adaptation to conversations and task formats
- Fine-Tuning (SFT): Supervised fine-tuning with PEFT/LoRA support

Each stage can use different datasets, learning rates, and optimizations.
"""

from moellama.stages.base_train import BaseTrainingStage
from moellama.stages.mid_train import MidtrainingStage
from moellama.stages.sft_train import SFTTrainingStage
from moellama.stages.pipeline import run_multistage_training

__all__ = [
    'BaseTrainingStage',
    'MidtrainingStage',
    'SFTTrainingStage',
    'run_multistage_training',
]
