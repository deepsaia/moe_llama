"""
Training script for the Mixture of Experts language model.

This script handles the complete training pipeline:
1. Load configuration
2. Setup device and random seeds
3. Prepare dataset and tokenizer
4. Create model and optimizer
5. Train the model
6. Save checkpoints

Usage:
    python -m scripts.train
    python -m scripts.train --config custom_config.hocon
"""

import argparse
import logging
import os
import random

import numpy as np
import torch

from moellama import (
    LLaMA4MoE,
    BPETokenizer,
    LLaMA4Trainer,
    prepare_dataset,
    setup_device,
    load_config
)
from moellama.utils import log_model_info, set_seed

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a Mixture of Experts language model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.hocon",
        help="Path to configuration file (default: config.hocon)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Starting MoE Language Model Training")
    logger.info("="*60)

    try:
        # Set random seed for reproducibility
        set_seed(args.seed)

        # Load configuration
        config = load_config(args.config)

        # Setup device
        device = setup_device(config)

        # Prepare dataset
        logger.info("Preparing dataset...")
        train_dataset, eval_dataset, tokenizer = prepare_dataset(config)

        # Update vocabulary size in config
        vocab_size = len(tokenizer)
        config.put("model.vocab_size", vocab_size)
        logger.info(f"Vocabulary size: {vocab_size}")

        # Create model
        logger.info("Creating model...")
        model = LLaMA4MoE(
            vocab_size=vocab_size,
            dim=config["model"]["dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            num_experts=config["model"]["num_experts"],
            top_k=config["model"]["top_k"],
            max_seq_len=config["model"]["max_seq_len"],
            dropout=config["model"]["dropout"],
            shared_expert=config["model"]["shared_expert"],
            load_balancing_loss_coef=config["model"]["load_balancing_loss_coef"]
        )

        # Log model information
        log_model_info(model)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=0.01
        )

        # Configure multi-GPU training if requested
        device_config = config.get('device', {})
        use_data_parallel = device_config.get('use_data_parallel', False)
        gpu_ids = device_config.get('gpu_ids', [0])

        # Create trainer
        trainer = LLaMA4Trainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            use_data_parallel=use_data_parallel,
            gpu_ids=gpu_ids,
            config=config
        )

        # Train model
        logger.info("Starting training...")
        logger.info(f"Training for {config['training']['epochs']} epochs")

        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=config["training"]["batch_size"],
            epochs=config["training"]["epochs"],
            eval_steps=config["training"]["eval_steps"],
            output_dir=config["paths"]["output_dir"],
            num_workers=config["training"].get("num_workers", 4)
        )

        # Generate training history plot
        logger.info("Generating training history plot...")
        trainer.plot_training_history(config["paths"]["output_dir"])

        # Generate sample text to verify the model works
        logger.info("\n" + "="*60)
        logger.info("Generating sample text...")
        logger.info("="*60)

        prompt = "To be or not to be"
        input_ids = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )

        generated_text = tokenizer.decode(generated_ids[0].tolist())
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text[len(prompt):]}")

        # Save final model
        logger.info("\n" + "="*60)
        logger.info("Saving final model...")
        trainer.save_model(config["paths"]["model_path"])

        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
