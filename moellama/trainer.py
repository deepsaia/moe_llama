"""
Training loop and optimization for the MoE language model.

This module handles the complete training process including:
- Training loop with gradient accumulation
- Mixed precision training (AMP) on CUDA
- Multi-GPU support with DataParallel
- Evaluation and checkpointing
- Training history visualization
"""

import os
import math
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LLaMA4Trainer:
    """
    Trainer class for the LLaMA 4 MoE model.

    Handles the complete training lifecycle:
    - Forward/backward passes with mixed precision
    - Gradient clipping for stability
    - Periodic evaluation
    - Model checkpointing
    - Training metrics tracking

    Args:
        model: The LLaMA4MoE model to train
        tokenizer: Tokenizer for encoding/decoding
        optimizer: Optimizer (typically AdamW)
        device: Device to train on ('cuda', 'cpu', 'mps')
        use_data_parallel: Whether to use DataParallel for multi-GPU
        gpu_ids: List of GPU IDs to use
        config: Configuration dictionary
    """

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_data_parallel=False,
        gpu_ids=None,
        config=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = gpu_ids
        self.config = config

        # Mixed precision training scaler (CUDA only)
        self.scaler = GradScaler() if self.device == 'cuda' else None

        # Setup multi-GPU training if requested
        if use_data_parallel and gpu_ids and len(gpu_ids) > 1 and torch.cuda.is_available():
            logger.info(f"Using DataParallel with GPUs: {gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)

        self.model.to(device)

        # Validate tokenizer
        if tokenizer is not None:
            logger.info(f"Trainer initialized with pad_token_id={tokenizer.pad_token_id}")
            if tokenizer.pad_token_id is None:
                raise RuntimeError("Tokenizer pad_token_id is None")

        logger.info(f"Trainer initialized on device: {device}")

        # Training history for visualization
        self.history = {
            'loss': [],
            'perplexity': [],
            'load_balancing_loss': []
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to pad sequences in a batch to same length.

        Finds the maximum length in the batch and pads all sequences
        to that length with zeros (padding token).

        Args:
            batch: List of tensors with varying lengths

        Returns:
            Batched tensor [batch_size, max_len]
        """
        # Find max length in batch
        max_len = max(len(item) for item in batch)

        padded = []
        for item in batch:
            # Pad sequence to max length
            num_pad = max_len - len(item)
            padding_tensor = torch.zeros(num_pad, dtype=torch.long, device=item.device)
            padded_item = torch.cat([item, padding_tensor], dim=0)
            padded.append(padded_item)

        # Stack into batch
        return torch.stack(padded)

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=16,
        epochs=3,
        eval_steps=100,
        save_steps=None,
        output_dir='./model',
        num_workers=4
    ):
        """
        Train the model.

        Process:
        1. Create data loaders
        2. For each epoch:
            a. Train on all batches
            b. Periodically evaluate
            c. Periodically save checkpoints
        3. Save final model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            batch_size: Number of sequences per batch
            epochs: Number of training epochs
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps (None = only at epoch end)
            output_dir: Directory to save checkpoints
            num_workers: Number of data loading workers
        """
        logger.info("Starting training")
        logger.info(
            f"Config: batch_size={batch_size}, epochs={epochs}, "
            f"eval_steps={eval_steps}, save_steps={save_steps}"
        )

        # Setup data loader
        pin_memory = (self.device == 'cuda')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,  # Faster GPU transfer
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=LLaMA4Trainer.collate_fn,
        )

        global_step = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            epoch_load_balancing_losses = []

            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch.to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass (with mixed precision on CUDA)
                if self.device == 'cuda':
                    with autocast():
                        outputs = self.model(input_ids, labels=input_ids, training=True)
                        loss = outputs["loss"]
                        load_balancing_loss = outputs["load_balancing_loss"]

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # Optimizer step with scaled gradients
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # CPU or MPS: full precision
                    outputs = self.model(input_ids, labels=input_ids, training=True)
                    loss = outputs["loss"]
                    load_balancing_loss = outputs["load_balancing_loss"]

                    # Backward pass
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # Optimizer step
                    self.optimizer.step()

                # Track metrics
                epoch_losses.append(loss.item())
                epoch_load_balancing_losses.append(load_balancing_loss.item())

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lb_loss': f"{load_balancing_loss.item():.4f}"
                })

                global_step += 1

                # Periodic evaluation
                if eval_dataset and global_step % eval_steps == 0:
                    eval_loss, eval_ppl = self.evaluate(eval_dataset, batch_size, num_workers)
                    logger.info(
                        f"Step {global_step} | "
                        f"Eval Loss: {eval_loss:.4f} | "
                        f"Perplexity: {eval_ppl:.2f}"
                    )
                    self.model.train()  # Back to training mode

                # Periodic checkpointing
                if save_steps and global_step % save_steps == 0:
                    self.save_model(output_dir)

            # Save at end of epoch
            self.save_model(output_dir)

            # Log epoch metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_lb_loss = sum(epoch_load_balancing_losses) / len(epoch_load_balancing_losses)
            logger.info(
                f"Epoch {epoch+1} completed | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Avg LB Loss: {avg_lb_loss:.4f}"
            )

    def evaluate(self, eval_dataset, batch_size=16, num_workers=4):
        """
        Evaluate the model on a dataset.

        Computes average loss and perplexity over the entire dataset.

        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers

        Returns:
            Tuple of (avg_loss, perplexity)
        """
        logger.info("Starting evaluation")
        self.model.eval()

        # Validate tokenizer
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is None")

        # Setup data loader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda'),
            collate_fn=LLaMA4Trainer.collate_fn,
        )

        total_loss = 0
        total_tokens = 0

        # Evaluate without gradients
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, labels=input_ids, training=False)
                loss = outputs["loss"]

                # Count non-padding tokens
                num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(f"Evaluation: Avg Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")
        return avg_loss, perplexity

    def save_model(self, output_dir, dataset=None):
        """
        Save model checkpoint and tokenizer.

        Saves:
        - Model state dict with timestamp
        - Tokenizer vocabulary

        Args:
            output_dir: Directory to save to
            dataset: Dataset name (for filename, optional)
        """
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Get dataset name from config if not provided
        dataset = dataset or self.config.get("training", {}).get("dataset", "unknown")

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{dataset}_{timestamp}.pt"

        # Handle DataParallel wrapper
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save model state dict
        torch.save(
            model_to_save.state_dict(),
            os.path.join(output_dir, model_filename)
        )

        # Save tokenizer
        vocab_path = os.path.join(output_dir, f"vocab_{dataset}.txt")
        self.tokenizer.save_vocab(vocab_path)

        logger.info(f"Model saved: {model_filename}")
        logger.info(f"Tokenizer saved: {vocab_path}")

    def plot_training_history(self, output_path):
        """
        Plot and save training metrics.

        Creates a 3-panel figure showing:
        - Training loss over time
        - Evaluation perplexity over time
        - Load balancing loss over time

        Args:
            output_path: Directory to save plot
        """
        logger.info("Plotting training history")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Loss plot
        axes[0].plot(self.history['loss'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # Perplexity plot
        if self.history['perplexity']:
            axes[1].plot(self.history['perplexity'])
            axes[1].set_title('Evaluation Perplexity')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Perplexity')
            axes[1].grid(True)

        # Load balancing loss plot
        axes[2].plot(self.history['load_balancing_loss'])
        axes[2].set_title('Load Balancing Loss')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)

        plt.tight_layout()

        # Save plot
        save_path = Path(output_path) / 'training_history.png'
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Training history plot saved: {save_path}")
