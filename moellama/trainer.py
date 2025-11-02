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
from loguru import logger
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None



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
        config: Configuration dictionary
        use_ddp: Whether to use DistributedDataParallel for multi-GPU
        ddp_rank: Global rank in DDP (0 to world_size-1)
        ddp_local_rank: Local rank on this machine (for device selection)
        ddp_world_size: Total number of processes in DDP
        use_compile: Whether to use torch.compile() for speedup (PyTorch 2.0+)
        grad_accum_steps: Gradient accumulation steps (simulates larger batch sizes)

    Device Support:
        - CPU: Single process, no DDP, compile may be slow
        - Single GPU: Single process, mixed precision, fast compile
        - Multiple GPUs: Use DDP (torchrun), NCCL backend, per-GPU processes
    """

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        config=None,
        use_ddp=False,
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
        use_compile=False,
        grad_accum_steps=1,
    ):
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config

        # Normalize device: accept both string and torch.device object
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            logger.warning(f"Invalid device type {type(device)}, defaulting to CPU")
            self.device = torch.device('cpu')

        # DDP settings (for multi-GPU training)
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.is_master = (ddp_rank == 0)  # Only rank 0 logs and saves

        # Gradient accumulation (simulate larger batch sizes)
        self.grad_accum_steps = grad_accum_steps
        if grad_accum_steps > 1:
            logger.info(f"Using gradient accumulation: {grad_accum_steps} steps")

        # Mixed precision training scaler (CUDA only)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None

        # Store original model before wrapping (for checkpointing)
        self.raw_model = model

        # Setup multi-GPU training with DDP
        if use_ddp:
            if self.device.type != 'cuda':
                logger.warning("DDP only supports CUDA. Falling back to single-device mode.")
                self.use_ddp = False
                self.model = model.to(self.device)
            else:
                logger.info(f"Using DDP: rank={ddp_rank}/{ddp_world_size}, local_rank={ddp_local_rank}")
                model = model.to(self.device)
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(model, device_ids=[ddp_local_rank])
                logger.info(f"✓ DDP model wrapper initialized")
        else:
            # Single device (CPU or single GPU)
            self.model = model.to(self.device)

        # Optional: Compile model with PyTorch 2.0+ for speedup
        if use_compile:
            if hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile() for speedup...")
                self.model = torch.compile(self.model)
                logger.info("✓ Model compiled successfully")
            else:
                logger.warning("torch.compile() not available (requires PyTorch 2.0+). Skipping compilation.")

        # Validate tokenizer
        if tokenizer is not None:
            logger.info(f"Trainer initialized with pad_token_id={tokenizer.pad_token_id}")
            if tokenizer.pad_token_id is None:
                raise RuntimeError("Tokenizer pad_token_id is None")

        logger.info(f"Trainer initialized on device: {self.device}")

        # Training history for visualization
        self.history = {
            'loss': [],
            'perplexity': [],
            'load_balancing_loss': []
        }

        # TensorBoard writer (initialized in train())
        self.writer = None

        # WandB run (initialized in train() if enabled)
        self.wandb_run = None
        self.use_wandb = False

        # Benchmarks (lazily loaded)
        self._benchmark_evaluator = None

        # Training report path
        self.report_path = None

    def log_metrics(self, metrics: dict, step: int):
        """
        Log metrics to TensorBoard and WandB.

        In DDP mode, only the master process (rank 0) logs to avoid duplicates.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        # Only master process logs
        if not self.is_master:
            return

        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

        # Log to WandB
        if self.use_wandb and self.wandb_run is not None:
            wandb_metrics = {'step': step}
            wandb_metrics.update(metrics)
            wandb.log(wandb_metrics)

    def initialize_report(self, output_dir: str, model_name: str = "moellama"):
        """
        Initialize training report file.

        Args:
            output_dir: Output directory
            model_name: Model name for report filename
        """
        # Create report directory
        report_dir = Path(output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{model_name}_{timestamp}.md"
        self.report_path = report_dir / filename

        # Write initial report
        config_summary = self.config if self.config else {}
        model_config = config_summary.get('model', {})

        content = f"""# Training Report: {model_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Configuration

- **Vocabulary Size:** {model_config.get('vocab_size', 'N/A')}
- **Dimensions:** {model_config.get('dim', 'N/A')}
- **Layers:** {model_config.get('num_layers', 'N/A')}
- **Attention Heads:** {model_config.get('num_heads', 'N/A')}
- **Experts:** {model_config.get('num_experts', 'N/A')}
- **Top-K:** {model_config.get('top_k', 'N/A')}
- **Max Sequence Length:** {model_config.get('max_seq_len', 'N/A')}
- **Shared Expert:** {model_config.get('shared_expert', 'N/A')}

## Training Configuration

- **Batch Size:** {config_summary.get('training', {}).get('batch_size', 'N/A')}
- **Learning Rate:** {config_summary.get('training', {}).get('learning_rate', 'N/A')}
- **Epochs:** {config_summary.get('training', {}).get('epochs', 'N/A')}
- **Device:** {self.device}

## Training Progress

"""
        with open(self.report_path, 'w') as f:
            f.write(content)

        logger.info(f"Training report initialized: {self.report_path}")

    def update_report_epoch(self, epoch: int, metrics: dict, benchmark_results: dict = None):
        """
        Update report with epoch metrics.

        Args:
            epoch: Epoch number
            metrics: Dictionary with loss, perplexity, etc.
            benchmark_results: Optional benchmark results
        """
        if self.report_path is None:
            return

        content = f"\n### Epoch {epoch}\n\n"
        content += f"**Training Metrics:**\n"
        content += f"- Average Loss: {metrics.get('loss', 0):.4f}\n"
        content += f"- Average Load Balancing Loss: {metrics.get('lb_loss', 0):.4f}\n"

        if 'eval_loss' in metrics:
            content += f"- Eval Loss: {metrics['eval_loss']:.4f}\n"
            content += f"- Perplexity: {metrics['perplexity']:.2f}\n"

        if benchmark_results:
            content += f"\n**Benchmark Results:**\n"
            for bench_name, result in benchmark_results.get('benchmarks', {}).items():
                score = result['primary_metric']
                content += f"- {bench_name}: {score:.4f}\n"
            content += f"- **Average Score**: {benchmark_results.get('average_score', 0):.4f}\n"

        content += "\n"

        with open(self.report_path, 'a') as f:
            f.write(content)

    def update_report_final(self, benchmark_results: dict, training_time: float):
        """
        Update report with final comprehensive benchmarks.

        Args:
            benchmark_results: Final benchmark results
            training_time: Total training time in seconds
        """
        if self.report_path is None:
            return

        content = "\n## Final Evaluation\n\n"
        content += f"**Total Training Time:** {training_time/3600:.2f} hours\n\n"
        content += "### Comprehensive Benchmark Results\n\n"
        content += "| Benchmark | Score | Samples |\n"
        content += "|-----------|-------|----------|\n"

        for bench_name, result in benchmark_results.get('benchmarks', {}).items():
            score = result['primary_metric']
            samples = result['num_samples']
            content += f"| {bench_name} | {score:.4f} | {samples} |\n"

        content += f"\n**Overall Average Score:** {benchmark_results.get('average_score', 0):.4f}\n"

        content += "\n---\n*Report generated by MoeLLaMA Training System*\n"

        with open(self.report_path, 'a') as f:
            f.write(content)

        logger.info(f"Training report finalized: {self.report_path}")

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

    def estimate_training_time(self, train_loader, batch_size, epochs, is_streaming=False):
        """
        Estimate training time based on dataset size and system specs.

        Args:
            train_loader: Training data loader
            batch_size: Batch size
            epochs: Number of epochs
            is_streaming: Whether using streaming mode

        Returns:
            Dict with estimation info
        """
        # Try to get dataset size
        try:
            if hasattr(train_loader, '__len__'):
                # DataLoader with known length
                iterations_per_epoch = len(train_loader)
                dataset_size = iterations_per_epoch * batch_size
            elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, '__len__'):
                # Has underlying dataset
                dataset_size = len(train_loader.dataset)
                iterations_per_epoch = math.ceil(dataset_size / batch_size)
            else:
                # Streaming mode - can't determine exact size
                iterations_per_epoch = None
                dataset_size = None
        except:
            iterations_per_epoch = None
            dataset_size = None

        # Estimate time per iteration based on device and model size
        # These are rough estimates based on empirical testing
        model_params = sum(p.numel() for p in self.model.parameters())

        if self.device.type == 'cuda':
            # GPU: ~0.01-0.1s per iteration depending on model size
            base_time = 0.02 + (model_params / 50_000_000) * 0.05
        elif self.device.type == 'mps':
            # Apple Silicon: ~0.1-0.5s per iteration
            base_time = 0.1 + (model_params / 50_000_000) * 0.2
        else:
            # CPU: ~0.5-3s per iteration depending on model size and cores
            base_time = 1.0 + (model_params / 10_000_000) * 0.5

        # Log estimation
        logger.info("=" * 60)
        logger.info("Training Estimates:")
        logger.info("=" * 60)

        if iterations_per_epoch is not None:
            logger.info(f"  Iterations per epoch: ~{iterations_per_epoch:,}")
            logger.info(f"  Total iterations: ~{iterations_per_epoch * epochs:,}")

            # Estimate time
            estimated_time_per_epoch = iterations_per_epoch * base_time
            total_estimated_time = estimated_time_per_epoch * epochs

            logger.info(f"  Estimated time per iteration: ~{base_time:.2f}s")
            logger.info(f"  Estimated time per epoch: ~{estimated_time_per_epoch/3600:.2f}h ({estimated_time_per_epoch/60:.0f}m)")
            logger.info(f"  Total estimated training time: ~{total_estimated_time/3600:.2f}h ({total_estimated_time/60:.0f}m)")
        else:
            # Streaming mode - can't estimate accurately
            logger.info(f"  Mode: Streaming (dynamic iteration count)")
            logger.info(f"  Estimated time per iteration: ~{base_time:.2f}s")
            logger.info(f"  Note: Actual time depends on dataset size (unknown in streaming mode)")
            logger.info(f"  Tip: Monitor first epoch to estimate total time")

        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Device: {self.device.type}")
        logger.info(f"  Model parameters: {model_params/1_000_000:.1f}M")
        logger.info("=" * 60)

        return {
            'iterations_per_epoch': iterations_per_epoch,
            'dataset_size': dataset_size,
            'estimated_time_per_iteration': base_time,
            'is_streaming': is_streaming
        }

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=16,
        epochs=3,
        eval_steps=100,
        save_steps=None,
        output_dir='./model',
        num_workers=4,
        max_eval_batches=None,
        run_benchmarks=True,
        benchmark_samples=100,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
    ):
        """
        Train the model.

        Process:
        1. Create data loaders
        2. For each epoch:
            a. Train on all batches
            b. Periodically evaluate
            c. Run benchmarks at end of epoch
            d. Periodically save checkpoints
        3. Run comprehensive benchmarks at end
        4. Save final model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            batch_size: Number of sequences per batch
            epochs: Number of training epochs
            eval_steps: Evaluate every N steps during training (0 or None = disable periodic eval)
            save_steps: Save checkpoint every N steps (None = only at epoch end)
            output_dir: Directory to save checkpoints
            num_workers: Number of data loading workers
            max_eval_batches: Maximum number of batches to evaluate (None = auto)
            run_benchmarks: Run benchmarks at end of each epoch - separate from eval_steps (default: True)
            benchmark_samples: Number of samples per benchmark (default: 100)
            use_wandb: Use WandB for logging (default: False, TensorBoard always used)
            wandb_project: WandB project name
            wandb_run_name: WandB run name
        """
        logger.info("Starting training")
        logger.info(
            f"Config: batch_size={batch_size}, epochs={epochs}, "
            f"eval_steps={eval_steps}, save_steps={save_steps}"
        )

        # Initialize TensorBoard writer (always)
        if TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(output_dir, 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None

        # Initialize WandB if requested
        self.use_wandb = use_wandb
        if use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("WandB requested but not available. Install with: pip install wandb")
                self.use_wandb = False
            else:
                # Initialize WandB
                self.wandb_run = wandb.init(
                    project=wandb_project or "moellama-training",
                    name=wandb_run_name,
                    config={
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'eval_steps': eval_steps,
                        'model_dim': self.config.get('model', {}).get('dim'),
                        'num_layers': self.config.get('model', {}).get('num_layers'),
                        'num_experts': self.config.get('model', {}).get('num_experts'),
                    }
                )
                logger.info(f"WandB logging initialized: {wandb_project}/{wandb_run_name}")

        # Initialize training report
        model_name = self.config.get('model', {}).get('name', 'moellama')
        self.initialize_report(output_dir, model_name)

        # Setup data loader
        # Check if train_dataset is already a DataLoader (e.g., StreamingDataLoader)
        # DataLoaders are iterable but not subscriptable (no __getitem__)
        is_already_loader = (
            hasattr(train_dataset, '__iter__') and
            not hasattr(train_dataset, '__getitem__')
        )

        if is_already_loader:
            # It's already an iterable/loader, use it directly
            logger.info("Using pre-configured data loader (streaming mode)")
            train_loader = train_dataset
        else:
            # It's a Dataset, wrap it with DataLoader
            logger.info("Creating DataLoader from Dataset")
            pin_memory = (self.device.type == 'cuda')
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,  # Faster GPU transfer
                persistent_workers=True if num_workers > 0 else False,
                collate_fn=LLaMA4Trainer.collate_fn,
            )

        # Estimate training time based on dataset size and system specs
        self.estimate_training_time(
            train_loader=train_loader,
            batch_size=batch_size,
            epochs=epochs,
            is_streaming=is_already_loader
        )

        global_step = 0
        import time
        training_start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            epoch_load_balancing_losses = []

            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            # Gradient accumulation tracking
            micro_step = 0
            accumulated_loss = 0.0
            accumulated_lb_loss = 0.0

            for batch in progress_bar:
                # Handle both formats:
                # - StreamingDataLoader yields (inputs, targets) tuple
                # - Regular Dataset yields single tensor
                if isinstance(batch, tuple) and len(batch) == 2:
                    # StreamingDataLoader format: already on device
                    input_ids, targets = batch
                else:
                    # Regular Dataset format: move to device
                    input_ids = batch.to(self.device, non_blocking=True)
                    targets = input_ids  # Use same tensor for labels (shifted internally)

                # Forward pass (with mixed precision on CUDA)
                if self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(input_ids, labels=targets, training=True)
                        loss = outputs["loss"]
                        load_balancing_loss = outputs["load_balancing_loss"]

                        # Normalize loss for gradient accumulation
                        loss = loss / self.grad_accum_steps

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()

                    # Accumulate losses for logging (use original unnormalized values)
                    accumulated_loss += loss.item() * self.grad_accum_steps
                    accumulated_lb_loss += load_balancing_loss.item()
                else:
                    # CPU or MPS: full precision
                    outputs = self.model(input_ids, labels=targets, training=True)
                    loss = outputs["loss"]
                    load_balancing_loss = outputs["load_balancing_loss"]

                    # Normalize loss for gradient accumulation
                    loss = loss / self.grad_accum_steps

                    # Backward pass
                    loss.backward()

                    # Accumulate losses for logging (use original unnormalized values)
                    accumulated_loss += loss.item() * self.grad_accum_steps
                    accumulated_lb_loss += load_balancing_loss.item()

                micro_step += 1

                # Perform optimizer step after accumulating enough gradients
                if micro_step % self.grad_accum_steps == 0:
                    if self.device.type == 'cuda':
                        # Unscale before clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients for stability
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        # Optimizer step with scaled gradients
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        # Optimizer step
                        self.optimizer.step()

                    # Zero gradients for next accumulation
                    self.optimizer.zero_grad()

                    # Average accumulated losses
                    avg_loss = accumulated_loss / self.grad_accum_steps
                    avg_lb_loss = accumulated_lb_loss / self.grad_accum_steps

                    # Track metrics
                    epoch_losses.append(avg_loss)
                    epoch_load_balancing_losses.append(avg_lb_loss)

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lb_loss': f"{avg_lb_loss:.4f}"
                    })

                    # Increment global step (one step = one optimizer update)
                    global_step += 1

                    # Reset accumulators
                    accumulated_loss = 0.0
                    accumulated_lb_loss = 0.0

                    # Log training metrics (only on optimizer steps)
                    self.log_metrics({
                        'train/loss': avg_loss,
                        'train/load_balancing_loss': avg_lb_loss,
                    }, global_step)

                    # Periodic evaluation (during training, every N steps)
                    # Set eval_steps=0 or None to disable periodic evaluation
                    if eval_dataset and eval_steps and global_step % eval_steps == 0:
                        eval_loss, eval_ppl = self.evaluate(
                            eval_dataset, batch_size, num_workers, max_eval_batches
                        )
                        if self.is_master:
                            logger.info(
                                f"Step {global_step} | "
                                f"Eval Loss: {eval_loss:.4f} | "
                                f"Perplexity: {eval_ppl:.2f}"
                            )
                        # Log eval metrics
                        self.log_metrics({
                            'eval/loss': eval_loss,
                            'eval/perplexity': eval_ppl,
                        }, global_step)
                        self.model.train()  # Back to training mode

                    # Periodic checkpointing (only on optimizer steps)
                    if save_steps and global_step % save_steps == 0:
                        self.save_model(output_dir)

            # Save at end of epoch
            self.save_model(output_dir)

            # Log epoch metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            avg_lb_loss = sum(epoch_load_balancing_losses) / len(epoch_load_balancing_losses) if epoch_load_balancing_losses else 0.0

            if self.is_master:
                logger.info(
                    f"Epoch {epoch+1} completed | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg LB Loss: {avg_lb_loss:.4f}"
                )

            # Log epoch-level metrics
            epoch_metrics = {
                'epoch/avg_loss': avg_loss,
                'epoch/avg_lb_loss': avg_lb_loss,
            }
            self.log_metrics(epoch_metrics, global_step)

            # Run benchmarks at end of epoch
            benchmark_results = None
            if run_benchmarks:
                if self.is_master:
                    logger.info(f"\nRunning benchmarks for epoch {epoch+1}...")
                benchmark_results = self.run_benchmarks(
                    benchmarks=None,
                    max_samples=benchmark_samples,
                    step=global_step
                )

            # Update training report
            report_metrics = {
                'loss': avg_loss,
                'lb_loss': avg_lb_loss,
            }
            self.update_report_epoch(epoch+1, report_metrics, benchmark_results)

        # Training complete
        training_time = time.time() - training_start_time
        if self.is_master:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            logger.info(f"{'='*80}\n")

        # Run comprehensive final benchmarks
        if run_benchmarks:
            from moellama.benchmarks import get_comprehensive_benchmarks
            if self.is_master:
                logger.info("Running comprehensive final benchmarks (full evaluation)...")
            comprehensive_benchmarks = get_comprehensive_benchmarks(max_samples=None)  # All samples
            final_results = self.run_benchmarks(
                benchmarks=comprehensive_benchmarks,
                max_samples=None,  # Use all samples for final eval
                step=global_step
            )
            # Update report with final benchmarks
            self.update_report_final(final_results, training_time)

        # Close TensorBoard writer (master process only)
        if self.is_master and self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")

        # Finish WandB run (master process only)
        if self.is_master and self.use_wandb and self.wandb_run is not None:
            wandb.finish()
            logger.info("WandB run finished")

        if self.is_master:
            logger.info(f"Training report saved to: {self.report_path}")

    def evaluate(self, eval_dataset, batch_size=16, num_workers=4, max_eval_batches=None):
        """
        Evaluate the model on a dataset.

        Computes average loss and perplexity over the entire dataset or a subset.

        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            max_eval_batches: Maximum number of batches to evaluate (None = all)
                             Useful for streaming datasets to avoid infinite loops

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
        # Check if eval_dataset is already a DataLoader (e.g., StreamingDataLoader)
        is_already_loader = (
            hasattr(eval_dataset, '__iter__') and
            not hasattr(eval_dataset, '__getitem__')
        )

        if is_already_loader:
            # It's already an iterable/loader, use it directly
            logger.info("Using pre-configured eval data loader (streaming mode)")
            eval_loader = eval_dataset
            # For streaming loaders, set a default max_eval_batches if not specified
            if max_eval_batches is None:
                max_eval_batches = 100  # Reasonable default for streaming
                logger.info(f"Limiting evaluation to {max_eval_batches} batches (streaming mode)")
        else:
            # It's a Dataset, wrap it with DataLoader
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == 'cuda'),
                collate_fn=LLaMA4Trainer.collate_fn,
            )
            # For regular datasets, use all batches if not specified
            if max_eval_batches:
                logger.info(f"Limiting evaluation to {max_eval_batches} batches")

        total_loss = 0
        total_tokens = 0
        batch_count = 0

        # Evaluate without gradients
        with torch.no_grad():
            for batch in eval_loader:
                # Check batch limit
                if max_eval_batches and batch_count >= max_eval_batches:
                    logger.info(f"Reached max_eval_batches limit ({max_eval_batches}), stopping evaluation")
                    break

                batch_count += 1
                # Handle both formats:
                # - StreamingDataLoader yields (inputs, targets) tuple
                # - Regular Dataset yields single tensor
                if isinstance(batch, tuple) and len(batch) == 2:
                    # StreamingDataLoader format: already on device
                    input_ids, targets = batch
                else:
                    # Regular Dataset format: move to device
                    input_ids = batch.to(self.device)
                    targets = input_ids  # Use same tensor for labels (shifted internally)

                # Forward pass
                outputs = self.model(input_ids, labels=targets, training=False)
                loss = outputs["loss"]

                # Count non-padding tokens
                num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(
            f"Evaluation complete: {batch_count} batches, {total_tokens} tokens | "
            f"Avg Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}"
        )
        return avg_loss, perplexity

    def run_benchmarks(
        self,
        benchmarks: Optional[List] = None,
        max_samples: int = 100,
        step: Optional[int] = None,
    ) -> dict:
        """
        Run benchmarks on the model.

        Args:
            benchmarks: List of benchmark instances (None = use defaults)
            max_samples: Max samples per benchmark
            step: Current training step (for logging)

        Returns:
            Dictionary with benchmark results
        """
        from moellama.benchmarks import BenchmarkEvaluator, get_default_benchmarks

        logger.info(f"Running benchmarks (max_samples={max_samples})...")

        # Create evaluator if needed
        if self._benchmark_evaluator is None:
            self._benchmark_evaluator = BenchmarkEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=256,
                temperature=0.0,  # Greedy for benchmarks
            )

        # Use default benchmarks if none provided
        if benchmarks is None:
            benchmarks = get_default_benchmarks(max_samples=max_samples)

        # Run evaluation
        results = self._benchmark_evaluator.evaluate_all(benchmarks, verbose=False)

        # Log to TensorBoard if available
        if self.writer is not None and step is not None:
            for bench_name, bench_result in results['benchmarks'].items():
                metric_value = bench_result['primary_metric']
                self.writer.add_scalar(
                    f'benchmarks/{bench_name}',
                    metric_value,
                    step
                )
            self.writer.add_scalar(
                'benchmarks/average_score',
                results['average_score'],
                step
            )

        # Log summary
        logger.info("Benchmark Results:")
        for bench_name, bench_result in results['benchmarks'].items():
            metric_name = bench_result['metrics'].get('accuracy', 'metric')
            metric_value = bench_result['primary_metric']
            logger.info(f"  {bench_name}: {metric_value:.4f}")
        logger.info(f"  Average Score: {results['average_score']:.4f}")

        return results

    def save_model(self, output_dir, dataset=None):
        """
        Save model checkpoint and tokenizer.

        In DDP mode, only the master process (rank 0) saves to avoid conflicts.

        Saves:
        - Model state dict with timestamp
        - Tokenizer vocabulary

        Args:
            output_dir: Directory to save to
            dataset: Dataset name (for filename, optional)
        """
        # Only master process saves in DDP mode
        if not self.is_master:
            return

        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Get dataset name from config if not provided
        dataset = dataset or self.config.get("training", {}).get("dataset", "unknown")

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{dataset}_{timestamp}.pt"

        # Use raw model (unwrapped, original model)
        model_to_save = self.raw_model

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
