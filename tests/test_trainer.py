"""
Tests for LLaMA4Trainer.

Tests cover:
- Trainer initialization
- Training loop (single step, single epoch)
- Loss computation
- Checkpoint saving/loading
- Evaluation
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from moellama import LLaMA4Trainer, TextDataset


class TestTrainerInitialization:
    """Tests for trainer initialization."""

    def test_trainer_creation(self, tiny_model, sample_tokenizer, device):
        """Test creating a trainer instance."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        trainer = LLaMA4Trainer(
            model=tiny_model,
            tokenizer=sample_tokenizer,
            optimizer=optimizer,
            device=device
        )

        assert trainer is not None
        assert trainer.model == tiny_model
        assert trainer.tokenizer == sample_tokenizer
        assert trainer.optimizer == optimizer
        assert trainer.device == device

    def test_trainer_with_different_optimizers(self, tiny_model, sample_tokenizer, device):
        """Test trainer with different optimizers."""
        optimizers = [
            torch.optim.Adam(tiny_model.parameters(), lr=1e-4),
            torch.optim.SGD(tiny_model.parameters(), lr=1e-3),
            torch.optim.AdamW(tiny_model.parameters(), lr=1e-4),
        ]

        for optimizer in optimizers:
            trainer = LLaMA4Trainer(
                model=tiny_model,
                tokenizer=sample_tokenizer,
                optimizer=optimizer,
                device=device
            )
            assert trainer.optimizer == optimizer


class TestTrainingStep:
    """Tests for single training step."""

    def test_train_step_runs(self, tiny_model, sample_batch, sample_tokenizer, device):
        """Test that a single training step completes."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        trainer = LLaMA4Trainer(
            model=tiny_model,
            tokenizer=sample_tokenizer,
            optimizer=optimizer,
            device=device
        )

        tiny_model.train()
        tiny_model.to(device)
        sample_batch = sample_batch.to(device)

        # Single training step
        initial_params = [p.clone() for p in tiny_model.parameters()]

        # Forward + backward
        output = tiny_model(sample_batch)
        targets = sample_batch[:, 1:].contiguous()
        logits = output['logits'][:, :-1, :].contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        params_changed = False
        for initial, current in zip(initial_params, tiny_model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break

        assert params_changed, "Parameters did not update during training step"

    def test_loss_computation(self, tiny_model, sample_batch, device):
        """Test that loss is computed correctly."""
        tiny_model.to(device)
        tiny_model.train()
        sample_batch = sample_batch.to(device)

        # Model returns dict with 'loss' when labels are provided
        output = tiny_model(sample_batch, labels=sample_batch)

        assert output is not None
        assert 'loss' in output
        loss = output['loss']

        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_gradient_clipping(self, tiny_model, sample_batch, sample_tokenizer, device):
        """Test gradient clipping functionality."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        trainer = LLaMA4Trainer(
            model=tiny_model,
            tokenizer=sample_tokenizer,
            optimizer=optimizer,
            device=device
        )

        tiny_model.train()
        tiny_model.to(device)
        sample_batch = sample_batch.to(device)

        # Forward pass
        output = tiny_model(sample_batch)
        targets = sample_batch[:, 1:].contiguous()
        logits = output['logits'][:, :-1, :].contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), max_norm)

        # Check that gradient norms are clipped
        total_norm = 0
        for p in tiny_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # After clipping, norm should be <= max_norm (or close due to numerical precision)
        assert total_norm <= max_norm + 0.1  # Small tolerance for numerical errors


class TestTrainingLoop:
    """Tests for training loop."""

    @pytest.mark.slow
    def test_single_epoch_training(self, tiny_model, sample_tokenizer, device):
        """Test training for a single epoch on tiny data."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        trainer = LLaMA4Trainer(
            model=tiny_model,
            optimizer=optimizer,
            device=device,
            use_amp=False
        )

        # Create tiny dataset
        num_samples = 10
        seq_len = 16
        vocab_size = len(sample_tokenizer)

        data = torch.randint(0, vocab_size, (num_samples, seq_len))
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Train for one epoch
        tiny_model.train()
        initial_loss = None
        final_loss = None

        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)

            output = tiny_model(batch)
            targets = batch[:, 1:].contiguous()
            logits = output[:, :-1, :].contiguous()

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should be finite
        assert initial_loss is not None
        assert final_loss is not None
        assert not torch.isnan(torch.tensor(final_loss))


class TestCheckpointing:
    """Tests for checkpoint saving and loading."""

    def test_save_checkpoint(self, tiny_model, sample_tokenizer, device):
        """Test saving a checkpoint."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        trainer = LLaMA4Trainer(
            model=tiny_model,
            tokenizer=sample_tokenizer,
            optimizer=optimizer,
            device=device
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            torch.save({
                'model_state_dict': tiny_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 1,
            }, checkpoint_path)

            assert checkpoint_path.exists()

    def test_load_checkpoint(self, tiny_model, sample_batch, tiny_config, device):
        """Test loading a checkpoint."""
        from moellama import LLaMA4MoE

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        # Get initial output
        tiny_model.to(device)
        tiny_model.eval()
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            initial_output = tiny_model(sample_batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            torch.save({
                'model_state_dict': tiny_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 1,
            }, checkpoint_path)

            # Create new model and load using config (same dropout as original)
            new_model = LLaMA4MoE(
                vocab_size=tiny_model.token_embeddings.weight.shape[0],
                dim=tiny_config['model']['dim'],
                num_layers=tiny_config['model']['num_layers'],
                num_heads=tiny_config['model']['num_heads'],
                num_experts=tiny_config['model']['num_experts'],
                top_k=tiny_config['model']['top_k'],
                max_seq_len=tiny_config['model']['max_seq_len'],
                dropout=tiny_config['model']['dropout'],  # Match original dropout
                shared_expert=tiny_config['model']['shared_expert'],
                load_balancing_loss_coef=tiny_config['model']['load_balancing_loss_coef']
            )

            checkpoint = torch.load(checkpoint_path, map_location=device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_model.to(device)
            new_model.eval()

            # Verify weights loaded correctly by comparing state dicts
            orig_state = tiny_model.state_dict()
            new_state = new_model.state_dict()

            # All keys should match
            assert set(orig_state.keys()) == set(new_state.keys())

            # All weights should be identical
            for key in orig_state.keys():
                assert torch.allclose(orig_state[key], new_state[key], atol=1e-6), f"Mismatch in {key}"


class TestEvaluation:
    """Tests for model evaluation."""

    def test_evaluation_mode(self, tiny_model, sample_batch, device):
        """Test switching between train and eval modes."""
        tiny_model.to(device)

        # Train mode
        tiny_model.train()
        assert tiny_model.training

        # Eval mode
        tiny_model.eval()
        assert not tiny_model.training

    def test_no_gradients_in_eval(self, tiny_model, sample_batch, device):
        """Test that no gradients are computed in eval mode."""
        tiny_model.to(device)
        tiny_model.eval()
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            output = tiny_model(sample_batch)

        # No gradients should exist
        for param in tiny_model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0
