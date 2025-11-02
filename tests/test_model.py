"""
Tests for LLaMA4MoE Model Architecture.

Tests cover:
- Model initialization
- Forward pass
- Generation
- Model components (attention, MoE, etc.)
- Save/load functionality
"""

import pytest
import torch
import tempfile
from pathlib import Path

from moellama import LLaMA4MoE


class TestModelInitialization:
    """Tests for model initialization."""

    def test_model_creation(self, tiny_config):
        """Test creating a model instance."""
        model = LLaMA4MoE(
            vocab_size=100,
            dim=tiny_config['model']['dim'],
            num_layers=tiny_config['model']['num_layers'],
            num_heads=tiny_config['model']['num_heads'],
            num_experts=tiny_config['model']['num_experts'],
            top_k=tiny_config['model']['top_k'],
            max_seq_len=tiny_config['model']['max_seq_len'],
            dropout=tiny_config['model']['dropout'],
            shared_expert=tiny_config['model']['shared_expert'],
            load_balancing_loss_coef=tiny_config['model']['load_balancing_loss_coef']
        )

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_has_correct_parameters(self, tiny_model):
        """Test that model has trainable parameters."""
        params = list(tiny_model.parameters())
        assert len(params) > 0

        # All parameters should be tensors
        for param in params:
            assert isinstance(param, torch.Tensor)

    def test_model_output_shape(self, tiny_model, sample_batch, device):
        """Test that model output has correct shape."""
        tiny_model.to(device)
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            output = tiny_model(sample_batch)

        # Output should be a dict with 'logits' key
        assert isinstance(output, dict)
        assert 'logits' in output

        # Logits should have shape (batch_size, seq_len, vocab_size)
        batch_size, seq_len = sample_batch.shape
        vocab_size = tiny_model.token_embeddings.weight.shape[0]

        assert output['logits'].shape == (batch_size, seq_len, vocab_size)


class TestForwardPass:
    """Tests for model forward pass."""

    def test_forward_pass_runs(self, tiny_model, sample_batch, device):
        """Test that forward pass completes without error."""
        tiny_model.to(device)
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            output = tiny_model(sample_batch)

        assert output is not None

    def test_forward_pass_with_different_batch_sizes(self, tiny_model, sample_tokenizer, device):
        """Test forward pass with different batch sizes."""
        tiny_model.to(device)
        seq_len = 16

        for batch_size in [1, 2, 4]:
            # Create dummy batch
            batch = torch.randint(0, len(sample_tokenizer), (batch_size, seq_len)).to(device)

            with torch.no_grad():
                output = tiny_model(batch)

            assert output['logits'].shape[0] == batch_size

    def test_forward_pass_different_sequence_lengths(self, tiny_model, sample_tokenizer, device):
        """Test forward pass with different sequence lengths."""
        tiny_model.to(device)
        batch_size = 2

        for seq_len in [8, 16, 32]:
            if seq_len > tiny_model.max_seq_len:
                continue

            batch = torch.randint(0, len(sample_tokenizer), (batch_size, seq_len)).to(device)

            with torch.no_grad():
                output = tiny_model(batch)

            assert output['logits'].shape[1] == seq_len

    def test_output_is_valid_logits(self, tiny_model, sample_batch, device):
        """Test that output logits are valid (not NaN or Inf)."""
        tiny_model.to(device)
        sample_batch = sample_batch.to(device)

        with torch.no_grad():
            output = tiny_model(sample_batch)

        logits = output['logits']
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestGeneration:
    """Tests for text generation."""

    def test_generate_runs(self, tiny_model, sample_tokenizer, device):
        """Test that generation completes without error."""
        tiny_model.to(device)
        tiny_model.eval()

        # Create a simple prompt
        prompt = "To be"
        token_ids = sample_tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = tiny_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.8
            )

        assert generated is not None
        assert generated.shape[0] == 1  # batch size
        assert generated.shape[1] > input_ids.shape[1]  # Should be longer

    def test_generate_respects_max_tokens(self, tiny_model, sample_tokenizer, device):
        """Test that generation respects max_new_tokens limit."""
        tiny_model.to(device)
        tiny_model.eval()

        prompt = "Hello"
        token_ids = sample_tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        input_len = input_ids.shape[1]

        max_new = 5
        with torch.no_grad():
            generated = tiny_model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=0.8
            )

        # Generated length should be input + max_new (or less if EOS)
        assert generated.shape[1] <= input_len + max_new

    def test_generate_with_temperature(self, tiny_model, sample_tokenizer, device):
        """Test generation with different temperatures."""
        tiny_model.to(device)
        tiny_model.eval()

        prompt = "Test"
        token_ids = sample_tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        temperatures = [0.1, 0.8, 1.5]
        for temp in temperatures:
            with torch.no_grad():
                generated = tiny_model.generate(
                    input_ids,
                    max_new_tokens=5,
                    temperature=temp
                )
            assert generated.shape[1] > input_ids.shape[1]

    def test_generate_with_sampling_params(self, tiny_model, sample_tokenizer, device):
        """Test generation with top-k and top-p sampling."""
        tiny_model.to(device)
        tiny_model.eval()

        prompt = "Test"
        token_ids = sample_tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = tiny_model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.8,
                top_k=10,
                top_p=0.9
            )

        assert generated.shape[1] > input_ids.shape[1]


class TestModelPersistence:
    """Tests for saving and loading models."""

    def test_save_and_load_state_dict(self, tiny_model, sample_batch, tiny_config, device):
        """Test saving and loading model state dict."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            # Get output before saving
            tiny_model.to(device)
            tiny_model.eval()
            sample_batch = sample_batch.to(device)

            with torch.no_grad():
                output_before = tiny_model(sample_batch)

            # Save
            torch.save(tiny_model.state_dict(), temp_path)

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
            new_model.load_state_dict(torch.load(temp_path, map_location=device))
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

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestModelComponents:
    """Tests for individual model components."""

    def test_model_has_embeddings(self, tiny_model):
        """Test that model has token embeddings."""
        assert hasattr(tiny_model, 'token_embeddings')
        assert isinstance(tiny_model.token_embeddings, torch.nn.Embedding)

    def test_model_has_layers(self, tiny_model):
        """Test that model has transformer layers."""
        assert hasattr(tiny_model, 'layers')
        assert len(tiny_model.layers) == tiny_model.num_layers

    def test_model_has_output_projection(self, tiny_model):
        """Test that model has output projection."""
        assert hasattr(tiny_model, 'lm_head')
        assert isinstance(tiny_model.lm_head, torch.nn.Linear)

    def test_moe_layers_exist(self, tiny_model):
        """Test that MoE layers are present in transformer blocks."""
        for layer in tiny_model.layers:
            # Each transformer block should have an MoE layer
            assert hasattr(layer, 'moe')

    def test_gradient_flow(self, tiny_model, sample_batch, device):
        """Test that gradients flow through the model."""
        tiny_model.to(device)
        tiny_model.train()
        sample_batch = sample_batch.to(device)

        # Forward pass
        output = tiny_model(sample_batch)

        # Create dummy loss from logits
        loss = output['logits'].mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in tiny_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in model parameters"
