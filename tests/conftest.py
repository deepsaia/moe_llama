"""
Shared pytest fixtures for moellama tests.

This module provides common fixtures used across multiple test files,
including tiny sample datasets for fast testing.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from moellama import BPETokenizer, LLaMA4MoE


# Sample text data for testing
TINY_TEXT_SAMPLE = """
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer.
The slings and arrows of outrageous fortune.
Or to take arms against a sea of troubles.
"""

TINY_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
"""


@pytest.fixture
def tiny_text():
    """Fixture providing tiny text sample for testing."""
    return TINY_TEXT_SAMPLE


@pytest.fixture
def tiny_shakespeare():
    """Fixture providing tiny Shakespeare sample for testing."""
    return TINY_SHAKESPEARE


@pytest.fixture
def temp_text_file(tiny_text):
    """Fixture creating a temporary text file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(tiny_text)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Fixture creating a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tokenizer(tiny_text):
    """Fixture providing a trained BPE tokenizer on tiny sample."""
    tokenizer = BPETokenizer(vocab_size=100)  # Very small vocab for testing
    tokenizer.train(tiny_text)  # Pass text content, not file path
    return tokenizer


@pytest.fixture
def tiny_config():
    """Fixture providing a minimal model configuration for testing."""
    return {
        'model': {
            'dim': 32,  # Very small for fast testing
            'num_layers': 2,
            'num_heads': 2,
            'num_experts': 4,
            'top_k': 2,
            'max_seq_len': 64,
            'dropout': 0.1,
            'shared_expert': True,
            'load_balancing_loss_coef': 0.01,
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 3e-4,
            'epochs': 1,
            'eval_steps': 10,
            'seq_len': 32,
            'num_workers': 0,
        },
        'device': {
            'type': 'cpu',  # Always use CPU for testing
            'num_cpu_threads': 1,
        },
        'paths': {
            'model_path': './test_models',
            'output_dir': './test_output',
        },
    }


@pytest.fixture
def tiny_model(sample_tokenizer, tiny_config):
    """Fixture providing a tiny model instance for testing."""
    vocab_size = len(sample_tokenizer)
    model = LLaMA4MoE(
        vocab_size=vocab_size,
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
    model.eval()
    return model


@pytest.fixture
def device():
    """Fixture providing the test device (always CPU)."""
    return torch.device('cpu')


@pytest.fixture
def sample_batch(sample_tokenizer):
    """Fixture providing a sample batch of tokenized data."""
    text = "To be or not to be. That is the question."
    token_ids = sample_tokenizer.encode(text)

    # Pad or truncate to fixed length
    seq_len = 16
    if len(token_ids) < seq_len:
        token_ids = token_ids + [0] * (seq_len - len(token_ids))
    else:
        token_ids = token_ids[:seq_len]

    # Create batch (batch_size=2)
    batch = torch.tensor([token_ids, token_ids], dtype=torch.long)
    return batch


# Pytest markers for organizing tests

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
