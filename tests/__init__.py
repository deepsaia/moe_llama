"""
Test suite for moellama.

This package contains comprehensive tests for the moellama project:
- test_tokenizer.py: BPE tokenization tests
- test_model.py: Model architecture tests
- test_dataset_manager.py: Multi-dataset loading tests
- test_trainer.py: Training loop tests
- test_benchmarks.py: Benchmark evaluation tests

Run all tests:
    pytest

Run specific test file:
    pytest tests/test_tokenizer.py

Run with coverage:
    pytest --cov=moellama --cov-report=html

Run fast tests only (skip slow tests):
    pytest -m "not slow"
"""
