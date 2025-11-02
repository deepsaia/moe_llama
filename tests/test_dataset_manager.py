"""
Tests for DatasetManager module.

These tests verify the multi-dataset loading functionality.
"""

import pytest
from pathlib import Path

from moellama.dataset_manager import DatasetManager, DatasetConfig


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid dataset config."""
        config = DatasetConfig(
            name="test_dataset",
            ratio=0.6,
            split="train",
            streaming=False,
            percentage=0.8,
        )

        assert config.name == "test_dataset"
        assert config.ratio == 0.6
        assert config.percentage == 0.8

    def test_invalid_ratio(self):
        """Test that invalid ratio raises error."""
        with pytest.raises(ValueError, match="ratio must be in"):
            DatasetConfig(name="test", ratio=1.5)

    def test_invalid_percentage(self):
        """Test that invalid percentage raises error."""
        with pytest.raises(ValueError, match="percentage must be in"):
            DatasetConfig(name="test", percentage=1.5)

    def test_local_dataset_requires_path(self):
        """Test that local dataset requires path."""
        with pytest.raises(ValueError, match="requires 'path'"):
            DatasetConfig(name="local")


class TestDatasetManager:
    """Tests for DatasetManager."""

    def test_parse_single_dataset(self):
        """Test parsing single dataset config (backward compatible)."""
        config = {
            'dataset': 'tiny_shakespeare',
            'batch_size': 16,
            'seq_len': 256,
        }

        manager = DatasetManager(config)
        assert len(manager.dataset_configs) == 1
        assert manager.dataset_configs[0].name == 'tiny_shakespeare'
        assert manager.dataset_configs[0].ratio == 1.0

    def test_parse_datasets_list(self):
        """Test parsing simple list of datasets."""
        config = {
            'datasets': ['dataset1', 'dataset2', 'dataset3'],
            'batch_size': 16,
            'seq_len': 256,
        }

        manager = DatasetManager(config)
        assert len(manager.dataset_configs) == 3

        # Check equal ratios
        for ds_config in manager.dataset_configs:
            assert abs(ds_config.ratio - 1/3) < 1e-6

    def test_parse_dataset_mixture(self):
        """Test parsing full dataset mixture config."""
        config = {
            'dataset_mixture': [
                {
                    'name': 'dataset1',
                    'ratio': 0.6,
                    'streaming': True,
                    'percentage': 0.5,
                },
                {
                    'name': 'dataset2',
                    'ratio': 0.4,
                    'streaming': False,
                },
            ],
            'batch_size': 16,
            'seq_len': 256,
        }

        manager = DatasetManager(config)
        assert len(manager.dataset_configs) == 2
        assert manager.dataset_configs[0].ratio == 0.6
        assert manager.dataset_configs[0].streaming == True
        assert manager.dataset_configs[1].ratio == 0.4

    def test_no_dataset_config_raises_error(self):
        """Test that missing dataset config raises error."""
        config = {
            'batch_size': 16,
            'seq_len': 256,
        }

        with pytest.raises(ValueError, match="must contain"):
            DatasetManager(config)

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        config = {
            'dataset_mixture': [
                {'name': 'dataset1', 'ratio': 0.7},
                {'name': 'dataset2', 'ratio': 0.3},
            ],
        }

        manager = DatasetManager(config)
        info = manager.get_dataset_info()

        assert info['num_datasets'] == 2
        assert len(info['datasets']) == 2
        assert info['datasets'][0]['name'] == 'dataset1'
        assert info['datasets'][0]['ratio'] == 0.7


class TestDatasetLoading:
    """Integration tests for actual dataset loading."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_tiny_shakespeare(self):
        """Test loading tiny_shakespeare dataset."""
        config = {
            'dataset': 'tiny_shakespeare',
            'data_dir': 'dataset',
            'batch_size': 16,
            'seq_len': 256,
        }

        manager = DatasetManager(config)
        # This would actually load the dataset
        # dataset = manager.load_datasets()
        # assert dataset is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_multiple_datasets(self):
        """Test loading and mixing multiple datasets."""
        # This test would require actual datasets to be available
        # Skipping for now
        pytest.skip("Requires actual datasets")


# Fixtures for testing

@pytest.fixture
def sample_config():
    """Fixture providing a sample config."""
    return {
        'dataset_mixture': [
            {
                'name': 'tiny_shakespeare',
                'ratio': 0.6,
            },
            {
                'name': 'dataset2',
                'ratio': 0.4,
                'percentage': 0.5,
            },
        ],
        'batch_size': 16,
        'seq_len': 256,
        'data_dir': 'dataset',
    }


def test_with_sample_config(sample_config):
    """Test using the sample config fixture."""
    manager = DatasetManager(sample_config)
    assert len(manager.dataset_configs) == 2
