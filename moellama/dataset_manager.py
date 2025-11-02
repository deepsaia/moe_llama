"""
Dataset Manager for Multi-Dataset Training.

This module provides utilities for loading, mixing, and streaming multiple datasets
with configurable ratios, percentage sampling, and domain filtering.

Key Features:
- Load multiple datasets with custom ratios
- Streaming support for large datasets (100B+ tokens)
- Percentage sampling (e.g., use only 10% of a dataset)
- Domain filtering for datasets like FineFineWeb
- HuggingFace datasets integration
- Backward compatible with single-dataset configs
"""

from loguru import logger
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from datasets import load_dataset, interleave_datasets, Dataset, IterableDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None
    IterableDataset = None



@dataclass
class DatasetConfig:
    """
    Configuration for a single dataset in a mixture.

    Args:
        name: Dataset name (HuggingFace dataset or 'local' for custom)
        ratio: Sampling ratio in the mixture (default: 1.0)
        split: Dataset split to use (default: 'train')
        streaming: Whether to use streaming mode (default: False)
        percentage: Percentage of dataset to use (default: 1.0 = 100%)
        subset: Specific subset/config name (optional)
        domains: List of domains to filter (for domain-based datasets)
        path: Local file path (for local datasets)
        format: Data format ('text', 'conversation', 'jsonl')
        text_field: Field name containing text (default: 'text')
    """
    name: str
    ratio: float = 1.0
    split: str = "train"
    streaming: bool = False
    percentage: float = 1.0
    subset: Optional[str] = None
    domains: Optional[List[str]] = None
    path: Optional[str] = None
    format: str = "text"
    text_field: str = "text"

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if not 0 < self.percentage <= 1.0:
            raise ValueError(f"percentage must be in (0, 1], got {self.percentage}")
        if self.name == "local" and not self.path:
            raise ValueError("'local' dataset requires 'path' parameter")


class DatasetManager:
    """
    Manages loading and mixing multiple datasets.

    Supports:
    - Single or multiple datasets
    - Streaming and map-style datasets
    - Percentage sampling
    - Domain filtering
    - Ratio-based mixing

    Example:
        >>> config = {
        ...     'dataset_mixture': [
        ...         {'name': 'tiny_shakespeare', 'ratio': 0.6},
        ...         {'name': 'Salesforce/wikitext', 'ratio': 0.4, 'subset': 'wikitext-2-v1'}
        ...     ]
        ... }
        >>> manager = DatasetManager(config)
        >>> dataset = manager.load_datasets()
    """

    def __init__(self, config: Dict):
        """
        Initialize DatasetManager.

        Args:
            config: Training configuration dictionary
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library not found. "
                "Install with: pip install datasets"
            )

        self.config = config
        self.dataset_configs = self._parse_config()
        logger.info(f"DatasetManager initialized with {len(self.dataset_configs)} datasets")

    def _parse_config(self) -> List[DatasetConfig]:
        """
        Parse configuration into DatasetConfig objects.

        Supports multiple config formats:
        1. dataset_mixture: List of dataset configs
        2. datasets: Simple list of dataset names (equal ratios)
        3. dataset: Single dataset name (backward compatible)

        Returns:
            List of DatasetConfig objects
        """
        # Format 1: dataset_mixture (most flexible)
        if 'dataset_mixture' in self.config:
            configs = []
            for ds_dict in self.config['dataset_mixture']:
                configs.append(DatasetConfig(**ds_dict))

            # Check for mixed streaming modes (now supported!)
            if len(configs) > 1:
                streaming_modes = [ds.streaming for ds in configs]
                if len(set(streaming_modes)) > 1:
                    # Mixed streaming modes detected - this is OK
                    logger.info(
                        "Mixed streaming modes detected: some datasets streaming, others not. "
                        "This is supported - small datasets can load fully while large ones stream."
                    )

            return configs

        # Format 2: datasets (simple list, equal ratios)
        elif 'datasets' in self.config:
            dataset_names = self.config['datasets']
            if not isinstance(dataset_names, list):
                raise ValueError("'datasets' must be a list of dataset names")

            ratio = 1.0 / len(dataset_names)
            # Default to streaming=true for multi-dataset
            streaming = self.config.get('streaming', True)
            configs = []
            for name in dataset_names:
                configs.append(DatasetConfig(name=name, ratio=ratio, streaming=streaming))
            return configs

        # Format 3: dataset (single dataset, backward compatible)
        elif 'dataset' in self.config:
            name = self.config['dataset']
            # Default to streaming=false for backward compatibility
            streaming = self.config.get('streaming', False)
            return [DatasetConfig(name=name, ratio=1.0, streaming=streaming)]

        else:
            raise ValueError(
                "Config must contain 'dataset_mixture', 'datasets', or 'dataset'"
            )

    def load_datasets(self) -> Union[Dataset, IterableDataset]:
        """
        Load and mix all configured datasets.

        Process:
        1. Load each dataset (streaming or map-style)
        2. Apply percentage sampling
        3. Apply domain filtering
        4. Interleave with configured ratios

        Returns:
            Combined dataset (Dataset or IterableDataset)

        Raises:
            ValueError: If no datasets configured or invalid configuration
            ImportError: If datasets library not available
            Exception: If dataset loading fails
        """
        if not self.dataset_configs:
            raise ValueError("No datasets configured. Please add datasets to configuration.")

        logger.info(f"Loading {len(self.dataset_configs)} dataset(s)...")

        # Load individual datasets
        datasets = []
        probabilities = []
        failed_datasets = []

        for ds_config in self.dataset_configs:
            logger.info(
                f"Loading: {ds_config.name} "
                f"(ratio={ds_config.ratio:.2f}, "
                f"percentage={ds_config.percentage:.2f}, "
                f"streaming={ds_config.streaming})"
            )

            try:
                # Load dataset
                dataset = self._load_single_dataset(ds_config)

                # Apply percentage sampling
                if ds_config.percentage < 1.0:
                    dataset = self._apply_percentage_sampling(dataset, ds_config)

                # Apply domain filtering
                if ds_config.domains:
                    dataset = self._apply_domain_filter(dataset, ds_config.domains)

                datasets.append(dataset)
                probabilities.append(ds_config.ratio)
                logger.info(f"✓ Successfully loaded: {ds_config.name}")

            except Exception as e:
                error_msg = f"Failed to load dataset '{ds_config.name}': {str(e)}"
                logger.error(error_msg)
                failed_datasets.append((ds_config.name, str(e)))
                # Continue trying other datasets instead of failing immediately

        # Check if any datasets loaded successfully
        if not datasets:
            error_details = "\n".join([f"  - {name}: {error}" for name, error in failed_datasets])
            raise RuntimeError(
                f"Failed to load any datasets. Errors:\n{error_details}\n\n"
                "Common issues:\n"
                "  1. Dataset name is incorrect\n"
                "  2. Dataset not available on HuggingFace\n"
                "  3. Network connection issues\n"
                "  4. Missing dependencies (install with: pip install datasets)"
            )

        # Warn about failed datasets
        if failed_datasets:
            logger.warning(f"Loaded {len(datasets)}/{len(self.dataset_configs)} datasets successfully")
            for name, error in failed_datasets:
                logger.warning(f"  Skipped: {name} ({error})")

        # Normalize probabilities to sum to 1.0
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # If single dataset, return as-is
        if len(datasets) == 1:
            logger.info("Single dataset loaded")
            return datasets[0]

        # Mix multiple datasets
        logger.info(f"Interleaving {len(datasets)} datasets with probabilities: {probabilities}")
        stopping_strategy = self.config.get('stopping_strategy', 'all_exhausted')

        # Check if we have mixed Dataset/IterableDataset types
        # HuggingFace interleave_datasets requires all same type
        has_streaming = any(isinstance(ds, IterableDataset) for ds in datasets)
        has_non_streaming = any(isinstance(ds, Dataset) and not isinstance(ds, IterableDataset) for ds in datasets)

        if has_streaming and has_non_streaming:
            # Mixed types detected - convert all to IterableDataset
            logger.info("Converting non-streaming datasets to iterable for interleaving")
            converted_datasets = []
            kept_probabilities = []
            for i, ds in enumerate(datasets):
                if isinstance(ds, IterableDataset):
                    converted_datasets.append(ds)
                    kept_probabilities.append(probabilities[i])
                else:
                    # Check if dataset is empty (can happen with percentage sampling)
                    if len(ds) == 0:
                        logger.warning(
                            f"Dataset at position {i} is empty after percentage sampling. "
                            "Skipping this dataset."
                        )
                        continue
                    # Convert Dataset to IterableDataset
                    converted_datasets.append(ds.to_iterable_dataset())
                    kept_probabilities.append(probabilities[i])

            datasets = converted_datasets
            probabilities = kept_probabilities

            # Renormalize probabilities after removing empty datasets
            if probabilities:
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]

        mixed_dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy
        )

        logger.info(f"Dataset mixing complete (strategy: {stopping_strategy})")
        return mixed_dataset

    def _load_single_dataset(self, ds_config: DatasetConfig) -> Union[Dataset, IterableDataset]:
        """
        Load a single dataset with all configurations.

        Args:
            ds_config: Dataset configuration

        Returns:
            Loaded dataset (Dataset or IterableDataset)
        """
        # Handle local datasets
        if ds_config.name == "local":
            return self._load_local_dataset(ds_config)

        # Handle tiny_shakespeare specially (no HF hosting)
        if ds_config.name == "tiny_shakespeare":
            return self._load_tiny_shakespeare(ds_config)

        # Load from HuggingFace
        try:
            load_kwargs = {
                'path': ds_config.name,
                'split': ds_config.split,
                'streaming': ds_config.streaming,
            }

            # Add subset if specified
            if ds_config.subset:
                load_kwargs['name'] = ds_config.subset

            dataset = load_dataset(**load_kwargs)
            logger.info(f"Loaded {ds_config.name} from HuggingFace")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {ds_config.name}: {str(e)}")
            raise

    def _load_local_dataset(self, ds_config: DatasetConfig) -> Union[Dataset, IterableDataset]:
        """
        Load dataset from local file.

        Supports:
        - .txt files (plain text)
        - .jsonl files (JSON lines)

        Args:
            ds_config: Dataset configuration with 'path'

        Returns:
            Loaded dataset (Dataset or IterableDataset)
        """
        path = Path(ds_config.path)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset not found: {path}")

        logger.info(f"Loading local dataset from {path}")

        # Plain text file
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Create simple dataset with text field
            from datasets import Dataset as HFDataset
            dataset = HFDataset.from_dict({ds_config.text_field: [text]})

            # Convert to IterableDataset if streaming requested
            if ds_config.streaming:
                dataset = dataset.to_iterable_dataset()

            return dataset

        # JSONL file
        elif path.suffix == '.jsonl':
            dataset = load_dataset(
                'json',
                data_files=str(path),
                split='train',
                streaming=ds_config.streaming  # Pass streaming mode to load_dataset
            )
            return dataset

        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: .txt, .jsonl"
            )

    def _load_tiny_shakespeare(self, ds_config: DatasetConfig) -> Union[Dataset, IterableDataset]:
        """
        Load Tiny Shakespeare dataset (special case).

        Uses the existing download_tiny_shakespeare function.

        Args:
            ds_config: Dataset configuration

        Returns:
            Dataset or IterableDataset with text field
        """
        from moellama.dataset import download_tiny_shakespeare
        from datasets import Dataset as HFDataset

        data_dir = self.config.get('data_dir', 'dataset')
        text = download_tiny_shakespeare(data_dir)

        # Split into train/eval if needed
        if ds_config.split == 'train':
            train_size = int(0.9 * len(text))
            text = text[:train_size]
        elif ds_config.split == 'eval' or ds_config.split == 'validation':
            train_size = int(0.9 * len(text))
            text = text[train_size:]

        # Create dataset
        dataset = HFDataset.from_dict({ds_config.text_field: [text]})

        # Convert to IterableDataset if streaming requested
        if ds_config.streaming:
            dataset = dataset.to_iterable_dataset()

        return dataset

    def _apply_percentage_sampling(
        self,
        dataset: Union[Dataset, IterableDataset],
        ds_config: DatasetConfig
    ) -> Union[Dataset, IterableDataset]:
        """
        Sample a percentage of the dataset.

        Args:
            dataset: Input dataset
            ds_config: Dataset configuration with percentage

        Returns:
            Sampled dataset
        """
        percentage = ds_config.percentage

        if percentage >= 1.0:
            return dataset

        logger.info(f"Sampling {percentage*100:.1f}% of dataset")

        # Map-style dataset: use select
        if isinstance(dataset, Dataset):
            total_size = len(dataset)
            sample_size = int(total_size * percentage)
            return dataset.select(range(sample_size))

        # Iterable dataset: use take
        else:
            # For streaming datasets, try to get actual size from dataset info
            try:
                # Try to get dataset info (works for HuggingFace datasets)
                if hasattr(dataset, '_info') and dataset._info is not None:
                    # Get split info
                    split_info = dataset._info.splits.get(ds_config.split)
                    if split_info and hasattr(split_info, 'num_examples'):
                        total_size = split_info.num_examples
                        sample_size = int(total_size * percentage)
                        logger.info(f"Streaming dataset size: {total_size:,} items, taking {sample_size:,} ({percentage*100:.1f}%)")
                        return dataset.take(sample_size)

                # Fallback: try to get n_shards info (alternative way)
                if hasattr(dataset, 'n_shards') and hasattr(dataset, '_ex_iterable'):
                    # Estimate based on shard info (approximate)
                    # Common datasets have known sizes we can hardcode
                    known_sizes = {
                        'wikitext-2-v1': {'train': 36718, 'validation': 3760, 'test': 4358},
                        'wikitext-103-v1': {'train': 1801350, 'validation': 3760, 'test': 4358},
                    }

                    # Try to match known datasets
                    dataset_name = ds_config.subset or ds_config.name
                    if dataset_name in known_sizes and ds_config.split in known_sizes[dataset_name]:
                        total_size = known_sizes[dataset_name][ds_config.split]
                        sample_size = int(total_size * percentage)
                        logger.info(f"Using known dataset size for {dataset_name}: {total_size:,} items, taking {sample_size:,} ({percentage*100:.1f}%)")
                        return dataset.take(sample_size)

                # Last resort: log warning and return full dataset
                logger.warning(
                    f"Cannot determine streaming dataset size for {ds_config.name}. "
                    f"Percentage sampling ({percentage*100:.1f}%) will not be applied. "
                    "Using full dataset instead."
                )
                return dataset

            except AttributeError:
                logger.warning(
                    "Dataset does not support .take(), "
                    "percentage sampling may not work correctly"
                )
                return dataset

    def _apply_domain_filter(
        self,
        dataset: Union[Dataset, IterableDataset],
        domains: List[str]
    ) -> Union[Dataset, IterableDataset]:
        """
        Filter dataset by domains.

        Requires dataset to have a 'domain' field.

        Args:
            dataset: Input dataset
            domains: List of domains to keep

        Returns:
            Filtered dataset
        """
        logger.info(f"Filtering by domains: {domains}")

        def domain_filter(example):
            return example.get('domain') in domains

        return dataset.filter(domain_filter)

    def get_dataset_info(self) -> Dict:
        """
        Get information about configured datasets.

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'num_datasets': len(self.dataset_configs),
            'datasets': []
        }

        for ds_config in self.dataset_configs:
            ds_info = {
                'name': ds_config.name,
                'ratio': ds_config.ratio,
                'streaming': ds_config.streaming,
                'percentage': ds_config.percentage,
            }
            if ds_config.subset:
                ds_info['subset'] = ds_config.subset
            if ds_config.domains:
                ds_info['domains'] = ds_config.domains

            info['datasets'].append(ds_info)

        return info


def create_dataset_manager(config: Dict) -> DatasetManager:
    """
    Factory function to create DatasetManager.

    Args:
        config: Training configuration

    Returns:
        DatasetManager instance
    """
    return DatasetManager(config)
