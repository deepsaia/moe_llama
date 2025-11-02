"""
Dataset utilities for training the language model.

This module provides:
- TextDataset: PyTorch dataset for tokenized text
- prepare_dataset: Helper to download and prepare training data
- download_tiny_shakespeare: Download the default dataset
"""

from loguru import logger
import requests
from pathlib import Path

import torch
from torch.utils.data import Dataset



class TextDataset(Dataset):
    """
    PyTorch dataset for language modeling.

    Converts raw text into fixed-length sequences of token IDs for training.
    Handles padding and creates multiple training examples from long texts.

    Args:
        texts: List of text strings to tokenize
        tokenizer: Tokenizer instance with encode/decode methods
        seq_len: Maximum sequence length (longer sequences are split)
    """

    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_ids = []

        logger.info(f"Initializing TextDataset with {len(texts)} texts, seq_len={seq_len}")

        # Warn about large datasets
        if len(texts) > 10000:
            logger.warning(
                f"Large dataset ({len(texts)} texts) - tokenization may take several minutes. "
                "Consider using streaming=true for faster loading."
            )

        # Tokenize all texts
        total_texts = len(texts)
        for idx, text in enumerate(texts):
            # Progress logging
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Tokenizing: {idx + 1}/{total_texts} texts...")

            # Process in chunks to avoid memory issues with huge texts
            chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]

            for chunk in chunks:
                # Add EOS token at the end of each chunk
                encoded = self.tokenizer.encode(chunk + " [EOS]")
                self.token_ids.extend(encoded)

        # Split into sequences of length seq_len
        self.sequences = []
        for i in range(0, len(self.token_ids), self.seq_len):
            seq = self.token_ids[i:i + self.seq_len]

            # Pad if needed
            if len(seq) < self.seq_len:
                seq = seq + [self.tokenizer.pad_token_id] * (self.seq_len - len(seq))

            self.sequences.append(seq)

        logger.info(
            f"Created {len(self.sequences)} sequences from "
            f"{len(self.token_ids)} tokens (seq_len={self.seq_len})"
        )

    def __len__(self):
        """Return number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a sequence by index.

        Args:
            idx: Index of sequence to retrieve

        Returns:
            Tensor of token IDs [seq_len]
        """
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def download_tiny_shakespeare(data_dir="dataset"):
    """
    Download the Tiny Shakespeare dataset.

    This is a ~1MB dataset containing Shakespeare's works, commonly
    used for character-level or small-scale language modeling experiments.

    Source: Karpathy's char-rnn repository

    Args:
        data_dir: Directory to save the dataset (default: "dataset")

    Returns:
        String containing the full text
    """
    logger.info("Downloading Tiny Shakespeare dataset")

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    file_path = data_dir / "tiny_shakespeare.txt"

    # Use cached file if available
    if file_path.exists():
        logger.info(f"Using cached dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    # Download from source
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()

        text = response.text

        # Save for future use
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        logger.info(
            f"Downloaded Tiny Shakespeare ({len(text)} characters) "
            f"and saved to {file_path}"
        )
        return text

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise


def prepare_dataset(config, tokenizer=None, use_streaming=None, device=None):
    """
    Prepare training and evaluation datasets.

    This function supports multiple modes:
    1. Single dataset (backward compatible)
    2. Multiple datasets with mixing
    3. Streaming for large datasets

    Config formats supported:
    - config['training']['dataset'] = "name"  # Single dataset (old)
    - config['training']['datasets'] = ["name1", "name2"]  # Multiple (equal ratios)
    - config['training']['dataset_mixture'] = [...]  # Multiple (custom ratios)

    Args:
        config: Configuration dictionary with 'training' section
        tokenizer: Optional pre-trained tokenizer (if None, creates new one)
        use_streaming: Force streaming mode (None = auto-detect from config, True/False = explicit)
        device: Actual device object from setup_device() (for streaming mode)

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
        OR
        Tuple of (train_loader, eval_loader, tokenizer) if streaming=True
    """
    training_config = config.get('training', {})

    if not training_config:
        raise ValueError(
            "No 'training' section found in configuration. "
            "Please add a 'training' section with dataset configuration."
        )

    # Detect configuration mode
    is_multi_dataset = (
        'dataset_mixture' in training_config or
        'datasets' in training_config
    )

    # Check that at least one dataset config exists
    has_dataset_config = (
        'dataset' in training_config or
        'datasets' in training_config or
        'dataset_mixture' in training_config
    )

    if not has_dataset_config:
        raise ValueError(
            "No dataset configuration found. Please specify one of:\n"
            "  - training.dataset = 'dataset_name'  (single dataset)\n"
            "  - training.datasets = ['ds1', 'ds2']  (multiple datasets, equal ratios)\n"
            "  - training.dataset_mixture = [...]  (multiple datasets with custom ratios)"
        )

    try:
        if is_multi_dataset:
            # NEW: Multi-dataset mode
            logger.info("Using multi-dataset mode")
            return _prepare_multi_dataset(config, tokenizer, use_streaming, device)
        else:
            # OLD: Single dataset mode (backward compatible)
            logger.info("Using single-dataset mode (backward compatible)")
            return _prepare_single_dataset(config, tokenizer, use_streaming, device)
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {str(e)}")
        raise


def _prepare_single_dataset(config, tokenizer=None, use_streaming=None, device=None):
    """
    Prepare single dataset (backward compatible).

    This preserves the original behavior exactly, with added support for
    subset, split, and streaming parameters.
    """
    logger.info(f"Preparing single dataset (backward compatible mode)")

    training_config = config['training']
    dataset_name = training_config['dataset']
    data_dir = training_config.get('data_dir', 'dataset')

    logger.info(f"Dataset: {dataset_name}")

    # Load dataset based on name
    if dataset_name == "tiny_shakespeare":
        # Download Tiny Shakespeare
        text = download_tiny_shakespeare(data_dir)
        train_size = int(0.9 * len(text))
        train_text = text[:train_size]
        eval_text = text[train_size:]

        logger.info(
            f"Dataset split: train={len(train_text)} chars, "
            f"eval={len(eval_text)} chars"
        )

        # Create BPE tokenizer if not provided
        if tokenizer is None:
            from moellama.tokenizer import BPETokenizer
            logger.info(f"Training BPE tokenizer on {dataset_name}...")
            tokenizer = BPETokenizer(
                text=train_text,
                special_tokens=["[PAD]", "[EOS]", "[UNK]"]
            )

        # Create datasets
        train_dataset = TextDataset(
            [train_text],
            tokenizer,
            seq_len=config['training']['seq_len']
        )
        eval_dataset = TextDataset(
            [eval_text],
            tokenizer,
            seq_len=config['training']['seq_len']
        )

    else:
        # Load from Hugging Face datasets
        logger.info(f"Loading dataset '{dataset_name}' from Hugging Face")
        from datasets import load_dataset

        # Get optional parameters from config
        subset = training_config.get('subset', None)
        split = training_config.get('split', 'train')
        streaming = training_config.get('streaming', False)

        # Build load_dataset arguments
        load_kwargs = {'path': dataset_name, 'split': split}
        if subset:
            load_kwargs['name'] = subset
            logger.info(f"Using subset: {subset}")

        logger.info(f"Loading split: {split}, streaming: {streaming}")

        if streaming:
            # Streaming mode - use DatasetManager for better support
            logger.info("Streaming mode detected, using DatasetManager for efficient loading")
            return _prepare_multi_dataset(config, tokenizer, use_streaming=True, device=device)

        # Non-streaming mode - load directly
        dataset = load_dataset(**load_kwargs)
        texts = dataset["text"]
        logger.info(f"Loaded {len(texts)} examples")

        # Join a sample to train tokenizer
        sample_text = " ".join(texts[:1000])

        if tokenizer is None:
            from moellama.tokenizer import BPETokenizer
            tokenizer = BPETokenizer(
                text=sample_text,
                special_tokens=["[PAD]", "[EOS]", "[UNK]"]
            )

        # Split into train/eval
        train_size = int(0.9 * len(texts))
        train_texts = texts[:train_size]
        eval_texts = texts[train_size:]

        train_dataset = TextDataset(
            train_texts,
            tokenizer,
            seq_len=config['training']['seq_len']
        )
        eval_dataset = TextDataset(
            eval_texts,
            tokenizer,
            seq_len=config['training']['seq_len']
        )

    logger.info(f"Train dataset: {len(train_dataset)} sequences")
    logger.info(f"Eval dataset: {len(eval_dataset)} sequences")

    return train_dataset, eval_dataset, tokenizer


def _prepare_multi_dataset(config, tokenizer=None, use_streaming=False, device=None):
    """
    Prepare multiple datasets with mixing (NEW).

    Uses DatasetManager for loading and mixing datasets.
    For non-streaming mode, returns TextDataset objects.
    For streaming mode, returns StreamingDataLoader objects.

    Args:
        config: Configuration dictionary
        tokenizer: Optional tokenizer
        use_streaming: Force streaming mode
        device: Actual device object from setup_device() (for streaming mode)

    Returns:
        Tuple of (train_dataset/loader, eval_dataset/loader, tokenizer)
        - For non-streaming: (TextDataset, TextDataset, tokenizer)
        - For streaming: (StreamingDataLoader, StreamingDataLoader, tokenizer)
    """
    logger.info("Preparing multi-dataset configuration")

    from moellama.dataset_manager import DatasetManager

    training_config = config['training']

    # Create separate configs for train and eval
    train_config = training_config.copy()
    eval_config = training_config.copy()

    # Update splits
    if 'dataset_mixture' in train_config:
        # Update split for each dataset in mixture
        train_mixture = []
        eval_mixture = []

        # Get eval_percentage from config (default: 0.1 = 10%)
        # This controls what percentage of each dataset to use for evaluation
        # For example, if train uses 50% of a dataset, eval might use 10%
        eval_percentage = training_config.get('eval_percentage', 0.1)

        for ds_dict in train_config['dataset_mixture']:
            # Train split
            train_ds = ds_dict.copy()
            train_ds['split'] = 'train'
            train_mixture.append(train_ds)

            # Eval split (use smaller percentage for eval)
            eval_ds = ds_dict.copy()
            eval_ds['split'] = 'validation' if 'validation' in ds_dict.get('split', '') else 'train'

            # Apply eval_percentage limit to avoid large eval sets
            # Uses min(eval_percentage, dataset_percentage) to ensure eval <= train size
            dataset_percentage = ds_dict.get('percentage', 1.0)
            eval_ds['percentage'] = min(eval_percentage, dataset_percentage)
            eval_mixture.append(eval_ds)

        train_config['dataset_mixture'] = train_mixture
        eval_config['dataset_mixture'] = eval_mixture

    # Load datasets
    train_manager = DatasetManager(train_config)
    train_hf_dataset = train_manager.load_datasets()

    eval_manager = DatasetManager(eval_config)
    eval_hf_dataset = eval_manager.load_datasets()

    logger.info("Datasets loaded successfully")
    logger.info(f"Train dataset info: {train_manager.get_dataset_info()}")
    logger.info(f"Eval dataset info: {eval_manager.get_dataset_info()}")

    # Create or train tokenizer
    if tokenizer is None:
        logger.info("Training tokenizer on sample data...")
        # Sample some text from train dataset to train tokenizer
        sample_texts = []
        for i, example in enumerate(train_hf_dataset):
            if i >= 1000:  # Use first 1000 examples
                break
            text = example.get('text', '')
            if text:
                sample_texts.append(text)

        sample_text = " ".join(sample_texts[:100])  # Use first 100 for tokenizer

        from moellama.tokenizer import BPETokenizer
        tokenizer = BPETokenizer(
            text=sample_text,
            special_tokens=["[PAD]", "[EOS]", "[UNK]"]
        )

    # Determine streaming mode
    # Priority: 1) explicitly passed, 2) from dataset config, 3) auto-detect from type
    if use_streaming is None:
        # Check if datasets were loaded with streaming mode from config
        streaming_requested = any(ds.streaming for ds in train_manager.dataset_configs)

        if streaming_requested:
            use_streaming = True
            logger.info("Using streaming mode (requested in dataset config)")
        else:
            # Auto-detect from dataset type
            try:
                from datasets import IterableDataset as HFIterableDataset
                use_streaming = isinstance(train_hf_dataset, HFIterableDataset)
                logger.info(f"Auto-detected streaming mode from dataset type: {use_streaming}")
            except ImportError:
                # Fallback to attribute check
                use_streaming = hasattr(train_hf_dataset, '__iter__') and not hasattr(train_hf_dataset, '__len__')
                logger.info(f"Auto-detected streaming mode from attributes: {use_streaming}")
    else:
        logger.info(f"Using explicitly specified streaming mode: {use_streaming}")

    seq_len = training_config['seq_len']

    # Handle based on streaming mode
    if use_streaming:
        # Streaming mode: Use StreamingDataLoader
        from moellama.streaming_dataloader import StreamingDataLoader

        logger.info("Creating streaming data loaders...")

        # Use actual device object from setup_device(), not config string
        if device is None:
            # Fallback for backward compatibility
            device_str = training_config.get('device', 'cpu')
            logger.warning(
                f"Device not passed to prepare_dataset(), falling back to config string: {device_str}. "
                "This may cause device mismatches. Please pass device from setup_device()."
            )
            device = device_str
        else:
            logger.info(f"Using device from setup_device(): {device}")

        batch_size = training_config['batch_size']

        train_loader = StreamingDataLoader(
            dataset=train_hf_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        eval_loader = StreamingDataLoader(
            dataset=eval_hf_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        logger.info("Streaming data loaders created successfully")
        return train_loader, eval_loader, tokenizer

    else:
        # Non-streaming mode: Convert to TextDataset objects
        logger.info("Converting to TextDataset objects...")

        # Check if dataset is too large for non-streaming mode
        try:
            dataset_len = len(train_hf_dataset)
            if dataset_len > 100000:
                logger.error(
                    f"Dataset too large for non-streaming mode ({dataset_len} examples). "
                    "Please set streaming=true in your dataset config."
                )
                raise ValueError(
                    f"Dataset has {dataset_len} examples, which is too large for non-streaming mode. "
                    "Set streaming=true in your dataset configuration to enable efficient loading. "
                    "Example: {name: 'your-dataset', streaming: true}"
                )
            elif dataset_len > 50000:
                logger.warning(
                    f"Large dataset detected ({dataset_len} examples). "
                    "Tokenization will take several minutes. "
                    "Consider using streaming=true for faster loading and lower memory usage."
                )
        except TypeError:
            # Some datasets don't support len()
            logger.info("Dataset size unknown (streaming-only dataset)")
            pass

        # Extract text from HuggingFace datasets
        train_texts = []
        logger.info("Extracting texts from train dataset...")
        for i, example in enumerate(train_hf_dataset):
            text = example.get('text', '')
            if text:
                train_texts.append(text)

            # Progress logging for large datasets
            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i + 1} examples...")

        logger.info("Extracting texts from eval dataset...")
        eval_texts = []
        for i, example in enumerate(eval_hf_dataset):
            text = example.get('text', '')
            if text:
                eval_texts.append(text)

            # Progress logging for large datasets
            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i + 1} examples...")

        logger.info(f"Extracted {len(train_texts)} train texts and {len(eval_texts)} eval texts")

        # Create TextDataset objects
        train_dataset = TextDataset(
            train_texts,
            tokenizer,
            seq_len=seq_len
        )

        eval_dataset = TextDataset(
            eval_texts,
            tokenizer,
            seq_len=seq_len
        )

        logger.info(f"Train dataset: {len(train_dataset)} sequences")
        logger.info(f"Eval dataset: {len(eval_dataset)} sequences")

        return train_dataset, eval_dataset, tokenizer
