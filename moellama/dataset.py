"""
Dataset utilities for training the language model.

This module provides:
- TextDataset: PyTorch dataset for tokenized text
- prepare_dataset: Helper to download and prepare training data
- download_tiny_shakespeare: Download the default dataset
"""

import logging
import requests
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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

        # Tokenize all texts
        for text in texts:
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


def prepare_dataset(config, tokenizer=None):
    """
    Prepare training and evaluation datasets.

    This function:
    1. Downloads or loads the dataset
    2. Splits into train/eval (90/10)
    3. Creates or uses provided tokenizer
    4. Creates TextDataset instances

    Supports:
    - "tiny_shakespeare": Small Shakespeare dataset (~1MB)
    - Any HuggingFace dataset name (e.g., "wikitext", "openwebtext")

    Args:
        config: Configuration dictionary with 'training' section
        tokenizer: Optional pre-trained tokenizer (if None, creates new one)

    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    """
    logger.info(f"Preparing dataset: {config['training']['dataset']}")

    dataset_name = config['training']['dataset']
    data_dir = config['training'].get('data_dir', 'dataset')

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

        dataset = load_dataset(dataset_name)["train"]
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
