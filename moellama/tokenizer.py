"""
Byte-Pair Encoding (BPE) tokenizer for the language model.

BPE is a subword tokenization algorithm that balances between character-level
and word-level tokenization, allowing the model to handle:
- Common words as single tokens (efficient)
- Rare words as subword pieces (flexible)
- Unknown words by breaking into known pieces (robust)

This implementation uses the HuggingFace tokenizers library for efficient
BPE training and encoding.
"""

import os
import logging

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer wrapper.

    BPE learns to merge frequently co-occurring byte pairs to create
    a vocabulary of subword units. This provides a good balance between
    vocabulary size and the ability to represent any text.

    Special tokens:
        [PAD]: Padding token (ID 0)
        [EOS]: End of sequence token
        [UNK]: Unknown token for out-of-vocabulary pieces

    Args:
        vocab_file: Path to saved tokenizer JSON file
        vocab_size: Target vocabulary size (if training)
        merges_file: Legacy path to merges file (optional)
        text: Text to train tokenizer on
        special_tokens: List of special tokens to include
    """

    def __init__(
        self,
        vocab_file=None,
        vocab_size=None,
        merges_file=None,
        text=None,
        special_tokens=None
    ):
        self.tokenizer = None
        self.vocab_size = vocab_size
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None

        if vocab_file and merges_file:
            # Load from vocab + merges files (old HuggingFace format)
            self.tokenizer = Tokenizer.from_file(vocab_file)
            self._setup_tokenizer()
            logger.info(f"Loaded BPE tokenizer from {vocab_file}")

        elif vocab_file:
            # Load from single JSON file (modern format)
            self.tokenizer = Tokenizer.from_file(vocab_file)
            self._setup_tokenizer()
            logger.info(f"Loaded BPE tokenizer from {vocab_file}")

        elif text:
            # Train from scratch on provided text
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.train(text, special_tokens=special_tokens)

        else:
            # Allow empty initialization (load later with load_vocab)
            self.tokenizer = None
            logger.debug("Initialized empty BPETokenizer (call load_vocab() to load)")

    def _setup_tokenizer(self):
        """
        Apply standard settings after loading or training.

        This method:
        1. Enables padding with [PAD] token at ID 0
        2. Retrieves special token IDs
        3. Validates that special tokens are correctly configured
        """
        if self.tokenizer is None:
            raise ValueError("Cannot setup tokenizer: tokenizer is None")

        # Enable padding with [PAD] token
        try:
            self.tokenizer.enable_padding(pad_token="[PAD]", pad_id=0)
        except Exception as e:
            logger.warning(f"Failed to enable padding: {e}")

        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")

        # Validate special tokens
        if self.pad_token_id != 0:
            raise ValueError(
                f"Expected pad_token_id=0, got {self.pad_token_id}. "
                "Make sure [PAD] is the first special token."
            )
        if self.eos_token_id is None:
            raise ValueError("EOS token ID is None - missing from vocabulary")
        if self.unk_token_id is None:
            raise ValueError("UNK token ID is None - missing from vocabulary")

        logger.info(
            f"Tokenizer setup: pad={self.pad_token_id}, "
            f"eos={self.eos_token_id}, unk={self.unk_token_id}"
        )

    def train(self, text, vocab_size=5000, min_frequency=2, special_tokens=None):
        """
        Train BPE tokenizer on provided text.

        Training process:
        1. Split text on whitespace (pre-tokenization)
        2. Count character pair frequencies
        3. Iteratively merge most frequent pairs
        4. Continue until vocabulary reaches target size

        Args:
            text: Training text
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a pair to be merged
            special_tokens: List of special tokens to add to vocabulary
        """
        if special_tokens is None:
            special_tokens = ["[PAD]", "[EOS]", "[UNK]"]

        logger.info(f"Training BPE tokenizer on {len(text)} characters")

        # Pre-tokenizer: split on whitespace before BPE
        # This ensures BPE merges happen within words, not across them
        self.tokenizer.pre_tokenizer = Whitespace()

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size or vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,  # These get IDs 0, 1, 2, ...
            show_progress=True
        )

        # Save text to temporary file for training
        # (HuggingFace tokenizers requires file input)
        with open("temp_train.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Train the tokenizer
        self.tokenizer.train(files=["temp_train.txt"], trainer=trainer)

        # Configure post-processor
        # This automatically adds [EOS] token at the end of sequences
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",  # Single sequence: text + [EOS]
            pair="$A [EOS] $B:1 [EOS]:1",  # Pair: text1 + [EOS] + text2 + [EOS]
            special_tokens=[("[EOS]", self.tokenizer.token_to_id("[EOS]"))],
        )

        # Setup special token IDs and validation
        self._setup_tokenizer()

        logger.info(f"BPE training complete. Vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text):
        """
        Encode text to token IDs.

        Process:
        1. Pre-tokenize (split on whitespace)
        2. Apply BPE merges to get subwords
        3. Convert subwords to IDs

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids)

    def save_vocab(self, file_path):
        """
        Save tokenizer to a JSON file.

        The saved file contains:
        - Vocabulary (token -> ID mapping)
        - Merges (BPE merge operations)
        - Configuration (special tokens, etc.)

        Args:
            file_path: Path to save tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save")

        self.tokenizer.save(file_path)
        logger.info(f"BPE tokenizer saved to {file_path}")

    def load_vocab(self, file_path):
        """
        Load tokenizer from a JSON file.

        Args:
            file_path: Path to tokenizer file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(file_path)

        # Setup special tokens and validation
        self._setup_tokenizer()

        vocab_size = self.tokenizer.get_vocab_size()
        logger.info(f"BPE tokenizer loaded from {file_path}, vocab size: {vocab_size}")

    def __len__(self):
        """
        Get vocabulary size.

        Returns:
            Number of tokens in vocabulary
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.get_vocab_size()

    def get_vocab(self):
        """
        Get the full vocabulary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.get_vocab()
