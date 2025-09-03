from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json
import os
import logging


logger = logging.getLogger("")

class BPETokenizer:
    def __init__(self, vocab_file=None, merges_file=None, text=None, special_tokens=None):
        self.tokenizer = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None

        if vocab_file and merges_file:
            # Load from vocab + merges files (old Hugging Face format)
            self.tokenizer = Tokenizer.from_file(vocab_file)
            self._setup_tokenizer()
            logger.info(f"Loaded BPE tokenizer from {vocab_file}")
        elif vocab_file:
            # ✅ New: Support loading from single JSON file
            self.tokenizer = Tokenizer.from_file(vocab_file)
            self._setup_tokenizer()
            logger.info(f"Loaded BPE tokenizer from {vocab_file}")
        elif text:
            # Train from scratch
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.train(text, special_tokens=special_tokens)
        else:
            # ✅ Allow empty init
            self.tokenizer = None
            logger.debug("Initialized empty BPETokenizer (call load_vocab() to load)")

    def _setup_tokenizer(self):
        """Apply standard settings after loading or training"""
        if self.tokenizer is None:
            raise ValueError("Cannot setup tokenizer: self.tokenizer is None")

        # ✅ Force-reapply padding and truncation
        try:
            self.tokenizer.enable_padding(pad_token="[PAD]", pad_id=0)
            # self.tokenizer.enable_truncation(max_length=512)
        except Exception as e:
            logger.warning(f"Failed to enable padding/truncation: {e}")

        # Now get token IDs
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")

        # ✅ Add strict validation
        if self.pad_token_id != 0:
            raise ValueError(f"Expected pad_token_id=0, got {self.pad_token_id}. "
                            "Make sure [PAD] is correctly defined.")
        if self.eos_token_id is None:
            raise ValueError("EOS token ID is None")
        if self.unk_token_id is None:
            raise ValueError("UNK token ID is None")

        logger.info(f"Tokenizer setup: pad={self.pad_token_id}, eos={self.eos_token_id}, unk={self.unk_token_id}")

    def train(self, text, vocab_size=5000, min_frequency=2, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["[PAD]", "[EOS]", "[UNK]"]

        # Pre-tokenizer splits on whitespace
        self.tokenizer.pre_tokenizer = Whitespace()

        # Trainer config
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )

        # Prepare text as list of strings (or split into lines)
        lines = text.split("\n")
        # Save temp file for training
        with open("temp_train.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Train tokenizer
        self.tokenizer.train(files=["temp_train.txt"], trainer=trainer)

        # Set post-processor (adds [EOS] at end, handles batching)
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            pair="$A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[EOS]", self.tokenizer.token_to_id("[EOS]"))],
        )

        self._setup_tokenizer()

        logger.info(f"BPE Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def save_vocab(self, file_path):
        # Save tokenizer to single JSON file
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save")
        self.tokenizer.save(file_path)
        logger.info(f"BPE tokenizer saved to {file_path}")

    def load_vocab(self, file_path):
        """Load tokenizer from a single JSON file (produced by Tokenizer.save())"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found at {file_path}")

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(file_path)

        # ✅ Use _setup_tokenizer to set padding, truncation, and token IDs
        self._setup_tokenizer()

        logger.info(f"BPE tokenizer loaded from {file_path}, vocab size: {self.tokenizer.get_vocab_size()}")

    def __len__(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.get_vocab_size()
