"""
Streaming DataLoader for Large-Scale Training.

Inspired by nanochat's tokenizing_distributed_data_loader, this module provides
efficient streaming data loading with token buffering for both single and
multi-dataset configurations.

Key Features:
- Streaming token buffer (no disk storage)
- Distributed training support (DDP)
- Memory-efficient for 100B+ token datasets
- Works with Dataset and IterableDataset
- Backward compatible with existing TextDataset
"""

from loguru import logger
from collections import deque
from typing import Iterator, Tuple, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None
    HFIterableDataset = None



class StreamingDataLoader:
    """
    Streaming data loader with token buffering.

    Similar to nanochat's approach:
    1. Stream text from datasets
    2. Tokenize on-the-fly
    3. Buffer tokens
    4. Yield (input, target) batches

    This avoids storing tokenized data on disk and handles arbitrary dataset sizes.

    Args:
        dataset: HuggingFace Dataset or IterableDataset
        tokenizer: Tokenizer with encode/decode methods
        batch_size: Batch size (number of sequences)
        seq_len: Sequence length
        device: Device to place tensors ('cuda', 'cpu', 'mps')
        text_field: Field name containing text (default: 'text')
        rank: DDP rank (default: 0)
        world_size: DDP world size (default: 1)
        prepend_bos: Whether to prepend BOS token (default: True)

    Example:
        >>> loader = StreamingDataLoader(
        ...     dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     batch_size=16,
        ...     seq_len=256,
        ...     device='cuda'
        ... )
        >>> for inputs, targets in loader:
        ...     loss = model(inputs, targets)
    """

    def __init__(
        self,
        dataset: Union[HFDataset, HFIterableDataset],
        tokenizer,
        batch_size: int,
        seq_len: int,
        device: Union[str, torch.device] = 'cpu',
        text_field: str = 'text',
        rank: int = 0,
        world_size: int = 1,
        prepend_bos: bool = True,
        max_buffer_size: int = 100000,  # Maximum tokens to buffer
    ):
        """Initialize streaming data loader."""
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Normalize device: accept both string and torch.device object
        if isinstance(device, str):
            self.device = torch.device(device)
            logger.debug(f"Device string '{device}' converted to torch.device({device})")
        elif isinstance(device, torch.device):
            self.device = device
            logger.debug(f"Using torch.device object: {device}")
        else:
            logger.warning(f"Invalid device type {type(device)}, defaulting to CPU")
            self.device = torch.device('cpu')

        self.text_field = text_field
        self.rank = rank
        self.world_size = world_size
        self.prepend_bos = prepend_bos
        self.max_buffer_size = max_buffer_size

        # Calculate tokens needed per batch
        self.needed_tokens = batch_size * seq_len + 1  # +1 for target shift

        # Token buffer for streaming
        self.token_buffer = deque()

        # Check if dataset is iterable or map-style
        self.is_iterable = self._is_iterable_dataset(dataset)

        logger.info(
            f"StreamingDataLoader initialized: "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"device={self.device}, rank={rank}/{world_size}, "
            f"iterable={self.is_iterable}"
        )

    def _is_iterable_dataset(self, dataset) -> bool:
        """Check if dataset is iterable (streaming) or map-style."""
        if DATASETS_AVAILABLE:
            return isinstance(dataset, HFIterableDataset)
        return False

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches.

        Yields:
            Tuple of (inputs, targets) tensors
            - inputs: [batch_size, seq_len] token IDs
            - targets: [batch_size, seq_len] token IDs (shifted by 1)
        """
        # Create document iterator
        doc_iter = self._create_document_iterator()

        # Main streaming loop
        while True:
            # Fill token buffer
            self._fill_token_buffer(doc_iter)

            # Check if we have enough tokens
            if len(self.token_buffer) < self.needed_tokens:
                # Not enough tokens, dataset exhausted
                logger.info("Dataset exhausted, stopping iteration")
                break

            # Extract tokens for one batch
            tokens = [self.token_buffer.popleft() for _ in range(self.needed_tokens)]

            # Create tensors (pinned memory for faster GPU transfer)
            scratch = torch.tensor(
                tokens,
                dtype=torch.int64,
                pin_memory=(self.device.type == 'cuda')
            )

            # Split into inputs (all but last) and targets (all but first)
            inputs_cpu = scratch[:-1].to(dtype=torch.int32)
            targets_cpu = scratch[1:]

            # Reshape to [batch_size, seq_len]
            inputs = inputs_cpu.view(self.batch_size, self.seq_len).to(
                device=self.device,
                dtype=torch.int32,
                non_blocking=True
            )
            targets = targets_cpu.view(self.batch_size, self.seq_len).to(
                device=self.device,
                dtype=torch.int64,
                non_blocking=True
            )

            yield inputs, targets

    def _create_document_iterator(self) -> Iterator[str]:
        """
        Create an iterator over documents (text strings).

        Handles both map-style and iterable datasets.
        For DDP, each rank processes different documents.

        Yields:
            Text strings from dataset
        """
        if self.is_iterable:
            # Iterable dataset: iterate with DDP-aware skipping
            # Each rank processes every Nth document where N = world_size
            doc_index = 0
            for example in self.dataset:
                # Skip documents that don't belong to this rank
                if doc_index % self.world_size == self.rank:
                    text = self._extract_text(example)
                    if text:
                        yield text
                doc_index += 1
        else:
            # Map-style dataset: iterate with DDP slicing
            # Each rank processes different indices
            cursor = self.rank
            dataset_size = len(self.dataset)

            while cursor < dataset_size:
                example = self.dataset[cursor]
                text = self._extract_text(example)
                if text:
                    yield text

                cursor += self.world_size

                # Wrap around for multiple epochs
                if cursor >= dataset_size:
                    logger.info("Dataset epoch complete, wrapping around")
                    cursor = self.rank

    def _extract_text(self, example) -> Optional[str]:
        """
        Extract text from dataset example.

        Args:
            example: Dataset example (dict or other)

        Returns:
            Text string or None
        """
        if isinstance(example, dict):
            return example.get(self.text_field)
        elif isinstance(example, str):
            return example
        else:
            logger.warning(f"Unknown example type: {type(example)}")
            return None

    def _tokenize_and_stream(self, text: str, is_first_chunk: bool = True):
        """
        Tokenize text and stream into buffer.

        For huge documents, processes in chunks to avoid memory issues.
        Inspired by nanochat's approach but handles single huge documents.

        Args:
            text: Text to tokenize
            is_first_chunk: Whether this is the first chunk (for BOS token)
        """
        # For very long texts (>100k chars ~= >20-30k tokens), process in chunks
        # to avoid memory allocation issues
        max_chars_per_chunk = 100000  # Process ~20-30k tokens at a time

        if len(text) > max_chars_per_chunk:
            # Long document: process in chunks
            for i in range(0, len(text), max_chars_per_chunk):
                chunk = text[i:i + max_chars_per_chunk]
                is_first = is_first_chunk and (i == 0)

                # Tokenize chunk
                if self.prepend_bos and is_first:
                    bos_id = getattr(self.tokenizer, 'bos_token_id', None)
                    if bos_id is not None:
                        tokens = [bos_id] + self.tokenizer.encode(chunk)
                    else:
                        tokens = self.tokenizer.encode(chunk)
                else:
                    tokens = self.tokenizer.encode(chunk)

                # Stream tokens into buffer
                self.token_buffer.extend(tokens)

                # Stop if we have enough for a batch
                if len(self.token_buffer) >= self.needed_tokens:
                    return
        else:
            # Normal-sized document: tokenize all at once
            if self.prepend_bos and is_first_chunk:
                bos_id = getattr(self.tokenizer, 'bos_token_id', None)
                if bos_id is not None:
                    tokens = [bos_id] + self.tokenizer.encode(text)
                else:
                    tokens = self.tokenizer.encode(text)
            else:
                tokens = self.tokenizer.encode(text)

            self.token_buffer.extend(tokens)

    def _fill_token_buffer(self, doc_iter: Iterator[str]):
        """
        Fill token buffer from document iterator.

        Inspired by nanochat's approach: simple streaming with chunked tokenization
        for huge documents to prevent memory issues.

        Args:
            doc_iter: Iterator over text documents
        """
        # Keep filling until we have enough tokens for one batch
        while len(self.token_buffer) < self.needed_tokens:
            try:
                # Get next document
                text = next(doc_iter)

                # Skip empty documents
                if not text or len(text.strip()) == 0:
                    continue

                # Tokenize and stream into buffer (handles chunking internally)
                self._tokenize_and_stream(text, is_first_chunk=True)

            except StopIteration:
                # Document iterator exhausted
                logger.debug("Document iterator exhausted")
                break


class FallbackDataLoader:
    """
    Fallback data loader for backward compatibility.

    Uses the existing TextDataset and DataLoader for cases where
    streaming is not needed or not available.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True)
        num_workers: Number of data loading workers (default: 4)
        pin_memory: Whether to use pinned memory (default: True)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize fallback data loader."""
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(
            f"FallbackDataLoader initialized: "
            f"batch_size={batch_size}, shuffle={shuffle}, "
            f"num_workers={num_workers}"
        )

    def __iter__(self):
        """Iterate over batches."""
        for batch in self.loader:
            # batch is already a tensor [batch_size, seq_len]
            # Create targets by shifting by 1
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            yield inputs, targets

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)


def create_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: str = 'cuda',
    streaming: bool = None,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
):
    """
    Factory function to create appropriate data loader.

    Auto-detects whether to use streaming or fallback loader.

    Args:
        dataset: Dataset (HuggingFace or PyTorch)
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to place tensors
        streaming: Force streaming mode (None = auto-detect)
        rank: DDP rank
        world_size: DDP world size
        **kwargs: Additional arguments for specific loaders

    Returns:
        Data loader instance
    """
    # Determine if we should use streaming
    if streaming is None:
        # Auto-detect based on dataset type
        if DATASETS_AVAILABLE and isinstance(dataset, (HFDataset, HFIterableDataset)):
            streaming = True
        else:
            streaming = False

    if streaming:
        logger.info("Using StreamingDataLoader")
        return StreamingDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            rank=rank,
            world_size=world_size,
            **kwargs
        )
    else:
        logger.info("Using FallbackDataLoader")
        return FallbackDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            **kwargs
        )
