"""
Base Benchmark Class for LLM Evaluation.

Defines the interface that all benchmarks must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from loguru import logger



class Benchmark(ABC):
    """
    Base class for all benchmarks.

    Each benchmark implements:
    - Dataset loading
    - Prompt formatting
    - Answer extraction
    - Metric computation
    """

    def __init__(
        self,
        name: str,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        few_shot: int = 0,
    ):
        """
        Initialize benchmark.

        Args:
            name: Benchmark name
            max_samples: Maximum number of samples to evaluate (None = all)
            batch_size: Batch size for evaluation
            few_shot: Number of few-shot examples
        """
        self.name = name
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.few_shot = few_shot
        self.dataset = None

    @abstractmethod
    def load_dataset(self) -> None:
        """Load the benchmark dataset."""
        pass

    @abstractmethod
    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format an example into a prompt.

        Args:
            example: Dataset example
            few_shot_examples: Optional few-shot examples

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def extract_answer(self, generation: str, example: Dict) -> str:
        """
        Extract answer from model generation.

        Args:
            generation: Model generated text
            example: Original example for context

        Returns:
            Extracted answer
        """
        pass

    @abstractmethod
    def compute_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute benchmark-specific metrics.

        Args:
            predictions: Model predictions
            references: Ground truth answers

        Returns:
            Dictionary of metric names and values
        """
        pass

    def get_num_samples(self) -> int:
        """Get total number of samples in benchmark."""
        if self.dataset is None:
            self.load_dataset()

        total = len(self.dataset)
        if self.max_samples is not None:
            return min(total, self.max_samples)
        return total

    def get_examples(self) -> List[Dict]:
        """
        Get examples for evaluation.

        Returns:
            List of examples (limited by max_samples)
        """
        if self.dataset is None:
            self.load_dataset()

        num_samples = self.get_num_samples()
        return [self.dataset[i] for i in range(num_samples)]

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Primary metric name for this benchmark (e.g., 'accuracy', 'exact_match')."""
        pass

    @property
    def description(self) -> str:
        """Short description of the benchmark."""
        return f"{self.name} benchmark"


class MultipleChoiceBenchmark(Benchmark):
    """
    Base class for multiple choice benchmarks.

    Handles common logic for MC questions (ARC, MMLU, HellaSwag, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = ['A', 'B', 'C', 'D']

    def extract_answer(self, generation: str, example: Dict) -> str:
        """
        Extract answer from generation for MC questions.

        Looks for first occurrence of A, B, C, or D.
        """
        generation = generation.strip().upper()

        # Check first character
        if generation and generation[0] in self.choices:
            return generation[0]

        # Search in first few tokens
        for char in generation[:10]:
            if char in self.choices:
                return char

        # Default to first choice if not found
        logger.warning(f"Could not extract answer from: '{generation[:50]}...'. Defaulting to 'A'")
        return 'A'

    def compute_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute accuracy for MC questions."""
        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions) if predictions else 0.0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(predictions)
        }

    @property
    def metric_name(self) -> str:
        return 'accuracy'


class GenerativeBenchmark(Benchmark):
    """
    Base class for generative benchmarks.

    Handles benchmarks requiring free-form text generation (GSM8K, HumanEval, etc.)
    """

    def extract_answer(self, generation: str, example: Dict) -> str:
        """Default: return full generation."""
        return generation.strip()

    @property
    def metric_name(self) -> str:
        return 'exact_match'
