"""
HellaSwag Benchmark.

Tests commonsense reasoning by asking models to complete scenarios.

https://huggingface.co/datasets/Rowan/hellaswag
"""

from loguru import logger
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from moellama.benchmarks.base import MultipleChoiceBenchmark



class HellaSwag(MultipleChoiceBenchmark):
    """
    HellaSwag benchmark for commonsense reasoning.

    Model must choose the most plausible continuation of a scenario.
    """

    def __init__(
        self,
        split: str = "validation",
        **kwargs
    ):
        """
        Initialize HellaSwag benchmark.

        Args:
            split: 'train' or 'validation'
            **kwargs: Additional args for Benchmark base class
        """
        super().__init__(name="HellaSwag", **kwargs)
        self.split = split

    def load_dataset(self) -> None:
        """Load HellaSwag dataset from HuggingFace."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        logger.info(f"Loading HellaSwag ({self.split} split)...")
        self.dataset = load_dataset("Rowan/hellaswag", split=self.split)

        # Shuffle with fixed seed
        self.dataset = self.dataset.shuffle(seed=42)

        logger.info(f"Loaded {len(self.dataset)} examples")

    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format HellaSwag example into a prompt.

        Example format:
        Context: [ctx_a] [ctx_b]
        What happens next?
        A. [ending 0]
        B. [ending 1]
        C. [ending 2]
        D. [ending 3]
        Answer:
        """
        # Get context
        ctx = example["ctx"]
        # Some examples have ctx_a and ctx_b that need to be combined
        if "ctx_a" in example and example["ctx_a"]:
            ctx = example["ctx_a"]
            if "ctx_b" in example and example["ctx_b"]:
                ctx += " " + example["ctx_b"].capitalize()

        endings = example["endings"]

        prompt = f"Context: {ctx}\n"
        prompt += "What happens next?\n"

        for i, (letter, ending) in enumerate(zip(self.choices, endings)):
            prompt += f"{letter}. {ending}\n"

        prompt += "Answer:"

        return prompt

    def extract_answer(self, generation: str, example: Dict) -> str:
        """Extract answer choice (A/B/C/D) from generation."""
        generation = generation.strip().upper()

        # Look for A, B, C, or D
        for char in generation[:20]:
            if char in self.choices:
                return char

        # Default to A
        logger.warning(f"Could not extract answer from: '{generation[:30]}...'")
        return 'A'

    def compute_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute accuracy.

        Args:
            predictions: List of predicted answers (A/B/C/D)
            references: List of correct answer indices (0/1/2/3) as strings

        Returns:
            Dictionary with accuracy metric
        """
        # Convert reference indices to letters
        ref_letters = [self.choices[int(ref)] for ref in references]

        correct = sum(p == r for p, r in zip(predictions, ref_letters))
        accuracy = correct / len(predictions) if predictions else 0.0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(predictions)
        }

    @property
    def description(self) -> str:
        return "HellaSwag - Commonsense reasoning via scenario completion"
