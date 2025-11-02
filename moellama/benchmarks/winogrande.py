"""
WinoGrande Benchmark.

Tests commonsense reasoning through pronoun resolution tasks.

https://huggingface.co/datasets/winogrande
"""

from loguru import logger
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from moellama.benchmarks.base import Benchmark



class WinoGrande(Benchmark):
    """
    WinoGrande benchmark for pronoun resolution.

    Model must choose which of two options fills in the blank correctly.
    """

    def __init__(
        self,
        subset: str = "winogrande_xl",
        split: str = "validation",
        **kwargs
    ):
        """
        Initialize WinoGrande benchmark.

        Args:
            subset: 'winogrande_xs', 'winogrande_s', 'winogrande_m',
                   'winogrande_l', 'winogrande_xl', 'winogrande_debiased'
            split: 'train' or 'validation'
            **kwargs: Additional args for Benchmark base class
        """
        super().__init__(name="WinoGrande", **kwargs)
        self.subset = subset
        self.split = split

    def load_dataset(self) -> None:
        """Load WinoGrande dataset from HuggingFace."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        logger.info(f"Loading WinoGrande-{self.subset} ({self.split} split)...")
        self.dataset = load_dataset("winogrande", self.subset, split=self.split)

        # Shuffle with fixed seed
        self.dataset = self.dataset.shuffle(seed=42)

        logger.info(f"Loaded {len(self.dataset)} examples")

    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format WinoGrande example into a prompt.

        Example format:
        Sentence: [sentence with _ blank]
        Option 1: [option1]
        Option 2: [option2]
        Which option correctly fills the blank? (1 or 2)
        Answer:
        """
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]

        prompt = f"Sentence: {sentence}\n"
        prompt += f"Option 1: {option1}\n"
        prompt += f"Option 2: {option2}\n"
        prompt += "Which option correctly fills the blank? (1 or 2)\n"
        prompt += "Answer:"

        return prompt

    def extract_answer(self, generation: str, example: Dict) -> str:
        """
        Extract answer (1 or 2) from generation.
        """
        generation = generation.strip()

        # Look for "1" or "2" in the generation
        for char in generation[:10]:
            if char in ['1', '2']:
                return char

        # Look for words "one" or "two"
        generation_lower = generation.lower()
        if "option 1" in generation_lower or "one" in generation_lower[:20]:
            return "1"
        if "option 2" in generation_lower or "two" in generation_lower[:20]:
            return "2"

        # Default to 1
        logger.warning(f"Could not extract answer from: '{generation[:30]}...'")
        return "1"

    def compute_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute accuracy.

        Args:
            predictions: List of predicted answers ("1" or "2")
            references: List of correct answers ("1" or "2")

        Returns:
            Dictionary with accuracy metric
        """
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

    @property
    def description(self) -> str:
        return "WinoGrande - Commonsense reasoning via pronoun resolution"
