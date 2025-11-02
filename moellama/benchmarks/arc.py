"""
ARC (AI2 Reasoning Challenge) Benchmark.

Two versions:
- ARC-Easy: Easier questions
- ARC-Challenge: More challenging questions

https://huggingface.co/datasets/allenai/ai2_arc
"""

from loguru import logger
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from moellama.benchmarks.base import MultipleChoiceBenchmark



class ARCBenchmark(MultipleChoiceBenchmark):
    """
    ARC benchmark base class.

    Handles both Easy and Challenge variants.
    """

    def __init__(
        self,
        subset: str = "ARC-Easy",
        split: str = "test",
        **kwargs
    ):
        """
        Initialize ARC benchmark.

        Args:
            subset: 'ARC-Easy' or 'ARC-Challenge'
            split: 'train', 'validation', or 'test'
            **kwargs: Additional args for Benchmark base class
        """
        assert subset in ["ARC-Easy", "ARC-Challenge"], \
            f"subset must be 'ARC-Easy' or 'ARC-Challenge', got '{subset}'"
        assert split in ["train", "validation", "test"], \
            f"split must be 'train', 'validation', or 'test', got '{split}'"

        name = f"ARC-{subset.split('-')[1]}"  # ARC-Easy or ARC-Challenge
        super().__init__(name=name, **kwargs)

        self.subset = subset
        self.split = split

    def load_dataset(self) -> None:
        """Load ARC dataset from HuggingFace."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        logger.info(f"Loading {self.name} ({self.split} split)...")
        self.dataset = load_dataset("allenai/ai2_arc", self.subset, split=self.split)

        # Shuffle with fixed seed for reproducibility
        self.dataset = self.dataset.shuffle(seed=42)

        logger.info(f"Loaded {len(self.dataset)} examples")

    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format ARC example into a prompt.

        Example format:
        Question: [question]
        A. [choice A]
        B. [choice B]
        C. [choice C]
        D. [choice D]
        Answer:
        """
        question = example["question"]
        choices_text = example["choices"]["text"]
        choices_labels = example["choices"]["label"]

        # Build prompt
        prompt = f"Question: {question}\n"
        for label, text in zip(choices_labels, choices_text):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"

        return prompt

    def extract_answer(self, generation: str, example: Dict) -> str:
        """Extract answer choice from generation."""
        # Get valid choices for this question
        valid_choices = example["choices"]["label"]
        generation = generation.strip().upper()

        # Look for first valid choice
        for char in generation[:20]:
            if char in valid_choices:
                return char

        # Default to first choice
        logger.warning(f"Could not extract answer from: '{generation[:30]}...'")
        return valid_choices[0]

    @property
    def description(self) -> str:
        return f"ARC {self.subset.split('-')[1]} - Science question answering"


class ARCEasy(ARCBenchmark):
    """ARC-Easy benchmark."""
    def __init__(self, **kwargs):
        super().__init__(subset="ARC-Easy", **kwargs)


class ARCChallenge(ARCBenchmark):
    """ARC-Challenge benchmark."""
    def __init__(self, **kwargs):
        super().__init__(subset="ARC-Challenge", **kwargs)
