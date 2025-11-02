"""
GSM8K (Grade School Math 8K) Benchmark.

Tests mathematical reasoning on grade school math problems.

https://huggingface.co/datasets/gsm8k
"""

from loguru import logger
import re
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from moellama.benchmarks.base import GenerativeBenchmark



class GSM8K(GenerativeBenchmark):
    """
    GSM8K benchmark for math reasoning.

    Expects model to generate the final numerical answer.
    """

    def __init__(
        self,
        split: str = "test",
        **kwargs
    ):
        """
        Initialize GSM8K benchmark.

        Args:
            split: 'train' or 'test'
            **kwargs: Additional args for Benchmark base class
        """
        super().__init__(name="GSM8K", **kwargs)
        self.split = split

    def load_dataset(self) -> None:
        """Load GSM8K dataset from HuggingFace."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        logger.info(f"Loading GSM8K ({self.split} split)...")
        self.dataset = load_dataset("gsm8k", "main", split=self.split)

        # Shuffle with fixed seed
        self.dataset = self.dataset.shuffle(seed=42)

        logger.info(f"Loaded {len(self.dataset)} examples")

    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format GSM8K example into a prompt.

        Example format:
        Q: [question]
        A: Let's think step by step.
        """
        question = example["question"]

        # Few-shot examples if provided
        prompt = ""
        if few_shot_examples:
            for fs_ex in few_shot_examples:
                prompt += f"Q: {fs_ex['question']}\n"
                prompt += f"A: {fs_ex['answer']}\n\n"

        # Current question
        prompt += f"Q: {question}\n"
        prompt += "A: Let's think step by step."

        return prompt

    def extract_answer(self, generation: str, example: Dict) -> str:
        """
        Extract numerical answer from generation.

        Looks for patterns like:
        - "The answer is 42"
        - "#### 42"
        - "= 42"
        - Final number in the text
        """
        generation = generation.strip()

        # Pattern 1: "#### [number]" (GSM8K format)
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', generation)
        if match:
            return match.group(1).replace(',', '')

        # Pattern 2: "The answer is [number]"
        match = re.search(r'(?:answer|total|result) is:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', generation, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')

        # Pattern 3: "= [number]" at end
        match = re.search(r'=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', generation)
        if match:
            return match.group(1).replace(',', '')

        # Pattern 4: Last number in generation
        matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', generation)
        if matches:
            return matches[-1].replace(',', '')

        logger.warning(f"Could not extract answer from: '{generation[:100]}...'")
        return ""

    def compute_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute exact match accuracy for GSM8K.

        Args:
            predictions: Predicted numerical answers
            references: Ground truth answers (with #### format)

        Returns:
            Dictionary with accuracy metrics
        """
        # Extract ground truth numbers
        true_answers = []
        for ref in references:
            # GSM8K answers are in format: "[reasoning] #### [answer]"
            match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', ref)
            if match:
                true_answers.append(match.group(1).replace(',', ''))
            else:
                # Try to find last number
                matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', ref)
                if matches:
                    true_answers.append(matches[-1].replace(',', ''))
                else:
                    true_answers.append("")

        # Compare predictions to ground truth
        correct = 0
        for pred, true in zip(predictions, true_answers):
            try:
                # Convert to float for comparison (handles decimals)
                pred_num = float(pred) if pred else None
                true_num = float(true) if true else None

                if pred_num is not None and true_num is not None:
                    # Check if numbers are close (for float comparison)
                    if abs(pred_num - true_num) < 0.01:
                        correct += 1
            except (ValueError, TypeError):
                # If conversion fails, do string comparison
                if pred == true:
                    correct += 1

        accuracy = correct / len(predictions) if predictions else 0.0

        return {
            'accuracy': accuracy,
            'exact_match': accuracy,  # Alias
            'correct': correct,
            'total': len(predictions)
        }

    @property
    def metric_name(self) -> str:
        return 'accuracy'

    @property
    def description(self) -> str:
        return "GSM8K - Grade school math word problems"
