"""
MMLU (Massive Multitask Language Understanding) Benchmark.

Tests knowledge across 57 subjects including STEM, humanities, social sciences.

https://huggingface.co/datasets/cais/mmlu
"""

from loguru import logger
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from moellama.benchmarks.base import MultipleChoiceBenchmark



class MMLU(MultipleChoiceBenchmark):
    """
    MMLU benchmark.

    Can evaluate on:
    - 'all': All 57 subjects combined
    - Specific subject: e.g., 'college_biology', 'high_school_mathematics'
    """

    SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_medicine',
        'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology',
        'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics',
        'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology',
        'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies',
        'machine_learning', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions'
    ]

    def __init__(
        self,
        subject: str = "all",
        split: str = "test",
        **kwargs
    ):
        """
        Initialize MMLU benchmark.

        Args:
            subject: 'all' for all subjects, or specific subject name
            split: 'auxiliary_train', 'validation', 'dev', or 'test'
            **kwargs: Additional args for Benchmark base class
        """
        if subject != "all":
            assert subject in self.SUBJECTS, \
                f"subject must be 'all' or one of {len(self.SUBJECTS)} subjects"

        name = f"MMLU-{subject}" if subject != "all" else "MMLU"
        super().__init__(name=name, **kwargs)

        self.subject = subject
        self.split = split

    def load_dataset(self) -> None:
        """Load MMLU dataset from HuggingFace."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        logger.info(f"Loading {self.name} ({self.split} split)...")

        if self.subject == "all":
            # Load all subjects
            datasets = []
            for subject in self.SUBJECTS:
                try:
                    ds = load_dataset("cais/mmlu", subject, split=self.split)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load subject {subject}: {e}")

            # Concatenate all subjects
            from datasets import concatenate_datasets
            self.dataset = concatenate_datasets(datasets)
        else:
            # Load specific subject
            self.dataset = load_dataset("cais/mmlu", self.subject, split=self.split)

        # Shuffle with fixed seed
        self.dataset = self.dataset.shuffle(seed=42)

        logger.info(f"Loaded {len(self.dataset)} examples")

    def format_prompt(self, example: Dict, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Format MMLU example into a prompt.

        Example format:
        Question: [question]
        A. [choice 0]
        B. [choice 1]
        C. [choice 2]
        D. [choice 3]
        Answer:
        """
        question = example["question"]
        choices = example["choices"]

        prompt = f"Question: {question}\n"
        for i, (letter, choice) in enumerate(zip(self.choices, choices)):
            prompt += f"{letter}. {choice}\n"
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
            references: List of correct answer indices (0/1/2/3)

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
        if self.subject == "all":
            return "MMLU - Multitask language understanding across 57 subjects"
        return f"MMLU - {self.subject.replace('_', ' ').title()}"
