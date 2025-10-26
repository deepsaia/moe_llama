"""
Benchmark evaluation suite for the MoE language model.

This module provides various metrics and benchmarks to evaluate model performance:
- Perplexity: Measures how well the model predicts the next token
- Token accuracy: Percentage of correctly predicted tokens
- Generation quality: Sample text generation
- Simple reasoning: Basic counting and math tasks

All benchmarks are configurable and optional via config.hocon.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Suite of benchmarks for evaluating language model performance.

    All benchmarks are optional and configurable. Results are returned
    as a dictionary that can be formatted into a report.

    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        device: Compute device
        config: Configuration dictionary with evaluation settings
    """

    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.eval_config = config.get('evaluation', {})

    def run_all_benchmarks(self, eval_dataset=None) -> Dict[str, any]:
        """
        Run all enabled benchmarks.

        Args:
            eval_dataset: Optional evaluation dataset for perplexity/accuracy

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        # Check which benchmarks are enabled
        enabled = self.eval_config.get('enabled_benchmarks', [
            'perplexity', 'accuracy', 'generation', 'counting'
        ])

        logger.info(f"Running benchmarks: {', '.join(enabled)}")

        # Run enabled benchmarks
        if 'perplexity' in enabled and eval_dataset is not None:
            logger.info("Evaluating perplexity...")
            ppl = self.evaluate_perplexity(eval_dataset)
            results['perplexity'] = ppl

        if 'accuracy' in enabled and eval_dataset is not None:
            logger.info("Evaluating token accuracy...")
            acc = self.evaluate_accuracy(eval_dataset)
            results['token_accuracy'] = acc

        if 'generation' in enabled:
            logger.info("Evaluating generation quality...")
            samples = self.evaluate_generation()
            results['generation_samples'] = samples

        if 'counting' in enabled:
            logger.info("Evaluating counting ability...")
            count_score = self.evaluate_counting()
            results['counting_score'] = count_score

        if 'simple_math' in enabled:
            logger.info("Evaluating simple math...")
            math_score = self.evaluate_simple_math()
            results['math_score'] = math_score

        return results

    def evaluate_perplexity(self, eval_dataset) -> float:
        """
        Evaluate perplexity on the evaluation dataset.

        Perplexity measures how well the model predicts the next token.
        Lower is better (1.0 is perfect, typical values: 10-100).

        Formula: perplexity = exp(average_loss)

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Perplexity score (float)
        """
        self.model.eval()

        batch_size = self.eval_config.get('batch_size', 16)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: torch.stack(x)
        )

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Computing perplexity"):
                input_ids = batch.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, labels=input_ids, training=False)
                loss = outputs["loss"]

                # Count non-padding tokens
                num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def evaluate_accuracy(self, eval_dataset) -> float:
        """
        Evaluate token prediction accuracy.

        Measures the percentage of tokens where the model's top prediction
        matches the actual next token.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Accuracy as a percentage (0-100)
        """
        self.model.eval()

        batch_size = self.eval_config.get('batch_size', 16)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: torch.stack(x)
        )

        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Computing accuracy"):
                input_ids = batch.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, training=False)
                logits = outputs["logits"]

                # Get predictions (most likely token)
                predictions = torch.argmax(logits, dim=-1)

                # Shift for next-token prediction
                # Predict token i+1 from tokens 0..i
                pred_shifted = predictions[:, :-1]
                labels_shifted = input_ids[:, 1:]

                # Mask out padding tokens
                non_pad_mask = (labels_shifted != self.tokenizer.pad_token_id)

                # Count correct predictions
                correct = (pred_shifted == labels_shifted) & non_pad_mask
                total_correct += correct.sum().item()
                total_tokens += non_pad_mask.sum().item()

        accuracy = (total_correct / total_tokens) * 100
        logger.info(f"Token Accuracy: {accuracy:.2f}%")
        return accuracy

    def evaluate_generation(self) -> List[Dict[str, str]]:
        """
        Evaluate generation quality with sample prompts.

        Tests the model's ability to generate coherent text from
        various types of prompts.

        Returns:
            List of dictionaries with prompt and generated text
        """
        self.model.eval()

        # Get test prompts from config or use defaults
        prompts = self.eval_config.get('test_prompts', [
            "Once upon a time",
            "The future of",
            "In a distant land",
        ])

        max_tokens = self.eval_config.get('generation_max_tokens', 50)
        temperature = self.eval_config.get('generation_temperature', 0.8)

        samples = []

        for prompt in prompts:
            # Encode prompt
            token_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95
                )

            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())

            samples.append({
                'prompt': prompt,
                'generated': generated_text
            })

            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Generated: {generated_text}\n")

        return samples

    def evaluate_counting(self) -> float:
        """
        Evaluate the model's ability to count letters in words.

        Tests: "How many 'r's in 'strawberry'?"
        This is a simple reasoning task that LLMs often struggle with.

        Returns:
            Score between 0 and 1 (percentage correct)
        """
        self.model.eval()

        # Test cases: (prompt, correct_answer)
        test_cases = [
            ("How many r in strawberry? Answer:", "3"),
            ("How many l in hello? Answer:", "2"),
            ("How many e in tree? Answer:", "2"),
            ("How many a in banana? Answer:", "3"),
            ("How many o in book? Answer:", "2"),
        ]

        correct = 0
        total = len(test_cases)

        for prompt, answer in test_cases:
            # Encode and generate
            token_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=5,
                    temperature=0.1,  # Low temp for more deterministic
                )

            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            response = generated_text[len(prompt):].strip()

            # Check if correct answer appears in response
            if answer in response[:3]:  # Check first few characters
                correct += 1

            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Expected: {answer}, Got: {response}\n")

        score = correct / total
        logger.info(f"Counting Score: {correct}/{total} ({score*100:.1f}%)")
        return score

    def evaluate_simple_math(self) -> float:
        """
        Evaluate simple arithmetic (addition/subtraction).

        Tests basic math ability with single-digit operations.

        Returns:
            Score between 0 and 1 (percentage correct)
        """
        self.model.eval()

        # Test cases: (prompt, correct_answer)
        test_cases = [
            ("What is 2 + 3? Answer:", "5"),
            ("What is 5 + 4? Answer:", "9"),
            ("What is 7 - 2? Answer:", "5"),
            ("What is 8 - 3? Answer:", "5"),
            ("What is 6 + 1? Answer:", "7"),
        ]

        correct = 0
        total = len(test_cases)

        for prompt, answer in test_cases:
            # Encode and generate
            token_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=5,
                    temperature=0.1,
                )

            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            response = generated_text[len(prompt):].strip()

            # Check if correct answer appears
            if answer in response[:3]:
                correct += 1

            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Expected: {answer}, Got: {response}\n")

        score = correct / total
        logger.info(f"Math Score: {correct}/{total} ({score*100:.1f}%)")
        return score


    def evaluate_standard_benchmark(self, benchmark_name: str) -> Optional[float]:
        """
        Placeholder for standard LLM benchmarks (ARC, MMLU, GSM8K, etc.).

        These benchmarks require downloading specific datasets and
        implementing task-specific evaluation logic. This method provides
        a framework for adding them.

        Supported benchmarks (to be implemented):
        - ARC-Easy: AI2 Reasoning Challenge (easy subset)
        - ARC-Challenge: AI2 Reasoning Challenge (challenge subset)
        - MMLU: Massive Multitask Language Understanding
        - GSM8K: Grade School Math 8K
        - HumanEval: Code generation evaluation
        - ChatCORE: Chat-oriented reasoning evaluation

        Args:
            benchmark_name: Name of the benchmark to run

        Returns:
            Score (float) or None if not implemented

        Note:
            To implement these benchmarks:
            1. Download the benchmark dataset
            2. Implement task-specific prompting
            3. Implement task-specific scoring
            4. See DATASETS.md for examples
        """
        logger.warning(
            f"Standard benchmark '{benchmark_name}' not yet implemented. "
            "This is a placeholder for future extensions."
        )
        return None


def run_benchmarks(model, tokenizer, device, config, eval_dataset=None) -> Dict[str, any]:
    """
    Convenience function to run all benchmarks.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer
        device: Compute device
        config: Configuration dictionary
        eval_dataset: Optional evaluation dataset

    Returns:
        Dictionary of benchmark results
    """
    suite = BenchmarkSuite(model, tokenizer, device, config)
    return suite.run_all_benchmarks(eval_dataset)
