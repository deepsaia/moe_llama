"""
Benchmark Evaluator for LLM Models.

Orchestrates running benchmarks and collecting metrics.
"""

from loguru import logger
from typing import Dict, List, Optional, Any
from tqdm import tqdm

import torch

from moellama.benchmarks.base import Benchmark



class BenchmarkEvaluator:
    """
    Evaluates a model on multiple benchmarks.

    Handles:
    - Running benchmarks
    - Generating predictions
    - Computing metrics
    - Aggregating results
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 256,
        temperature: float = 0.0,  # Greedy for benchmarks
        batch_size: int = 1,
    ):
        """
        Initialize evaluator.

        Args:
            model: LLM model
            tokenizer: Tokenizer
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            batch_size: Batch size for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text (without prompt)
        """
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=1 if self.temperature == 0 else 50,  # Greedy if temp=0
        )

        # Decode only the generated part
        prompt_length = input_ids.shape[1]
        generated_ids = output_ids[0, prompt_length:]
        generation = self.tokenizer.decode(generated_ids.tolist())

        return generation

    def evaluate_benchmark(
        self,
        benchmark: Benchmark,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single benchmark.

        Args:
            benchmark: Benchmark to evaluate
            verbose: Show progress bar

        Returns:
            Dictionary with metrics and metadata
        """
        logger.info(f"Evaluating {benchmark.name}...")

        # Get examples
        examples = benchmark.get_examples()
        num_samples = len(examples)

        predictions = []
        references = []

        # Progress bar
        iterator = tqdm(examples, desc=f"Eval {benchmark.name}") if verbose else examples

        for example in iterator:
            # Format prompt
            prompt = benchmark.format_prompt(example)

            # Generate
            generation = self.generate(prompt)

            # Extract answer
            prediction = benchmark.extract_answer(generation, example)
            reference = example.get('answer', example.get('label', ''))

            predictions.append(prediction)
            references.append(str(reference))

        # Compute metrics
        metrics = benchmark.compute_metric(predictions, references)

        # Package results
        results = {
            'benchmark': benchmark.name,
            'num_samples': num_samples,
            'metrics': metrics,
            'primary_metric': metrics[benchmark.metric_name],
            'description': benchmark.description,
        }

        logger.info(
            f"{benchmark.name}: {benchmark.metric_name}={metrics[benchmark.metric_name]:.4f} "
            f"({num_samples} samples)"
        )

        return results

    def evaluate_all(
        self,
        benchmarks: List[Benchmark],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on multiple benchmarks.

        Args:
            benchmarks: List of benchmarks
            verbose: Show progress

        Returns:
            Dictionary with all results
        """
        self.model.eval()

        all_results = {}
        for benchmark in benchmarks:
            result = self.evaluate_benchmark(benchmark, verbose=verbose)
            all_results[benchmark.name] = result

        # Compute average score
        primary_scores = [r['primary_metric'] for r in all_results.values()]
        avg_score = sum(primary_scores) / len(primary_scores) if primary_scores else 0.0

        summary = {
            'benchmarks': all_results,
            'average_score': avg_score,
            'num_benchmarks': len(benchmarks),
        }

        logger.info(f"\nAverage score across {len(benchmarks)} benchmarks: {avg_score:.4f}")

        return summary
