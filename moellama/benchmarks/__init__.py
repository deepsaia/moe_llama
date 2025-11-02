"""
Benchmark Suite for LLM Evaluation.

Comprehensive set of standard LLM benchmarks including:
- ARC (Easy & Challenge): Science question answering
- MMLU: Multitask language understanding
- GSM8K: Grade school math reasoning
- HellaSwag: Commonsense reasoning
- WinoGrande: Pronoun resolution

Usage:
    from moellama.benchmarks import BenchmarkEvaluator, ARCEasy, MMLU

    evaluator = BenchmarkEvaluator(model, tokenizer, device)
    benchmarks = [ARCEasy(max_samples=100), MMLU(max_samples=100)]
    results = evaluator.evaluate_all(benchmarks)
"""

from moellama.benchmarks.base import Benchmark, MultipleChoiceBenchmark, GenerativeBenchmark
from moellama.benchmarks.evaluator import BenchmarkEvaluator
from moellama.benchmarks.arc import ARCEasy, ARCChallenge
from moellama.benchmarks.mmlu import MMLU
from moellama.benchmarks.gsm8k import GSM8K
from moellama.benchmarks.hellaswag import HellaSwag
from moellama.benchmarks.winogrande import WinoGrande

__all__ = [
    # Base classes
    'Benchmark',
    'MultipleChoiceBenchmark',
    'GenerativeBenchmark',
    # Evaluator
    'BenchmarkEvaluator',
    # Benchmarks
    'ARCEasy',
    'ARCChallenge',
    'MMLU',
    'GSM8K',
    'HellaSwag',
    'WinoGrande',
    # Factory functions
    'get_default_benchmarks',
    'get_comprehensive_benchmarks',
]


def get_default_benchmarks(max_samples: int = 100):
    """
    Get default set of benchmarks for quick evaluation.

    Args:
        max_samples: Max samples per benchmark

    Returns:
        List of benchmark instances
    """
    return [
        ARCEasy(max_samples=max_samples),
        ARCChallenge(max_samples=max_samples),
        MMLU(subject="all", max_samples=max_samples),
        GSM8K(max_samples=max_samples),
        HellaSwag(max_samples=max_samples),
        WinoGrande(max_samples=max_samples),
    ]


def get_comprehensive_benchmarks(max_samples: int = None):
    """
    Get comprehensive set of benchmarks for full evaluation.

    Args:
        max_samples: Max samples per benchmark (None = all)

    Returns:
        List of benchmark instances
    """
    return [
        ARCEasy(max_samples=max_samples),
        ARCChallenge(max_samples=max_samples),
        MMLU(subject="all", max_samples=max_samples),
        GSM8K(max_samples=max_samples),
        HellaSwag(max_samples=max_samples),
        WinoGrande(max_samples=max_samples),
    ]
