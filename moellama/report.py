"""
Report generation for model evaluation results.

This module creates formatted markdown reports summarizing:
- Model configuration and size
- Training details
- Benchmark results
- Sample generations

Reports are config-driven and customizable.
"""

from loguru import logger
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional



class ReportGenerator:
    """
    Generate formatted markdown reports for model evaluation.

    Reports include model info, training details, benchmark results,
    and sample generations. All sections are customizable via config.

    Args:
        config: Configuration dictionary
        output_path: Where to save the report (default: report.md)
    """

    def __init__(self, config, output_path="report.md"):
        self.config = config
        self.output_path = Path(output_path)
        self.sections = []

    def add_header(self, title="Model Evaluation Report"):
        """Add report header with title and timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# {title}

**Generated:** {timestamp}

---

"""
        self.sections.append(header)

    def add_model_info(self, model):
        """
        Add model architecture and parameter information.

        Args:
            model: The model to report on
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Format numbers
        def format_num(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            return str(num)

        section = f"""## Model Architecture

| Component | Value |
|-----------|-------|
| **Model Type** | Mixture of Experts (MoE) |
| **Dimension** | {self.config['model']['dim']} |
| **Layers** | {self.config['model']['num_layers']} |
| **Attention Heads** | {self.config['model']['num_heads']} |
| **Experts** | {self.config['model']['num_experts']} |
| **Top-K Experts** | {self.config['model']['top_k']} |
| **Shared Expert** | {self.config['model']['shared_expert']} |
| **Max Sequence Length** | {self.config['model']['max_seq_len']} |
| **Total Parameters** | {format_num(total_params)} ({total_params:,}) |
| **Trainable Parameters** | {format_num(trainable_params)} ({trainable_params:,}) |

"""
        self.sections.append(section)

    def add_training_info(self):
        """Add training configuration information."""
        train_config = self.config['training']

        section = f"""## Training Configuration

| Setting | Value |
|---------|-------|
| **Dataset** | {train_config['dataset']} |
| **Batch Size** | {train_config['batch_size']} |
| **Learning Rate** | {train_config['learning_rate']} |
| **Epochs** | {train_config['epochs']} |
| **Sequence Length** | {train_config['seq_len']} |
| **Optimizer** | AdamW |
| **Load Balancing Coef** | {self.config['model']['load_balancing_loss_coef']} |

"""
        self.sections.append(section)

    def add_benchmark_results(self, results: Dict[str, any]):
        """
        Add benchmark results section.

        Args:
            results: Dictionary of benchmark results from BenchmarkSuite
        """
        section = "## Benchmark Results\n\n"

        # Perplexity
        if 'perplexity' in results:
            ppl = results['perplexity']
            section += f"""### Perplexity

**Score:** {ppl:.2f}

Lower is better. Perplexity measures how well the model predicts the next token.
- Excellent: < 20
- Good: 20-50
- Fair: 50-100
- Poor: > 100

"""

        # Token Accuracy
        if 'token_accuracy' in results:
            acc = results['token_accuracy']
            section += f"""### Token Accuracy

**Score:** {acc:.2f}%

Percentage of tokens where the model's top prediction matches the actual next token.

"""

        # Counting Score
        if 'counting_score' in results:
            score = results['counting_score']
            percentage = score * 100
            section += f"""### Counting Ability

**Score:** {percentage:.1f}%

Tests the model's ability to count letters in words (e.g., "How many 'r's in 'strawberry'?").

"""

        # Math Score
        if 'math_score' in results:
            score = results['math_score']
            percentage = score * 100
            section += f"""### Simple Math

**Score:** {percentage:.1f}%

Tests basic arithmetic (single-digit addition/subtraction).

"""

        self.sections.append(section)

    def add_generation_samples(self, samples):
        """
        Add text generation samples.

        Args:
            samples: List of dictionaries with 'prompt' and 'generated' keys
        """
        if not samples:
            return

        section = "## Generation Samples\n\n"
        section += "Sample text generated by the model:\n\n"

        for i, sample in enumerate(samples, 1):
            section += f"""### Sample {i}

**Prompt:** {sample['prompt']}

**Generated:**
```
{sample['generated']}
```

"""

        self.sections.append(section)

    def add_summary_table(self, results: Dict[str, any]):
        """
        Add a summary table with key metrics.

        Args:
            results: Dictionary of benchmark results
        """
        section = "## Summary\n\n"
        section += "| Metric | Score |\n"
        section += "|--------|-------|\n"

        if 'perplexity' in results:
            section += f"| Perplexity | {results['perplexity']:.2f} |\n"

        if 'token_accuracy' in results:
            section += f"| Token Accuracy | {results['token_accuracy']:.2f}% |\n"

        if 'counting_score' in results:
            score = results['counting_score'] * 100
            section += f"| Counting Ability | {score:.1f}% |\n"

        if 'math_score' in results:
            score = results['math_score'] * 100
            section += f"| Simple Math | {score:.1f}% |\n"

        section += "\n"
        self.sections.append(section)

    def add_custom_section(self, title: str, content: str):
        """
        Add a custom section to the report.

        Args:
            title: Section title
            content: Section content (markdown formatted)
        """
        section = f"## {title}\n\n{content}\n\n"
        self.sections.append(section)

    def generate(self) -> str:
        """
        Generate the complete report.

        Returns:
            Report as a markdown string
        """
        return "".join(self.sections)

    def save(self):
        """Save the report to file."""
        report_text = self.generate()

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Report saved to {self.output_path}")

    def print_summary(self, results: Dict[str, any]):
        """
        Print a brief summary to console.

        Args:
            results: Dictionary of benchmark results
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        if 'perplexity' in results:
            print(f"Perplexity:        {results['perplexity']:.2f}")

        if 'token_accuracy' in results:
            print(f"Token Accuracy:    {results['token_accuracy']:.2f}%")

        if 'counting_score' in results:
            score = results['counting_score'] * 100
            print(f"Counting Ability:  {score:.1f}%")

        if 'math_score' in results:
            score = results['math_score'] * 100
            print(f"Simple Math:       {score:.1f}%")

        print("="*60 + "\n")


def generate_report(model, config, results: Dict[str, any], output_path="report.md"):
    """
    Convenience function to generate a complete report.

    Args:
        model: The evaluated model
        config: Configuration dictionary
        results: Benchmark results from BenchmarkSuite
        output_path: Where to save the report

    Returns:
        ReportGenerator instance (for further customization)
    """
    generator = ReportGenerator(config, output_path)

    # Add standard sections
    generator.add_header()
    generator.add_model_info(model)
    generator.add_training_info()
    generator.add_summary_table(results)
    generator.add_benchmark_results(results)

    # Add generation samples if available
    if 'generation_samples' in results:
        generator.add_generation_samples(results['generation_samples'])

    # Save and print summary
    generator.save()
    generator.print_summary(results)

    return generator
