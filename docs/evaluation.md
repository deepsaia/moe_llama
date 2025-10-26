# Evaluation & Benchmarking Guide

This guide explains the evaluation system in moellama - a config-driven, modular approach to benchmarking language models.

## 🎯 Overview

The evaluation system provides:
- **Config-driven**: Enable/disable benchmarks via `config.hocon`
- **Modular**: Easy to add custom benchmarks
- **Comprehensive**: Multiple evaluation metrics
- **Automated reports**: Markdown reports with detailed analysis
- **Optional**: Can be completely disabled if not needed

## 🏗️ Architecture

### Components

```
moellama/
├── benchmarks.py      # Evaluation metrics and benchmarks
├── report.py          # Report generation and formatting
└── ...

scripts/
└── evaluate.py        # Evaluation script
```

### Flow

```
Load Model & Tokenizer
    ↓
Load Evaluation Dataset (optional)
    ↓
Run Enabled Benchmarks
    ↓
Generate Report
    ↓
Save to report.md
```

## 📊 Available Benchmarks

### 1. Perplexity

**What it measures**: How well the model predicts the next token.

**Formula**: `perplexity = exp(average_cross_entropy_loss)`

**Interpretation**:
- Excellent: < 20
- Good: 20-50
- Fair: 50-100
- Poor: > 100

**Example**:
```
Perplexity: 45.2
```

Lower is better. A perplexity of 45 means the model is "45-ways confused" on average when predicting the next token.

### 2. Token Accuracy

**What it measures**: Percentage of tokens where the model's top prediction matches the actual next token.

**Example**:
```
Token Accuracy: 32.5%
```

This means 32.5% of the time, the model's most likely prediction is correct.

### 3. Generation Quality

**What it measures**: Qualitative assessment of generated text.

**Test**: Generate text from sample prompts and inspect coherence.

**Example**:
```
Prompt: "Once upon a time"
Generated: "Once upon a time in a distant land, there lived a..."
```

### 4. Counting Ability

**What it measures**: Ability to count letters in words (tests basic reasoning).

**Test cases**:
- "How many 'r's in 'strawberry'?" → Answer: 3
- "How many 'l's in 'hello'?" → Answer: 2

**Example**:
```
Counting Score: 60% (3/5 correct)
```

This is a challenging task even for larger models!

### 5. Simple Math

**What it measures**: Basic arithmetic ability (single-digit operations).

**Test cases**:
- "What is 2 + 3?" → Answer: 5
- "What is 7 - 2?" → Answer: 5

**Example**:
```
Math Score: 80% (4/5 correct)
```

### Standard Benchmarks (Future)

Placeholders are ready for:
- **ARC-Easy / ARC-Challenge**: Reasoning challenge questions
- **MMLU**: Multi-domain knowledge questions
- **GSM8K**: Grade school math word problems
- **HumanEval**: Code generation tasks
- **ChatCORE**: Chat-oriented reasoning

These require downloading specific datasets and implementing task-specific evaluation logic.

## ⚙️ Configuration

All evaluation settings are in `config.hocon`:

```hocon
evaluation {
  # Enable/disable entire evaluation system
  enabled = true

  # Select which benchmarks to run
  enabled_benchmarks = [
    "perplexity",      # Requires eval dataset
    "accuracy",        # Requires eval dataset
    "generation",      # Standalone
    "counting",        # Standalone
    "simple_math"      # Standalone
  ]

  # Evaluation batch size (for perplexity/accuracy)
  batch_size = 16

  # Generation benchmark settings
  test_prompts = [
    "Once upon a time",
    "The future of",
    "In a distant land"
  ]
  generation_max_tokens = 50
  generation_temperature = 0.8

  # Where to save the report
  report_path = "report.md"
}
```

### Disable Evaluation

To skip evaluation entirely:

```hocon
evaluation {
  enabled = false
}
```

### Run Only Specific Benchmarks

```hocon
evaluation {
  enabled_benchmarks = ["perplexity", "generation"]
}
```

## 🚀 Usage

### Basic Evaluation

```bash
# Evaluate the latest trained model
python -m scripts.evaluate
```

This will:
1. Load the latest model checkpoint
2. Load the latest tokenizer
3. Optionally load evaluation dataset
4. Run all enabled benchmarks
5. Generate `report.md`

### Advanced Options

```bash
# Specific model
python -m scripts.evaluate --model-file trained_models/model_tiny_shakespeare_20250126.pt

# Specific vocab
python -m scripts.evaluate --vocab-file trained_models/vocab_tiny_shakespeare.txt

# Custom report location
python -m scripts.evaluate --output my_evaluation.md

# Skip dataset loading (faster, but no perplexity/accuracy)
python -m scripts.evaluate --skip-eval-dataset
```

### Programmatic Usage

```python
from moellama import load_config, setup_device, run_benchmarks, generate_report

# Load config
config = load_config("config.hocon")
device = setup_device(config)

# Load model and tokenizer (your code)
model, tokenizer = ...

# Run benchmarks
results = run_benchmarks(model, tokenizer, device, config, eval_dataset)

# Generate report
generate_report(model, config, results, output_path="report.md")

# Access individual results
print(f"Perplexity: {results['perplexity']}")
print(f"Accuracy: {results['token_accuracy']}%")
```

## 📄 Generated Report

The report is a markdown file with these sections:

### 1. Header
- Report title
- Generation timestamp

### 2. Model Architecture
- Model type (MoE)
- Dimensions, layers, heads
- Expert configuration
- Parameter counts

### 3. Training Configuration
- Dataset used
- Batch size, learning rate
- Epochs, sequence length
- Optimizer details

### 4. Summary Table
Quick overview of all metrics:
```
| Metric | Score |
|--------|-------|
| Perplexity | 45.2 |
| Token Accuracy | 32.5% |
...
```

### 5. Benchmark Results
Detailed explanation of each metric with interpretation guidance.

### 6. Generation Samples
Sample outputs from the model with prompts.

## 🛠️ Adding Custom Benchmarks

### Step 1: Add Benchmark Method

Edit `moellama/benchmarks.py`:

```python
class BenchmarkSuite:
    # ... existing methods ...

    def evaluate_my_custom_task(self) -> float:
        """
        Your custom evaluation task.

        Returns:
            Score between 0 and 1
        """
        self.model.eval()

        # Your evaluation logic here
        test_cases = [...]
        correct = 0

        for test_case in test_cases:
            # Run model
            # Check if correct
            pass

        score = correct / len(test_cases)
        logger.info(f"My Custom Task: {score*100:.1f}%")
        return score
```

### Step 2: Add to run_all_benchmarks

```python
def run_all_benchmarks(self, eval_dataset=None):
    results = {}
    enabled = self.eval_config.get('enabled_benchmarks', [])

    # ... existing benchmarks ...

    if 'my_custom_task' in enabled:
        logger.info("Evaluating my custom task...")
        score = self.evaluate_my_custom_task()
        results['my_custom_task_score'] = score

    return results
```

### Step 3: Add to Config

```hocon
evaluation {
  enabled_benchmarks = [
    "perplexity",
    "my_custom_task"  # Add your benchmark
  ]
}
```

### Step 4: Update Report (Optional)

Edit `moellama/report.py` to format your benchmark results.

## 📈 Interpreting Results

### Perplexity

- **What to look for**: Lower is better
- **Typical values**: 20-100 for small models
- **If too high (>100)**: Model hasn't learned well, try:
  - More training epochs
  - Better dataset
  - Larger model

### Token Accuracy

- **What to look for**: Higher is better
- **Typical values**: 20-40% for small models
- **Context**: Even 30% accuracy is useful since the model has thousands of tokens to choose from

### Generation Quality

- **What to look for**: Coherence, grammar, relevance
- **Red flags**: Repetition, nonsense, unrelated text
- **Improvements**: More training, better dataset, temperature tuning

### Counting/Math

- **What to look for**: Any non-zero score is good for small models
- **Context**: These tasks are challenging even for large models
- **Purpose**: Tests reasoning ability, not just memorization

## 🔍 Debugging

### Issue: Evaluation is skipped

**Check**:
```hocon
evaluation {
  enabled = true  # Make sure this is true
}
```

### Issue: Perplexity/Accuracy not running

**Reason**: These require an evaluation dataset.

**Solution**:
```bash
# Don't use --skip-eval-dataset flag
python -m scripts.evaluate
```

### Issue: Out of memory during evaluation

**Solution**:
```hocon
evaluation {
  batch_size = 4  # Reduce from 16
}
```

### Issue: Evaluation is slow

**Solutions**:
1. Reduce enabled benchmarks
2. Use `--skip-eval-dataset` to skip perplexity/accuracy
3. Reduce test prompts count
4. Use GPU if available

## 📚 Best Practices

1. **Baseline First**: Run evaluation on your first trained model to establish baseline
2. **Track Over Time**: Save reports with timestamps to track improvements
3. **Multiple Prompts**: Use diverse test prompts to assess generalization
4. **Compare Models**: Evaluate different model sizes/configurations
5. **Dataset Matters**: Evaluation on same domain as training is easier
6. **Iterate**: Use results to guide training improvements

## 🎓 Example Workflow

```bash
# 1. Train model
python -m scripts.train

# 2. Evaluate
python -m scripts.evaluate --output report_v1.md

# 3. Review report, tune hyperparameters

# 4. Train again
python -m scripts.train

# 5. Evaluate again
python -m scripts.evaluate --output report_v2.md

# 6. Compare reports
diff report_v1.md report_v2.md
```

## 🔗 References

- **Perplexity**: Standard metric for language models
- **Token Accuracy**: Simple but informative metric
- **Counting Task**: Tests reasoning (inspired by GPT-4 evaluations)
- **Standard Benchmarks**:
  - [ARC](https://allenai.org/data/arc)
  - [MMLU](https://arxiv.org/abs/2009.03300)
  - [GSM8K](https://arxiv.org/abs/2110.14168)
  - [HumanEval](https://arxiv.org/abs/2107.03374)

## 🤝 Contributing Benchmarks

To contribute standard benchmarks (ARC, MMLU, etc.):

1. Implement the benchmark in `moellama/benchmarks.py`
2. Add configuration options
3. Update report formatting
4. Add documentation
5. Submit a pull request!

See the placeholder method `evaluate_standard_benchmark()` for the framework.
