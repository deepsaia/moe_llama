# 🌟 moellama: Mixture of Experts Language Model

<div align="center">
  <img src="icon.png" alt="MoE Architecture" width="20%"/>
</div>

A clean, modular, educational implementation of the **Mixture of Experts (MoE)** architecture. This project provides a full-stack implementation of MoE from scratch, designed for learning and experimentation.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda
- For GPU: CUDA-compatible NVIDIA GPU, or Apple Silicon for MPS, or just CPU

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/deepsaia/moe_llama.git
cd moe_llama
```

2. **Create virtual environment**
```bash
uv venv
```

3. **Install dependencies**
```bash
uv sync
source .venv/bin/activate
```

## ✨ Key Features

- **Modular Architecture** - Clean separation of concerns with distinct modules
- **Mixture of Experts** - Sparse expert activation for efficient scaling
- **Rotary Positional Embeddings (RoPE)** - Better positional understanding
- **RMS Normalization** - Stable training without mean centering
- **Load Balancing** - Prevents expert collapse during training
- **Shared Expert Option** - Ensures certain knowledge is always available
- **Multi-Device Support** - Works on CPU, CUDA (NVIDIA), MPS (Apple Silicon)
- **Interactive Inference** - Real-time text generation with parameter tuning
- **HOCON Configuration** - Easy, readable configuration files
- **Comprehensive Documentation** - Detailed docstrings and inline comments

## 📋 What is Mixture of Experts?

Unlike traditional transformers that route every token through the same feed-forward network, MoE models use multiple "expert" networks. A router network decides which experts should process each token, enabling:

- **Efficient Scaling** - More parameters without proportional compute cost
- **Specialization** - Different experts learn different patterns
- **Sparse Activation** - Only top-k experts process each token
- **Better Performance** - Match dense models with less computation

**Example**: With 8 experts and `top_k=2`, each token is processed by only 2 experts, but different tokens may select different experts based on their content.

## 📁 Project Structure

### Module Overview

#### Core Components

- **`model.py`** - The complete LLaMA4MoE model
  - Token embeddings
  - Transformer blocks with MoE
  - Language modeling head
  - Generation with sampling strategies

- **`moe.py`** - Mixture of Experts implementation
  - `Expert`: Individual feed-forward networks
  - `Router`: Selects which experts to activate
  - `MoELayer`: Combines routing and expert processing

- **`attention.py`** - Multi-head self-attention
  - Query, Key, Value projections
  - Causal masking for autoregressive generation
  - Optional rotary positional embeddings

- **`layers.py`** - Fundamental building blocks
  - `RMSNorm`: Root Mean Square normalization
  - `RotaryPositionalEmbeddings`: RoPE implementation
  - `TransformerBlock`: Attention + MoE with residual connections

- **`trainer.py`** - Training infrastructure
  - Mixed precision training (AMP)
  - Multi-GPU support (DataParallel)
  - Gradient clipping and optimization
  - Checkpointing and evaluation

- **`dataset.py`** - Data utilities
  - `TextDataset`: Tokenized sequences
  - `prepare_dataset`: Download and prepare data
  - Support for Tiny Shakespeare and HuggingFace datasets

- **`tokenizer.py`** - BPE tokenization
  - Training BPE from text
  - Encoding/decoding
  - Special token handling

- **`utils.py`** - Helper functions
  - Configuration loading (HOCON)
  - Device setup (CPU/CUDA/MPS)
  - Model inspection utilities

- **`benchmarks.py`** - Evaluation suite
  - Perplexity and accuracy metrics
  - Generation quality tests
  - Counting and math benchmarks
  - Framework for standard benchmarks (ARC, MMLU, etc.)

- **`report.py`** - Report generation
  - Markdown report formatting
  - Model and training information
  - Benchmark visualization
  - Customizable sections

## ⚙️ Configuration

Configuration is managed through `config.hocon` (Human-Optimized Config Object Notation):

```hocon
{
  # Model architecture
  model {
    dim = 256              # Model dimension
    num_layers = 4         # Transformer layers
    num_heads = 8          # Attention heads
    num_experts = 8        # Number of experts
    top_k = 2              # Experts activated per token
    max_seq_len = 256      # Maximum sequence length
    dropout = 0.1          # Dropout rate
    shared_expert = true   # Include shared expert
    load_balancing_loss_coef = 0.01  # Load balancing weight
  }

  # Device configuration
  device {
    type = "auto"          # "auto", "cpu", "cuda", "mps"
    num_cpu_threads = -1   # -1 = use all but 2 cores
    gpu_ids = [0]          # GPUs for DataParallel
    use_mps = false        # Use Apple Metal
  }

  # Training configuration
  training {
    batch_size = 16
    learning_rate = 3e-4
    epochs = 3
    eval_steps = 100
    dataset = "tiny_shakespeare"
    seq_len = 256
    num_workers = 4
  }

  # Inference configuration
  inference {
    max_new_tokens = 200
    temperature = 0.8
    top_k = 50
    top_p = 0.95
  }

  # Paths
  paths {
    model_path = "./trained_models"
    output_dir = "./model"
  }
}
```

## 📊 Working with Datasets

### Default Dataset: Tiny Shakespeare

The default configuration uses **Tiny Shakespeare** (~1MB), which automatically downloads to `dataset/`:

```bash
python -m scripts.train  # Downloads and caches to dataset/tiny_shakespeare/
```

### Using HuggingFace Datasets

Use any text dataset from HuggingFace by editing `config.hocon`:

```hocon
training {
  dataset = "wikitext"     # Change to any HF dataset
  data_dir = "dataset"     # Cached here (gitignored)
  ...
}
```

**Popular Options:**
- `tiny_shakespeare` - 1MB, Shakespeare (default)
- `wikitext` - 500MB, Wikipedia articles
- `openwebtext` - 38GB, web pages from Reddit
- `bookcorpus` - 5GB, books
- Browse more: https://huggingface.co/datasets?task_categories=text-generation

### Using Custom Datasets

**Quick Start** - Place your text files in `dataset/`:

```bash
mkdir -p dataset/my_data
echo "Your training text..." > dataset/my_data/data.txt
```

Then modify `config.hocon`:
```hocon
training {
  dataset = "my_data"  # Add custom loader in dataset.py
}
```

**Complete Guide**: See [DATASETS.md](DATASETS.md) for:
- Loading custom text files
- Using JSON/JSONL data
- Processing multiple files
- Preprocessing and augmentation
- Memory management for large datasets

### Dataset Directory Structure

```
dataset/                    # All datasets (gitignored)
├── tiny_shakespeare/       # Default dataset
├── wikitext/              # HuggingFace datasets
└── my_custom_data/        # Your custom data
```

## 🚀 Usage

### Training a Model

Train a new model from scratch:

```bash
python -m scripts.train
```

With custom configuration:

```bash
python -m scripts.train --config custom_config.hocon
```

The training script will:
1. Download the dataset (Tiny Shakespeare by default to `dataset/`)
2. Train the BPE tokenizer on your dataset
3. Create and train the MoE model
4. Save checkpoints periodically
5. Generate sample text to verify the model
6. Create training history plots

**Training logs** are saved to `logs/train.log`.

### Batch Inference

Generate text from prompts:

```bash
# Single prompt
python -m scripts.inference --prompt "The future of AI is"

# Multiple prompts from file
python -m scripts.inference --prompts-file prompts.txt

# Custom parameters
python -m scripts.inference \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.9 \
  --top-k 40 \
  --top-p 0.95
```

### Interactive Chat

**Terminal Interface:**

```bash
python -m scripts.interactive

# With verbose statistics
python -m scripts.interactive --verbose
```

**Interactive commands:**
- Type your prompt and press Enter to generate
- Type `params` to adjust generation parameters
- Type `exit` or `quit` to quit
- Press Ctrl+C to interrupt generation

**Web Interface:**

Launch the modern web-based chat UI with streaming responses and thread management. Built with **React**, **@assistant-ui/react**, **Tailwind CSS**, and **FastAPI**.

**First Time Setup:**
```bash
# Install frontend dependencies (using yarn)
cd frontend
yarn install

# Build the frontend
yarn build

# Return to project root
cd ..
```

Or use the build script from project root:
```bash
bash scripts/build_frontend.sh
```

This builds the React app to `prebuilt_frontend/dist/` which the FastAPI server will serve.

**Start the Server:**
```bash
# Start the chat server
python -m scripts.chat_server

# Open browser to http://localhost:8000
```

**Features:**
- 🌊 **Streaming responses** - Real-time text generation with Server-Sent Events
- 💬 **Thread management** - Create, switch between, and manage multiple conversation threads
- 🔄 **Model selection** - Switch between trained checkpoints on the fly
- 💾 **Message persistence** - All conversations saved to localStorage for later access
- 🌓 **Dark/Light theme** - Toggle between themes with smooth transitions
- 🎨 **Modern UI** - Clean, responsive interface built with Tailwind CSS and @assistant-ui/react primitives
- ⚙️ **Generation controls** - Adjust temperature, max tokens, top-k, top-p
- 📝 **Full message history** - View complete conversation history when switching threads
- 🗑️ **Thread deletion** - Remove unwanted conversations
- 🚀 **FastAPI backend** - Production-ready async API with streaming support
- 📱 **Responsive design** - Works seamlessly on desktop and mobile

**Development Mode:**

For frontend development with hot reload:
```bash
# Terminal 1: Start backend in dev mode
python -m scripts.chat_server --dev-mode

# Terminal 2: Start frontend dev server
cd frontend
yarn dev
# Opens at http://localhost:5173
```

**Custom Server Options:**
```bash
# Custom port
python -m scripts.chat_server --port 8080

# Bind to all interfaces (remote access)
python -m scripts.chat_server --host 0.0.0.0
```

📖 **Complete guides:**
- [docs/chat_ui.md](docs/chat_ui.md) - Backend API documentation, deployment
- [frontend/README.md](frontend/README.md) - Frontend development, customization

---

**Sample Interaction** with a model trained on tiny_shakespeare dataset:

```markdown
Prompt: what say you

Generating...

================================================================================
Generated Text:
--------------------------------------------------------------------------------
what say you at : I shall be the night . QUEEN MARGARET : O , the duke , I have be no too . You ' s a man , we have I have thee . LADY LADY CAPULET : The father , I , I speak , my son . Second First Murderer : O , he shall I have your blood : What have not ' s a a man ' d to me , That have been the time . LUCIO : O , this . DUKE VINCENTIO : I I am this . I ' s the heart ! DUKE VINCENTIO : Ay , my brother , let me , let you am have the other s life : I ' Tis ' s , I will be be your body . HENRY BOLINGBROKE : Why , so ? O , they ' ll see ' s with me . Why , and me , and I have been it will ; And so I think thou do my good your queen : And that you I know with a a man : And not , and a son ' d , let : I I am not
--------------------------------------------------------------------------------

Stats: 204 tokens, 27.13 tokens/sec
================================================================================

Prompt: exit
Goodbye!
```

Hooray! the response looks as good as the model is and as good as the data it's trained upon.

---

## 📈 Model Evaluation & Benchmarking

Evaluate your trained model with comprehensive benchmarks and generate a detailed report.

### Running Evaluation

```bash
# Evaluate the latest trained model
python -m scripts.evaluate

# Evaluate a specific checkpoint
python -m scripts.evaluate --model-file path/to/model.pt

# Custom output location
python -m scripts.evaluate --output my_report.md
```

### Available Benchmarks

The evaluation suite includes:

**Implemented Benchmarks:**
- **Perplexity** - How well the model predicts next tokens (lower is better)
- **Token Accuracy** - Percentage of correct predictions
- **Generation Quality** - Sample text generation with custom prompts
- **Counting Ability** - Letter counting tasks (e.g., "How many 'r's in 'strawberry'?")
- **Simple Math** - Basic arithmetic (single-digit addition/subtraction)

**Future Benchmarks** (placeholders for implementation):
- ARC-Easy & ARC-Challenge - AI2 Reasoning Challenge
- MMLU - Massive Multitask Language Understanding
- GSM8K - Grade School Math problems
- HumanEval - Code generation evaluation
- ChatCORE - Chat-oriented reasoning

### Configuration

Control evaluation via `config.hocon`:

```hocon
evaluation {
  enabled = true  # Enable/disable evaluation

  # Select which benchmarks to run
  enabled_benchmarks = [
    "perplexity",
    "accuracy",
    "generation",
    "counting",
    "simple_math"
  ]

  # Custom test prompts for generation
  test_prompts = [
    "Once upon a time",
    "The future of AI is",
    "In a distant land"
  ]

  # Report output location
  report_path = "report.md"
}
```

### Generated Report

The evaluation generates a markdown report including:

- **Model Architecture** - Parameter counts, configuration
- **Training Details** - Dataset, hyperparameters, optimizer
- **Benchmark Results** - Detailed scores with explanations
- **Generation Samples** - Example outputs from the model
- **Summary Table** - Quick overview of all metrics

**Example report structure:**
```
# Model Evaluation Report

## Model Architecture
| Component | Value |
|-----------|-------|
| Model Type | Mixture of Experts |
| Total Parameters | 2.5M |
...

## Summary
| Metric | Score |
|--------|-------|
| Perplexity | 45.2 |
| Token Accuracy | 32.5% |
...

## Generation Samples
...
```

### Programmatic Usage

You can also run evaluations programmatically:

```python
from moellama import run_benchmarks, generate_report

# Run benchmarks
results = run_benchmarks(model, tokenizer, device, config, eval_dataset)

# Generate report
generate_report(model, config, results, output_path="report.md")
```

### Adding Custom Benchmarks

To add your own benchmark:

1. Add a method to `BenchmarkSuite` in `moellama/benchmarks.py`
2. Add the benchmark name to `enabled_benchmarks` in config
3. Update report generation if needed

See [moellama/benchmarks.py](moellama/benchmarks.py) for examples.

---

## 📊 Model Architecture Details

### Transformer Block

Each transformer block consists of:

```
Input
  ↓
RMSNorm → Multi-Head Attention (with RoPE) → Add & Norm
  ↓
RMSNorm → Mixture of Experts → Add & Norm
  ↓
Output
```

### Mixture of Experts Layer

```
Input tokens
  ↓
Router (learns which experts to use)
  ↓
Top-k Expert Selection
  ↓
Expert 1    Expert 2    ...    Expert N    [Shared Expert]
  ↓           ↓                    ↓              ↓
Weighted combination of expert outputs
  ↓
Output
```

**Key mechanisms:**
- **Router**: Linear layer + softmax to select experts
- **Load Balancing Loss**: Encourages even expert usage
- **Noise during training**: Prevents expert collapse
- **Shared Expert**: Optional expert that always processes all tokens

### Attention Mechanism

Multi-head attention with:
- **RoPE (Rotary Position Embeddings)**: Encodes position into Q and K
- **Causal Masking**: Prevents attending to future tokens
- **Multi-head**: Parallel attention with different learned projections

## 🔍 Understanding the Code

### Training Flow

1. **Configuration** - Load from `config.hocon`
2. **Device Setup** - Detect and configure compute device
3. **Data Preparation** - Download, tokenize, create datasets
4. **Model Creation** - Initialize LLaMA4MoE with config parameters
5. **Training Loop**:
   - Forward pass through model
   - Compute loss (cross-entropy + load balancing)
   - Backward pass and optimization
   - Periodic evaluation and checkpointing
6. **Saving** - Save final model and tokenizer

### Generation Flow

1. **Load Model** - Load checkpoint and tokenizer
2. **Encode Prompt** - Convert text to token IDs
3. **Autoregressive Generation**:
   - Feed tokens through model
   - Get logits for next token
   - Apply sampling (temperature, top-k, top-p)
   - Sample next token
   - Append and repeat
4. **Decode** - Convert token IDs back to text

## 🐞 Troubleshooting

### Common Issues

**Vocabulary Size Mismatch**
```
size mismatch for token_embeddings.weight
```
**Solution**: Use the same vocab file from training for inference.

**Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `batch_size` in config.hocon

**Import Errors**
```
ModuleNotFoundError: No module named 'moellama'
```
**Solution**: Run scripts as modules: `python -m scripts.train`

**Slow CPU Training**
**Solution**:
- Set `num_cpu_threads = -1` in config to use more cores
- Consider using a GPU or reducing model size

### Performance Tips

- **GPU**: Use CUDA for 10-100x speedup
- **Mixed Precision**: Automatically enabled on CUDA (AMP)
- **Batch Size**: Increase if you have memory
- **Multi-GPU**: Set `use_data_parallel = true` and `gpu_ids = [0, 1, ...]`
- **CPU Threads**: Set `num_cpu_threads = -1` to use all but 2 cores

## 📈 Example Results

After training on Tiny Shakespeare:

```
Prompt: To be or not to be
Generated: To be or not to be,
That is the question that makes me wonder,
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...

Stats: 45 tokens, 8.2 tokens/sec
```

## 🤝 Contributing

Contributions are welcome! This project focuses on:
- **Clarity** - Code should be easy to understand
- **Modularity** - Components should be independent
- **Documentation** - Every function should have clear docstrings
- **Educational Value** - Prioritize learning over performance

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Normalization
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Sparse MoE at scale
- [GShard](https://arxiv.org/abs/2006.16668) - Scaling with MoE
- [LLaMA](https://arxiv.org/abs/2302.13971) - Efficient large language models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the LLaMA architecture from Meta AI
- Built for educational purposes to understand MoE architectures
- Thanks to the PyTorch and HuggingFace communities

---

> "The future of AI isn't about replacing humans, but about creating tools that enhance our capabilities while respecting our values." - moellama Team

**Happy Learning! 🚀**
