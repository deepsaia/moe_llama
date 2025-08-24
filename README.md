# 🌟 moellama: Mixture of Experts Language Model

![MoE Architecture](https://i.imgur.com/0YkZJtL.png)

This project implements a **Mixture of Experts (MoE)** language model inspired by the LLaMA 4 architecture. Unlike traditional transformer models that use a single feed-forward network per layer, this implementation uses multiple expert networks with a router that selects which experts to activate for each token.

## ✨ Key Features

- **Character-level tokenization** with proper escaping for special characters
- **Rotary Positional Embeddings (RoPE)** for better positional understanding
- **RMS Normalization** for stable training
- **Mixture of Experts** with configurable number of experts and routing
- **Shared expert option** to ensure certain knowledge is always available
- **Multi-device support** (CPU, GPU, MPS for Apple Silicon)
- **Interactive inference mode** for real-time text generation
- **HOCON configuration** for easy setup and reproducibility
- **Load balancing loss** to prevent expert collapse during training

## 📋 Project Overview

LLaMA4MoE is a from-scratch implementation of the Mixture of Experts architecture used in LLaMA 4. Instead of processing every token through the same feed-forward network, MoE models route tokens to specialized "expert" networks, allowing for:
- Efficient scaling to larger parameter counts
- Better specialization for different token types
- Reduced computational cost during inference
- Improved model performance for the same computational budget

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda
- For GPU acceleration: CUDA-compatible NVIDIA GPU or Apple Silicon for MPS or even just a cpu

### Step-by-Step Setup

1. Clone the repository
```bash
git clone https://github.com/deepsai8/moe_llama.git
cd llama4-moe
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash 
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `config.hocon` file with the following structure: 

A sample hocon file is provided in this project.

```hocon
{
  # Model architecture configuration
  model {
    dim = 256         # Model dimension
    num_layers = 4    # Number of transformer layers
    num_heads = 8     # Number of attention heads
    num_experts = 8   # Number of experts in MoE layer
    top_k = 2         # Number of experts to select per token
    max_seq_len = 128 # Maximum sequence length
    dropout = 0.1     # Dropout rate
    shared_expert = true  # Include a shared expert
    load_balancing_loss_coef = 0.01  # Coefficient for load balancing loss
  }
  
  # Device configuration
  device {
    type = "auto"  # "auto", "cpu", "cuda", "mps"
    num_cpu_threads = -1  # Uses all but 2 cores (-1 = auto)
    num_cpu_interop_threads = 2  # Number of CPU interop threads
    gpu_ids = [0]  # GPUs to use (for DataParallel)
    use_mps = true  # Whether to use Apple's Metal Performance Shaders (for Macs)
  }
  
  # Training configuration
  training {
    batch_size = 16
    learning_rate = 3e-4
    epochs = 3
    eval_steps = 100
    dataset = "tiny_shakespeare"  # Can be changed to other datasets
    seq_len = 128
    data_dir = "data"  # Directory to store datasets
    num_workers = 4  # Number of data loading workers
  }
  
  # Inference configuration
  inference {
    max_new_tokens = 50
    temperature = 0.7
    top_k = null
    top_p = null
  }
  
  # Paths configuration
  paths {
    model_path = "./llama4_moe_model"
    output_dir = "./model"
  }
}
```

## 🚀 Training Your Model

### Basic Training
```bash
python -m moellama
```

This will:
- Download and prepare the dataset (if using tiny_shakespeare)
- Build the vocabulary
- Train the model for the specified number of epochs
- Save the model and tokenizer to the output directory
- Generate a sample text using the trained model

### Monitoring Training
Training logs are saved to `llama4_moe.log`. You can monitor training progress with:
```bash
tail -f llama4_moe.log
```

### Advanced Training Options

**Multi-GPU Training** (update config.hocon):
```hocon
device {
  type = "cuda"
  gpu_ids = [0, 1, 2, 3]
  use_data_parallel = true
}
```

**CPU Core Management** (for better system responsiveness):
```hocon
device {
  type = "cpu"
  num_cpu_threads = -1  # Uses all but 2 cores
  num_cpu_interop_threads = 2
}
```

**Using Different Datasets**:
```hocon
training {
  dataset = "wikitext"  # Or any Hugging Face dataset
}
```

## 💬 Inference with Your Trained Model

### Basic Inference
```bash
python -m infermoe
```

This will run a basic inference example with predefined prompts.

### Interactive Mode (Recommended)
```bash
python infermoe.py --im
```

This starts an interactive session where you can:
- Enter custom prompts
- Adjust generation parameters (temperature, top_k, top_p)
- See detailed generation statistics

Example interactive session:
```
Starting interactive inference session. Type 'exit' to quit.

Prompt: The future of AI is
Max new tokens (default 50): 100
Temperature (default 0.7): 0.85
Top-k (default None): 50
Top-p (default None): 0.9

Generating...

=== Generated Text ===
The future of AI is increasingly intertwined with human creativity and decision-making processes. As we continue to develop more sophisticated models, the line between human and machine intelligence becomes more blurred. This evolution is not just about technological advancement but also about how we, as a society, choose to integrate these tools into our daily lives. The ethical considerations surrounding AI development are becoming more prominent, with discussions about bias, privacy, and the potential for job displacement. Despite these challenges, the potential benefits of AI are immense, from healthcare advancements to climate change solutions. The key will be finding the right balance between innovation and responsibility.

=== New Completion ===
 increasingly intertwined with human creativity and decision-making processes. As we continue to develop more sophisticated models, the line between human and machine intelligence becomes more blurred. This evolution is not just about technological advancement but also about how we, as a society, choose to integrate these tools into our daily lives. The ethical considerations surrounding AI development are becoming more prominent, with discussions about bias, privacy, and the potential for job displacement. Despite these challenges, the potential benefits of AI are immense, from healthcare advancements to climate change solutions. The key will be finding the right balance between innovation and responsibility.

Stats: 118 total tokens, 5.89 tokens/sec
```

### One-off Prompt Generation
```bash
python infermoe.py --prompt "The future of AI is"
```

## 🔍 Understanding MoE Architecture

Unlike traditional transformer models that use a single feed-forward network per layer, LLaMA4MoE uses a Mixture of Experts approach:

1. **Router Network**: Determines which experts should process each token
2. **Expert Networks**: Specialized feed-forward networks that handle specific token types
3. **Load Balancing**: Ensures all experts get adequate training

When the model processes a token like "was", the router might select Expert 1 (70%) and Expert 3 (30%) while ignoring Experts 2 and 4. This selective activation makes the model more efficient while maintaining high performance.

## 🐞 Troubleshooting

### Common Issues & Solutions

**Vocabulary Size Mismatch**
```
size mismatch for token_embeddings.weight: copying a param with shape torch.Size([68, 256]) from checkpoint, the shape in current model is torch.Size([66, 256]).
```
- **Cause**: Tokenizer vocabulary doesn't match what was used during training
- **Solution**: Always use the same `vocab.txt` file from training for inference

**NoneType has no attribute 'backward'**
```
AttributeError: 'NoneType' object has no attribute 'backward'
```
- **Cause**: Loss is None because labels weren't provided during training
- **Solution**: Ensure you're providing labels: `outputs = self.model(input_ids, labels=input_ids, training=True)`

**Interactive Mode Errors**
```
too many values to unpack (expected 2)
```
- **Cause**: Mismatch in return values from `load_model_and_tokenizer`
- **Solution**: Make sure the function returns model, tokenizer, and device

## 🌱 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> "The future of AI isn't about replacing humans, but about creating tools that enhance our capabilities while respecting our values." - moellama Team