# Dataset Guide

This guide explains how to use different datasets with moellama.

## 📁 Dataset Directory Structure

All datasets are stored in the `dataset/` directory by default:

```
moe_llama/
├── dataset/                    # All datasets go here (gitignored)
│   ├── tiny_shakespeare/       # Example dataset
│   │   └── tiny_shakespeare.txt
│   ├── wikitext/              # HuggingFace dataset (if downloaded)
│   └── your_custom_data/       # Your custom datasets
└── config.hocon
```

**Note**: The `dataset/` directory is gitignored, so your data won't be committed to version control.

## 🚀 Quick Start

### Using Tiny Shakespeare (Default)

The simplest option - just run training:

```bash
python -m scripts.train
```

This automatically downloads ~1MB of Shakespeare text to `dataset/tiny_shakespeare/`.

## 📚 Using HuggingFace Datasets

You can use any text dataset from HuggingFace by changing the config:

### Example: Wikitext

```hocon
# config.hocon
training {
  dataset = "wikitext"  # Change this line
  ...
}
```

Then run:
```bash
python -m scripts.train
```

### Popular Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `tiny_shakespeare` | ~1MB | Shakespeare's works (default) |
| `wikitext` | ~500MB | Wikipedia articles |
| `openwebtext` | ~38GB | Web pages from Reddit |
| `bookcorpus` | ~5GB | Books |
| `c4` | ~300GB | Common Crawl web data |

**Browse more**: https://huggingface.co/datasets?task_categories=text-generation

### Configuration

```hocon
training {
  dataset = "wikitext"           # Dataset name
  data_dir = "dataset"           # Where to cache downloaded data
  seq_len = 256                  # Sequence length for training
  ...
}
```

## 📝 Using Custom Text Data

### Option 1: Single Text File

1. **Create your dataset file**:
   ```bash
   mkdir -p dataset/my_dataset
   # Add your text to a .txt file
   echo "Your text here..." > dataset/my_dataset/data.txt
   ```

2. **Modify dataset.py** to load your file:

   ```python
   # moellama/dataset.py

   def load_custom_dataset(data_dir="dataset/my_dataset"):
       """Load custom text dataset."""
       file_path = Path(data_dir) / "data.txt"
       with open(file_path, 'r', encoding='utf-8') as f:
           text = f.read()
       return text
   ```

3. **Update prepare_dataset** function:

   ```python
   # In prepare_dataset() function, add:
   elif dataset_name == "my_dataset":
       text = load_custom_dataset(data_dir)
       # ... rest of the processing
   ```

4. **Update config**:
   ```hocon
   training {
     dataset = "my_dataset"
   }
   ```

### Option 2: Multiple Files

For multiple text files:

```python
# moellama/dataset.py

def load_custom_dataset_multiple(data_dir="dataset/my_dataset"):
    """Load multiple text files."""
    texts = []
    for file_path in Path(data_dir).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return " ".join(texts)
```

### Option 3: JSON/JSONL Data

For structured data (conversations, Q&A pairs):

```python
# moellama/dataset.py
import json

def load_jsonl_dataset(data_dir="dataset/my_jsonl"):
    """Load JSONL dataset with text field."""
    texts = []
    file_path = Path(data_dir) / "data.jsonl"

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Adjust field name based on your data structure
            texts.append(data['text'])

    return " ".join(texts)
```

Example JSONL format:
```jsonl
{"text": "First example text..."}
{"text": "Second example text..."}
{"text": "Third example text..."}
```

## 🔧 Advanced: Dataset Processing

### Custom Preprocessing

Add preprocessing to `prepare_dataset()`:

```python
def prepare_dataset(config, tokenizer=None):
    # ... load dataset as 'text' ...

    # Example: Clean text
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    # Example: Filter by length
    min_length = 100
    sentences = text.split('.')
    text = '. '.join([s for s in sentences if len(s) > min_length])

    # ... continue with train/eval split ...
```

### Data Augmentation

```python
# Example: Add data augmentation
def augment_text(text):
    """Simple text augmentation."""
    augmented = []

    # Original text
    augmented.append(text)

    # Shuffled sentences (for variety)
    sentences = text.split('.')
    random.shuffle(sentences)
    augmented.append('. '.join(sentences))

    return ' '.join(augmented)
```

## 📊 Dataset Statistics

To see statistics about your dataset:

```python
# Add to scripts/train.py or create a new script

from moellama import prepare_dataset, load_config

config = load_config("config.hocon")
train_dataset, eval_dataset, tokenizer = prepare_dataset(config)

print(f"Training sequences: {len(train_dataset)}")
print(f"Evaluation sequences: {len(eval_dataset)}")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Sequence length: {config['training']['seq_len']}")
```

## ⚠️ Important Notes

### Memory Considerations

Large datasets require more RAM:

- **Tiny Shakespeare**: ~5MB RAM
- **Wikitext**: ~2GB RAM
- **OpenWebText**: ~40GB+ RAM

If you have limited RAM:
1. Use smaller datasets
2. Process data in chunks
3. Use memory-mapped files
4. Stream data instead of loading all at once

### Tokenizer Training

The tokenizer is trained on your dataset. For best results:
- Use at least 1MB of text
- Include diverse examples
- For domain-specific data, train on that domain

### Train/Eval Split

By default, data is split 90/10 (train/eval). To change:

```python
# In prepare_dataset()
train_size = int(0.95 * len(text))  # 95/5 split
```

## 🌐 Remote Datasets

### From URL

```python
def download_from_url(url, data_dir="dataset/custom"):
    """Download dataset from URL."""
    import requests

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    filename = url.split('/')[-1]
    file_path = data_dir / filename

    if file_path.exists():
        print(f"Using cached: {file_path}")
    else:
        print(f"Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(response.content)

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
```

Usage:
```python
# In prepare_dataset()
elif dataset_name == "custom_url":
    url = "https://example.com/dataset.txt"
    text = download_from_url(url, data_dir)
```

## 🔍 Debugging

### Check Dataset Loading

```python
# Test dataset loading
from moellama import load_config, prepare_dataset

config = load_config("config.hocon")
train_ds, eval_ds, tok = prepare_dataset(config)

# Check a sample
sample = train_ds[0]
print(f"Sample shape: {sample.shape}")
print(f"Sample tokens: {sample[:10]}")
print(f"Decoded: {tok.decode(sample[:10].tolist())}")
```

### Verify Data Directory

```bash
# Check what's in your dataset directory
ls -lh dataset/
ls -lh dataset/tiny_shakespeare/

# Check file sizes
du -sh dataset/*
```

## 📦 Example: Complete Custom Dataset

Here's a complete example for adding a custom dataset:

### 1. Add Data

```bash
mkdir -p dataset/my_books
cat > dataset/my_books/book1.txt << EOF
Once upon a time in a land far away...
(your text here)
EOF
```

### 2. Add Loader Function

```python
# moellama/dataset.py

def load_my_books(data_dir="dataset/my_books"):
    """Load my custom books dataset."""
    texts = []
    for txt_file in Path(data_dir).glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return "\n\n".join(texts)
```

### 3. Update prepare_dataset

```python
# In prepare_dataset(), add this branch:
elif dataset_name == "my_books":
    text = load_my_books(data_dir)
    train_size = int(0.9 * len(text))
    train_text = text[:train_size]
    eval_text = text[train_size:]

    # ... rest of processing (same as tiny_shakespeare)
```

### 4. Update Config

```hocon
training {
  dataset = "my_books"
  data_dir = "dataset"
}
```

### 5. Train

```bash
python -m scripts.train
```

## 🎯 Best Practices

1. **Start Small**: Test with tiny_shakespeare before using large datasets
2. **Check Quality**: Inspect your data for errors, encoding issues
3. **Monitor Training**: Watch loss curves - bad data shows up as unstable loss
4. **Version Control**: Keep track of which dataset version you used
5. **Document**: Add comments about data sources and preprocessing

## 🆘 Common Issues

**Issue**: `FileNotFoundError: dataset not found`
- **Solution**: Check `data_dir` path in config.hocon

**Issue**: `UnicodeDecodeError`
- **Solution**: Specify encoding: `open(file, 'r', encoding='utf-8')`

**Issue**: `MemoryError` with large datasets
- **Solution**: Use smaller batch size or process data in chunks

**Issue**: Slow training with HuggingFace datasets
- **Solution**: First run downloads data; subsequent runs use cache

## 📚 References

- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [Text Data Best Practices](https://huggingface.co/docs/datasets/about_dataset_load)
- [Tokenizer Training Guide](https://huggingface.co/docs/tokenizers/training_from_memory)
