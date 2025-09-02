import logging


logger = logging.getLogger("")

class CharacterTokenizer:
    """Character-level tokenizer as mentioned in the knowledge base"""
    
    def __init__(self, text=None, vocab_file=None):
        if vocab_file:
            self.load_vocab(vocab_file)
            logger.info(f"Loaded vocabulary from {vocab_file}, size: {self.vocab_size}")
        elif text:
            self.build_vocab(text)
            logger.info(f"Built vocabulary from text, size: {self.vocab_size}")
        else:
            raise ValueError("Either text or vocab_file must be provided")
    
    def build_vocab(self, text):
        """Build vocabulary from text"""
        logger.debug("Building vocabulary from text")
        # Get all unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mapping from characters to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        # Add special tokens to vocabulary
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token
                
        self.vocab_size = len(self.stoi)
        self.pad_token_id = self.stoi[self.pad_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.unk_token_id = self.stoi[self.unk_token]
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.stoi.get(ch, self.unk_token_id) for ch in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        return ''.join([self.itos.get(idx, self.unk_token) for idx in token_ids])
    
    def __len__(self):
        return self.vocab_size
    
    def save_vocab(self, file_path):
        """Save vocabulary to file with proper escaping"""
        with open(file_path, 'w', encoding='utf-8') as f:
            # Sort by index to ensure consistent ordering
            for idx in sorted(self.itos.keys()):
                char = self.itos[idx]
                
                # Escape special characters for reliable parsing
                escaped_char = char
                if char == '\t':
                    escaped_char = '\\t'
                elif char == '\n':
                    escaped_char = '\\n'
                elif char == '\r':
                    escaped_char = '\\r'
                elif char == '\\':
                    escaped_char = '\\\\'
                
                f.write(f"{escaped_char}\t{idx}\n")
        logger.info(f"Vocabulary saved to {file_path}")
    
    def load_vocab(self, file_path):
        """Load vocabulary from file with proper unescaping"""
        logger.info(f"Loading vocabulary from {file_path}")
        self.stoi = {}
        self.itos = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.rstrip('\n\r')  # Only strip line endings, not spaces/tabs!
                    if not line:
                        continue

                    # Split on the FIRST tab only
                    if '\t' not in line:
                        logger.warning(f"Skipping invalid line {line_number} (no tab): '{line}'")
                        continue

                    first_tab_idx = line.find('\t')
                    escaped_char = line[:first_tab_idx]
                    idx_str = line[first_tab_idx + 1:]

                    # Unescape special sequences
                    char = escaped_char
                    if char == '\\t':
                        char = '\t'
                    elif char == '\\n':
                        char = '\n'
                    elif char == '\\r':
                        char = '\r'
                    elif char == '\\\\':
                        char = '\\'
                    # No special handling for empty string — it's valid as ''
                    # So '' remains as ''

                    try:
                        idx = int(idx_str)
                    except ValueError:
                        logger.warning(f"Invalid index on line {line_number}: '{idx_str}'")
                        continue

                    # Avoid duplicates
                    if idx in self.itos:
                        logger.warning(f"Duplicate index {idx} on line {line_number}")
                    if char in self.stoi:
                        logger.warning(f"Duplicate character '{char}' on line {line_number}")

                    self.stoi[char] = idx
                    self.itos[idx] = char

            if len(self.stoi) == 0:
                raise ValueError("No valid vocabulary entries found in file")

            self.vocab_size = len(self.stoi)
            self.pad_token = '<pad>'
            self.eos_token = '<eos>'
            self.unk_token = '<unk>'

            # Validate and set token IDs
            missing_tokens = []
            if self.pad_token in self.stoi:
                self.pad_token_id = self.stoi[self.pad_token]
            else:
                missing_tokens.append(f"'{self.pad_token}'")

            if self.eos_token in self.stoi:
                self.eos_token_id = self.stoi[self.eos_token]
            else:
                missing_tokens.append(f"'{self.eos_token}'")

            if self.unk_token in self.stoi:
                self.unk_token_id = self.stoi[self.unk_token]
            else:
                missing_tokens.append(f"'{self.unk_token}'")

            if missing_tokens:
                raise RuntimeError(f"Missing required special tokens in vocab: {', '.join(missing_tokens)}")

            logger.info(f"Successfully loaded vocabulary with {self.vocab_size} tokens")
        except FileNotFoundError:
            logger.error(f"Vocabulary file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading vocabulary from {file_path}: {str(e)}")
            raise