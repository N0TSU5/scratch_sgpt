import torch
from torch.utils.data import Dataset


class TextDataLoader(Dataset):
    """Loads text, tokenizes, splits into train/test, and provides batches."""

    def __init__(self, file_path, tokenizer, split_ratio=0.9, train=True):
        self.tokenizer = tokenizer

        # Load and tokenize text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized_text = self.tokenize(text)
        split_idx = int(len(tokenized_text) * split_ratio)

        # Split into train/test
        self.data = tokenized_text[:split_idx] if train else tokenized_text[split_idx:]

    def get_batch(self, batch_size=4, block_size=32, device="cuda"):
        """Returns a batch of (xb, yb) tensors."""

        # Sample random indices for the starting position of each sequence
        indices = torch.randint(len(self.data) - block_size, (batch_size,))

        # Get the sequences, x is the input and y is the target.
        # x is the same as y, but shifted one position to the left.
        x = torch.stack([self.data[i : i + block_size] for i in indices])
        y = torch.stack([self.data[i + 1 : i + block_size + 1] for i in indices])

        return x.to(device), y.to(device)
    
    def tokenize(self, text, max_length=1024):
        tokenized_chunks = []
        for i in range(0, len(text), max_length):
            chunk = text[i : i + max_length]
            tokenized_chunk = self.tokenizer.encode(chunk).squeeze(0).to(device="cuda")
            tokenized_chunks.append(tokenized_chunk)

        return torch.cat(tokenized_chunks)

