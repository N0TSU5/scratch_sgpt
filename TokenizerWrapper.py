import torch.nn as nn
from transformers import AutoTokenizer


class TokenizerWrapper(nn.Module):
    """Wrapper for AutoTokenizer class. Tokenizes a batch of texts."""

    def __init__(self, model_name="gpt2", device="cuda"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.device = device

    def encode(self, texts):
        """Tokenize a batch of texts. Returns a tensor of token ids."""

        encoded_tokens = self.tokenizer(texts, padding=True, return_tensors="pt")
        return encoded_tokens["input_ids"].to(self.device)

    def decode(self, token_ids):
        """Decode a tensor of token ids. Returns a list of texts."""

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
