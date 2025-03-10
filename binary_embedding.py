import torch
import torch.nn as nn
from transformers import AutoTokenizer

class SubwordBinaryEmbedding(nn.Module):
    def __init__(self, embedding_dim, model_name="gpt2", device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_dim).to(device)

    def tokenize(self, texts):
        """Tokenize a batch of texts."""
        encoded = self.tokenizer(texts, padding=True, return_tensors='pt')
        return encoded['input_ids'].to(self.device)

    def heaviside(self, embeddings):
        """Heaviside step function: returns 1 for x >= 0, 0 for x < 0"""
        return (embeddings >= 0).float()

    def forward(self, texts):
        # Handle batch of texts
        if isinstance(texts, str):
            texts = [texts]
        
        # Get the sub-word token ids of the input texts
        token_ids = self.tokenize(texts)  # shape: [batch_size, sequence_length]
    
        # Get the embeddings of the token ids
        embeddings = self.embedding(token_ids)  # shape: [batch_size, sequence_length, embedding_dim] 

        # Convert the embeddings to binary
        binary_embeddings = self.heaviside(embeddings)
        
        return binary_embeddings 
