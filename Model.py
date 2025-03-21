import torch
import torch.nn as nn
import torch.nn.functional as F
from Block import Block
from SpikeEmbedding import SpikeEmbedding
from TokenizerWrapper import TokenizerWrapper


class Model(nn.Module):
    """Spike-based resursive language model."""

    def __init__(self, vocab_size, embedding_dim, num_blocks, device="cuda"):
        super().__init__()

        self.embedding = SpikeEmbedding(vocab_size, embedding_dim).to(device)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, device) for _ in range(num_blocks)]
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size).to(device)

        self.device = device

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids).to(self.device)
        embeddings = self.blocks(embeddings)
        logits = self.lm_head(embeddings)

        return logits

    def generate(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the logits for the next token.
            logits = self(input_ids)
            logits = logits[:, -1, :]

            # Apply softmax to compute probabilities.
            probs = F.softmax(logits, dim=-1) 
            
            # Sample from the distribution.
            idx_next = torch.multinomial(
                probs, num_samples=1
            ) 

            input_ids = torch.cat([input_ids, idx_next], dim=-1)

        return input_ids
