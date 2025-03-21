import torch.nn as nn
from FeedForward import FeedForward
from Head import Head


class Block(nn.Module):
    """Transformer block. Consists of a head and a feedforward layer."""

    def __init__(self, embedding_dim, device="cuda"):
        super().__init__()

        self.head = Head(embedding_dim, device)
        self.feed_forward = FeedForward(embedding_dim, device)

        self.device = device

    def forward(self, x):
        x = self.head(x)
        x = self.feed_forward(x)
        
        return x.to(self.device)
