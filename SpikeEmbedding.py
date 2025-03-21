import torch.nn as nn


class SpikeEmbedding(nn.Module):
    """Convert input ids to spike embeddings."""

    def __init__(self, vocab_size, embedding_dim, device="cuda"):
        super().__init__()

        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)

    def heaviside(self, embeddings):
        """Heaviside step function: returns 1 for x >= 0, 0 for x < 0"""

        return (embeddings >= 0).float()

    def forward(self, input_ids):
        """Convert input ids to spike embeddings."""

        # Get the embeddings of the input ids.
        embeddings = self.embedding(input_ids)

        # Convert the embeddings to spike trains.
        spike_embeddings = self.heaviside(embeddings)

        return spike_embeddings
