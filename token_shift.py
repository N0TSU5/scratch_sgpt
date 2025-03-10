import torch
import torch.nn as nn


class TokenShift(nn.Module):
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): Dimension of the embedding space
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.shift_mask = nn.Parameter(torch.randn(embedding_dim)).cuda()
        self.register_buffer(
            "indices", torch.arange(1, embedding_dim + 1).float().cuda()
        )

    def compute_shift_weights(self):
        """
        Compute shift weights based on embedding dimension

        Returns:
            torch.Tensor: Learnable shift weights
        """
        # Compute base shift weights: i/E
        base_weights = self.indices / self.embedding_dim

        # Apply learnable mask through sigmoid
        learnable_weights = torch.sigmoid(self.shift_mask)

        return learnable_weights * base_weights

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor: Input with token shifting applied
        """
        batch_size, _, embedding_dim = x.shape

        # Compute shift weights
        shift_weights = self.compute_shift_weights()

        # Zero Padding: Shift tokens down by 1 and pad first row with zeros
        x_padded = torch.cat(
            [torch.zeros(batch_size, 1, embedding_dim, device=x.device), x[:, :-1, :]],
            dim=1,
        )

        # Apply weights (broadcasting over batch and sequence length)
        output = (
            shift_weights.view(1, 1, -1) * x
            + (1 - shift_weights.view(1, 1, -1)) * x_padded
        )

        return output
