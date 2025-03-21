import torch
import torch.nn as nn


class TokenShift(nn.Module):
    """Layer which shifts tokens in the embedding space."""

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.shift_mask = nn.Parameter(torch.randn(embedding_dim)).cuda()

        self.register_buffer(
            "indices", torch.arange(1, embedding_dim + 1).float().cuda()
        )

    def compute_shift_weights(self):
        # Compute base shift weights: i/E.
        base_weights = self.indices / self.embedding_dim

        # Compute learnable shift weights using the learnable shift mask.
        learnable_weights = torch.sigmoid(self.shift_mask)

        return learnable_weights * base_weights

    def forward(self, x):
        batch_size, _, embedding_dim = x.shape

        # Zero Padding: Shift tokens down by 1 and pad first row with zeros.
        x_padded = torch.cat(
            [torch.zeros(batch_size, 1, embedding_dim, device=x.device), x[:, :-1, :]],
            dim=1,
        )

        # Apply the shift weights to the input tensor.
        shift_weights = self.compute_shift_weights()
        output = (
            shift_weights.view(1, 1, -1) * x
            + (1 - shift_weights.view(1, 1, -1)) * x_padded
        )

        return output
