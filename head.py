import torch
import torch.nn as nn
from torch import exp
from token_shift import TokenShift
import snntorch as snn


class Head(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        # Initialize token shift
        self.token_shift = TokenShift(embedding_dim=embedding_dim)

        # Initialize matrices
        self.R = nn.Linear(embedding_dim, embedding_dim).cuda()  # Receptance
        self.K = nn.Linear(embedding_dim, embedding_dim).cuda()  # Key
        self.V = nn.Linear(embedding_dim, embedding_dim).cuda()  # Value

        # Initialize decay parameters
        self.token_decay = nn.Linear(
            embedding_dim, embedding_dim
        ).cuda()  # Token-level decay
        self.global_decay = nn.Parameter(
            torch.randn(embedding_dim)
        ).cuda()  # Global decay

        # Initialize spiking layer
        self.spiking_layer = snn.Leaky(beta=0.3, threshold=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = x.shape

        # Initialize hidden states (persists across timesteps)
        state_a = torch.zeros(batch_size, embedding_dim).cuda()
        state_b = torch.zeros(batch_size, embedding_dim).cuda()

        # Project input through matrices
        r = self.R(x)  # Receptance
        k = self.K(x)  # Key
        v = self.V(x)  # Value

        # Apply sigmoid to receptance
        r = torch.sigmoid(r)

        # Output tensor storage
        new_embeddings = torch.zeros_like(x)

        # Initialize membrane potential for spiking neurons
        mem = torch.zeros(
            batch_size, embedding_dim
        ).cuda()  # Memory state for spiking layer

        # Process sequence
        for t in range(seq_len):
            # Apply token shift
            x = self.token_shift(x)

            # Compute decays for current token
            token_decay_t = exp(self.token_decay(x[:, t, :]))
            global_decay_exp = exp(self.global_decay)

            # Update hidden states (accumulate over time)
            state_a = exp(k[:, t, :]) * v[:, t, :] + token_decay_t * state_a
            state_b = exp(k[:, t, :]) + token_decay_t * state_b

            # Compute normalized output (WKV)
            time_decayed_kv = (
                global_decay_exp * exp(k[:, t, :]) * v[:, t, :] + state_a
            ) / (global_decay_exp * exp(k[:, t, :]) + state_b + 1e-6)

            time_decayed_embeddings = r[:, t, :] * time_decayed_kv

            # Apply spiking layer
            spk, mem = self.spiking_layer(time_decayed_embeddings, mem)

            # Store spiking output
            new_embeddings[:, t, :] = spk

        return new_embeddings
