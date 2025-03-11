import torch
import torch.nn as nn
import torch.nn.functional as F
from token_shift import TokenShift
import snntorch as snn


class Head(nn.Module):
    """A single head of the recurrent spiking state space model"""
    def __init__(self, embedding_dim: int, device="cuda") -> None:
        super().__init__()

        self.device = device

        # Initialize token shift
        self.token_shift = TokenShift(embedding_dim=embedding_dim).to(device)

        # Initialize matrices
        self.R = nn.Linear(embedding_dim, embedding_dim, bias=False).to(
            device
        )  # Receptance
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)  # Key
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)  # Value

        # Initialize decay parameters
        self.token_decay = nn.Linear(embedding_dim, embedding_dim, bias=False).to(
            device
        )
        self.global_decay = nn.Parameter(
            torch.randn(embedding_dim, device=device)
        )  # Global decay

        # Initialize spiking layer
        self.spiking_layer = snn.Leaky(beta=0.3, threshold=0.2).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = x.shape

        # Apply token shift before processing sequence
        x = self.token_shift(x)

        # Initialize hidden states
        state_a = torch.zeros(batch_size, embedding_dim, device=self.device)
        state_b = torch.zeros(batch_size, embedding_dim, device=self.device)

        # Project input through matrices
        r = torch.sigmoid(self.R(x))  # Receptance
        k = self.K(x)  # Key
        v = self.V(x)  # Value

        # Initialize output tensor
        new_embeddings = torch.zeros_like(x, device=self.device)

        # Initialize membrane potential for spiking neurons
        mem = torch.zeros(batch_size, embedding_dim, device=self.device)

        # Process sequence
        for t in range(seq_len):
            # Compute decays for current token
            token_decay_t = torch.exp(self.token_decay(x[:, t, :]))  # Token-level decay
            global_decay_exp = torch.exp(self.global_decay)  # Global decay

            # Prevent numerical instability in exponentials
            exp_k = torch.exp(torch.clamp(k[:, t, :], max=10))

            # Update hidden states (accumulate over time)
            state_a = exp_k * v[:, t, :] + token_decay_t * state_a
            state_b = exp_k + token_decay_t * state_b

            # Compute normalized output (WKV)
            time_decayed_kv = (global_decay_exp * exp_k * v[:, t, :] + state_a) / (
                global_decay_exp * exp_k + state_b + 1e-6
            )
            time_decayed_embeddings = r[:, t, :] * time_decayed_kv

            # Apply spiking layer
            spk, mem = self.spiking_layer(time_decayed_embeddings, mem)

            # Store spiking output
            new_embeddings[:, t, :] = spk

        return new_embeddings
