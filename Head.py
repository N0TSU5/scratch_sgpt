import torch
import torch.nn as nn
import torch.nn.functional as F
from TokenShift import TokenShift
import snntorch as snn


class Head(nn.Module):
    """A single head of the recurrent spiking state space model"""

    def __init__(self, embedding_dim: int, device="cuda") -> None:
        super().__init__()

        self.device = device

        self.token_shift = TokenShift(embedding_dim=embedding_dim).to(device)

        # Initialize Receptance, Key and Value matrices.
        self.R = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device)

        # Initialize decay parameters.
        self.token_decay = nn.Linear(embedding_dim, embedding_dim, bias=False).to(
            device
        )
        self.global_decay = nn.Parameter(torch.randn(embedding_dim, device=device))

        # Initialize spiking layer.
        self.spiking_layer = snn.Leaky(beta=0.3, threshold=0.2).to(device)

    def forward(self, embeddings):
        batch_size, seq_len, embedding_dim = embeddings.shape

        embeddings = self.token_shift(embeddings)

        # Initialize hidden states.
        state_a = torch.zeros(batch_size, embedding_dim, device=self.device)
        state_b = torch.zeros(batch_size, embedding_dim, device=self.device)

        # Project input through Receptance, Key and Value matrices.
        r = torch.sigmoid(self.R(embeddings))
        k = self.K(embeddings)
        v = self.V(embeddings)

        # Initialize output tensor.
        new_embeddings = torch.zeros_like(embeddings, device=self.device)

        # Initialize membrane potential for spiking neurons.
        mem = torch.zeros(batch_size, embedding_dim, device=self.device)

        for t in range(seq_len):
            # Compute decays for current token
            token_decay_t = torch.exp(self.token_decay(embeddings[:, t, :]))
            global_decay_exp = torch.exp(self.global_decay)

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

            spk, mem = self.spiking_layer(time_decayed_embeddings, mem)

            # Store spiking output.
            new_embeddings[:, t, :] = spk

        return new_embeddings
