import torch
import torch.nn as nn
import torch.nn.functional as F
from token_shift import TokenShift
import snntorch as snn

class FeedForward(nn.Module):
    """A single feedforward layer of the recurrent spiking state space model"""
    def __init__(self, embedding_dim: int, device="cuda") -> None:
        super().__init__()

        self.device = device

        # Initialize token shift
        self.token_shift = TokenShift(embedding_dim=embedding_dim).to(device)

        # Initialize matrices
        self.G = nn.Linear(embedding_dim, embedding_dim*2, bias=False).to(device) # Growth matrix
        self.P = nn.Linear(embedding_dim, embedding_dim, bias=False).to(device) # Gating matrix
        self.S = nn.Linear(embedding_dim*2, embedding_dim, bias=False).to(device) # Projection matrix

        # Initialize activation function
        self.relu = nn.ReLU()

        # Initialize spiking layer
        self.spiking_layer = snn.Leaky(beta=0.3, threshold=0.2).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the spiking feedforward layer"""
        # Apply token shift preprocessing
        batch_size, seq_len, embedding_dim = x.shape
        
        # Token shift
        x = self.token_shift(x)

        # Compute transformation values
        growth_potential = self.G(x)  # Expanded feature transformation
        gating_signal = torch.sigmoid(self.P(x))  # Sigmoid gating mechanism

        # Apply ReLU6 and square activation for stability
        growth_potential = self.relu(growth_potential) ** 2  

        # Project back to embedding dimensions
        projected_potential = self.S(growth_potential)

        # Apply gating
        gated_potential = gating_signal * projected_potential  

        # Initialize membrane potential for spiking neurons
        mem = torch.zeros(batch_size, embedding_dim, device=self.device)

        # Initialize output tensor
        new_embeddings = torch.zeros_like(x, device=self.device)

        for t in range(seq_len):
            # Apply spiking activation
            spk, mem = self.spiking_layer(gated_potential[:, t, :], mem)
            new_embeddings[:, t, :] = spk

        return new_embeddings

    