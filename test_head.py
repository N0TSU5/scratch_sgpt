import torch
import sys
from pathlib import Path
from head import Head
from binary_embedding import SubwordBinaryEmbedding

def test_srwkv_layer(string):
    """Test the basic functionality of the SRWKV layer"""
    embedding_dim = 16

    # Initialize embedding and tokenize string
    embedding = SubwordBinaryEmbedding(embedding_dim=embedding_dim)
    x = embedding(string)

    # Initialize layer with lower threshold to ensure spikes
    head = Head(embedding_dim=embedding_dim)

    # Forward pass
    output = head(x)

    # Check shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Check if output contains spikes (binary values)
    binary_values = torch.logical_or(
        torch.isclose(output, torch.zeros_like(output)),
        torch.isclose(output, torch.ones_like(output)),
    )
    binary_ratio = binary_values.float().mean().item()
    assert (
        binary_ratio > 0.99
    ), f"Expected binary values, got binary ratio: {binary_ratio}"

    # Check if there are some spikes (not all zeros)
    spike_ratio = output.float().mean().item()
    assert (
        0.01 < spike_ratio < 0.99
    ), f"Expected some spikes, got spike ratio: {spike_ratio}"

    # Check gradient flow
    x.requires_grad = True
    output = head(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Expected gradients to flow back to input"

    print("✓ SRWKV Layer Test Passed!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Binary values: {binary_ratio:.2%}")
    print(f"  - Spike ratio: {spike_ratio:.2%}")
    print(f"  - Gradient flow: OK")

    return output


if __name__ == "__main__":
    print("=== Running SRWKV Layer Tests ===\n")
    test_srwkv_layer("feel what they feel, Chelsea. 120 years")
    test_srwkv_layer("who lives in a pineapple under the sea?")
    test_srwkv_layer("i'm a little teapot, short and stout")
    test_srwkv_layer("the quick brown fox jumps over the lazy dog")
    test_srwkv_layer("the five boxing wizards jump quickly")
    test_srwkv_layer("pack my box with five dozen liquor jugs")
    test_srwkv_layer("the seven deadly sins!")

