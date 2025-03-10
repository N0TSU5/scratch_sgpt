import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from head import Head
from token_shift import TokenShift


def test_srwkv_layer():
    # Test parameters
    batch_size = 2
    seq_len = 4
    embedding_dim = 8

    # Create a simple input tensor
    x = torch.rand(batch_size, seq_len, embedding_dim).cuda()

    # Initialize SRWKV layer
    head = Head(embedding_dim=embedding_dim)

    print("=== Testing SRWKV Layer ===")
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = head(x)

    print(f"Output shape: {output.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")

    # Check if output contains binary values (0 or 1)
    binary_check = torch.logical_or(
        torch.isclose(output, torch.zeros_like(output)),
        torch.isclose(output, torch.ones_like(output)),
    )
    binary_ratio = binary_check.float().mean().item()
    print(f"Binary values ratio: {binary_ratio:.4f}")

    # Check if there are any spikes
    spike_ratio = output.float().mean().item()
    print(f"Spike ratio: {spike_ratio:.4f}")

    # Test with different sequence lengths
    print("\n=== Testing with different sequence lengths ===")
    for test_seq_len in [1, 8, 16]:
        test_x = torch.rand(batch_size, test_seq_len, embedding_dim).cuda()
        test_output = head(test_x)
        print(
            f"Seq len {test_seq_len}: input {test_x.shape} → output {test_output.shape}"
        )

    # Test with different batch sizes
    print("\n=== Testing with different batch sizes ===")
    for test_batch_size in [1, 4, 8]:
        test_x = torch.rand(test_batch_size, seq_len, embedding_dim).cuda()
        test_output = head(test_x)
        print(
            f"Batch size {test_batch_size}: input {test_x.shape} → output {test_output.shape}"
        )

    # Test gradient flow
    print("\n=== Testing gradient flow ===")
    x.requires_grad = True
    output = head(x)
    loss = output.sum()
    loss.backward()

    print(f"Input gradient exists: {x.grad is not None}")
    if x.grad is not None:
        print(f"Input gradient shape: {x.grad.shape}")
        print(f"Input gradient mean: {x.grad.abs().mean().item():.6f}")

    # Check model parameters
    print("\n=== Model Parameters ===")
    total_params = sum(p.numel() for p in head.parameters())
    print(f"Total parameters: {total_params}")

    # Print parameter shapes
    print("\nParameter shapes:")
    for name, param in head.named_parameters():
        print(f"{name}: {param.shape}")

    return output


def test_integration():
    # Test integration with TokenShift
    print("\n=== Testing Integration with TokenShift ===")

    embedding_dim = 8
    token_shift = TokenShift(embedding_dim=embedding_dim)

    # Create a simple input
    x = torch.rand(2, 4, embedding_dim).cuda()

    # Apply token shift
    shifted_x = token_shift(x)

    print(f"Original shape: {x.shape}")
    print(f"After TokenShift: {shifted_x.shape}")

    # Pass through SRWKV
    head = Head(embedding_dim=embedding_dim)
    output = head(shifted_x)

    print(f"After SRWKV: {output.shape}")
    print(f"Contains spikes: {output.float().mean().item():.4f}")


if __name__ == "__main__":
    print("=== Running SRWKV Layer Tests ===\n")
    test_srwkv_layer()
    test_integration()
