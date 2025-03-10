import torch
from token_shift import TokenShift

def test_token_shift():
    # Test parameters
    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    
    # Create a simple binary input tensor on CUDA
    x = torch.tensor([
        # Batch 1: simple pattern
        [[1, 0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1]],
        # Batch 2: different pattern
        [[0, 0, 1, 1, 0, 0, 1, 1],
         [1, 1, 0, 0, 1, 1, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1, 0, 1, 0]]
    ]).cuda().float()
    
    # Initialize shifter (will be on CUDA)
    shifter = TokenShift(embedding_dim=embedding_dim)
    
    print("Original input:")
    print("Batch 1:")
    print(x[0].cpu())  # Move to CPU for printing
    print("\nBatch 2:")
    print(x[1].cpu())
    
    # Apply shift and get output (stays on CUDA)
    output = shifter(x)
    
    print("\n=== Testing Token Shift ===")
    print("\nOutput:")
    print("Batch 1:")
    print(output[0].cpu())  # Move to CPU for printing
    print("\nBatch 2:")
    print(output[1].cpu())
    
    # Verify properties (on CUDA)
    print("\nVerifying properties:")
    print(f"Output shape matches input: {output.shape == x.shape}")
    print(f"Values are between 0 and 1: {torch.all((output >= 0) & (output <= 1))}")
    print(f"Different from input: {not torch.equal(output, x)}")
    
    # Show how much shifting occurred
    diff = torch.abs(output - x).mean().item()
    print(f"Average difference from input: {diff:.4f}")
    
    # Print the shift weights
    shift_weights = shifter.compute_shift_weights().cpu()
    print("\nShift weights:")
    print(shift_weights)
    
    # Visualize the effect on the first token of each sequence
    print("\nEffect on first token (should be unchanged):")
    print(f"Batch 1, token 0: {torch.allclose(x[0, 0], output[0, 0])}")
    print(f"Batch 2, token 0: {torch.allclose(x[1, 0], output[1, 0])}")
    
    # Visualize the effect on subsequent tokens
    print("\nEffect on subsequent tokens (should be shifted):")
    for i in range(1, seq_len):
        print(f"Batch 1, token {i}: diff = {torch.abs(x[0, i] - output[0, i]).mean().item():.4f}")
        print(f"Batch 2, token {i}: diff = {torch.abs(x[1, i] - output[1, i]).mean().item():.4f}")

if __name__ == "__main__":
    print("=== Testing Token Shift Module ===\n")
    test_token_shift()
