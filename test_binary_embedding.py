from binary_embedding import SubwordBinaryEmbedding

def test_binary_embedding():
    # Initialize the model
    embedding_dim = 256
    model = SubwordBinaryEmbedding(embedding_dim=embedding_dim)
    
    # Test cases
    test_texts = [
        "Hello world!",
        "This is a longer sentence to test the embedding.",
        "Testing 123, with some numbers and punctuation!",
    ]
    
    print("Testing SubwordBinaryEmbedding:")
    print("-" * 50)
    
    for text in test_texts:
        print(f"\nInput text: {text}")
        
        # Show tokenization
        tokens = model.tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Get token IDs
        encoded = model.tokenizer([text], padding=True, return_tensors='pt')
        token_ids = encoded['input_ids']
        print(f"Token IDs: {token_ids[0].tolist()}")
        
        # Get binary embeddings
        binary_embeddings = model(text)
        print(f"Embedding shape: {binary_embeddings.shape}")
        
        # Print separator for readability
        print("-" * 50)

if __name__ == "__main__":
    test_binary_embedding() 