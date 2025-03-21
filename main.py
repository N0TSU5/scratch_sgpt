from Model import Model
from TokenizerWrapper import TokenizerWrapper

if __name__ == "__main__":
    tokenizer = TokenizerWrapper()
    model = Model(vocab_size=tokenizer.vocab_size, embedding_dim=768, num_blocks=10).to(
        "cuda"
    )

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "sigma sigma on the wall, who is the",
    ]

    input_ids = tokenizer.encode(texts)
    logits = model.generate(input_ids, max_new_tokens=25)
    decoded_outputs = tokenizer.decode(logits)

    for i, output in enumerate(decoded_outputs):
        print(f"\nGenerated text for input {i}:\n{output}\n")
