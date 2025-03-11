import torch
import torch.nn as nn
import torch.nn.functional as F
from feedforward import FeedForward
from head import Head
from binary_embedding import SubwordBinaryEmbedding
from transformers import AutoTokenizer


class Model(nn.Module):
    def __init__(self, embedding_dim, device="cuda"):
        super().__init__()

        self.device = device

        # Binary Embeddings
        self.embedding = SubwordBinaryEmbedding(embedding_dim=embedding_dim).to(device)

        self.block = nn.Sequential(
            Head(embedding_dim=embedding_dim).to(device),
            FeedForward(embedding_dim=embedding_dim).to(device),
        )

        self.blocks = nn.Sequential(*[self.block for _ in range(5)])

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Classification Head
        self.classification_head = nn.Linear(
            embedding_dim, self.tokenizer.vocab_size
        ).to(device)

    def forward(self, x):
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token indices

        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Binary Embedding
        x = self.embedding(x)

        # Block
        x = self.blocks(x)

        # Classification Head
        logits = self.classification_head(x)  # Shape: [batch_size, seq_len, vocab_size]
        logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).tolist()[0]

        return logits

    def generate(self, prompt, max_length=100):
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        for _ in range(max_length):
            logits = self(tokens)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).tolist()[0]
            tokens = torch.cat([tokens, torch.tensor([idx_next]).unsqueeze(0)], dim=-1)

        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)


if __name__ == "__main__":
    model = Model(embedding_dim=768).to("cuda")
    print(model.generate("i'm in the thick of it"))
