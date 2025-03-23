import torch
from DataLoader import TextDataLoader
from Model import Model
from TokenizerWrapper import TokenizerWrapper

if __name__ == "__main__":
    tokenizer = TokenizerWrapper()
    model = Model(vocab_size=tokenizer.vocab_size, embedding_dim=768, num_blocks=10).to(
        "cuda"
    )

    train_loader = TextDataLoader("input.txt", tokenizer, train=True)
    test_loader = TextDataLoader("input.txt", tokenizer, train=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    @torch.no_grad()
    def estimate_loss():
        eval_iters = 200

        out = []
        model.eval()
        for loader in [train_loader, test_loader]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = loader.get_batch()
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out.append(losses.mean())
        model.train()
        return out

    for iter in range(5000):
        # sample a batch of data
        xb, yb = train_loader.get_batch()

        # every once in a while evaluate the loss on train and val sets
        if iter % 1000 == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}"
            )

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model, "model_full.pth")  # Save the whole model

