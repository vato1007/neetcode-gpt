import torch
import torch.nn.functional as F


class Solution:
    def train(self, model, data, epochs, context_length, batch_size, lr):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            max_start = len(data) - context_length

            starts = torch.randint(
                0,
                max_start,
                (batch_size,)
            )

            X = torch.stack([
                data[start:start + context_length]
                for start in starts
            ])

            Y = torch.stack([
                data[start + 1:start + context_length + 1]
                for start in starts
            ])

            logits = model(X)

            # logits: (B, T, vocab_size)
            # Y:     (B, T)
            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = Y.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)