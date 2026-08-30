import torch

class Solution:
    def batch_loader(self, raw_dataset, context_length, batch_size):
        tokens = raw_dataset.split()

        torch.manual_seed(0)

        starts = torch.randint(
            0,
            len(tokens) - context_length,
            (batch_size,)
        )

        X = [
            tokens[start:start + context_length]
            for start in starts.tolist()
        ]

        Y = [
            tokens[start + 1:start + context_length + 1]
            for start in starts.tolist()
        ]

        return X, Y