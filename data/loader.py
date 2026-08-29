import torch

class Solution:
    def create_batches(self, data, context_length, batch_size):
        torch.manual_seed(0)

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

        return X, Y