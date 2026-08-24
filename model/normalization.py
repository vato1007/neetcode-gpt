import numpy as np

class Solution:
    def forward(self, x, gamma, beta):
        eps = 1e-5

        mean = np.mean(x)
        variance = np.mean((x - mean) ** 2)

        normalized = (x - mean) / np.sqrt(variance + eps)

        output = normalized * gamma + beta

        return np.round(output, 5)