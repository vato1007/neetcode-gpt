import numpy as np

class Solution:
    def forward(self, x, weights, biases):
        x = np.array(x, dtype=float)

        for i in range(len(weights)):
            x = np.dot(x, np.array(weights[i])) + np.array(biases[i])

            # ReLU for hidden layers only
            if i < len(weights) - 1:
                x = np.maximum(0, x)

        return np.round(x, 5)