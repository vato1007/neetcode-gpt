import numpy as np

class Solution:
    def forward(self, x, w, b, activation):
        z = np.dot(x, w) + b

        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        else:  # relu
            output = max(0, z)

        return round(float(output), 5)