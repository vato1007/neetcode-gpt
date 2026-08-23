import numpy as np

class Solution:
    def sigmoid(self, z):
        result = 1 / (1 + np.exp(-z))
        return np.round(result, 5)

    def relu(self, z):
        return np.maximum(0, z)