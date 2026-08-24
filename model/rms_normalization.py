import numpy as np

class Solution:
    def rms_norm(self, x, gamma, eps=1e-5):
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        rms = np.sqrt(np.mean(x ** 2) + eps)

        output = gamma * (x / rms)

        return np.round(output, 4).tolist()