import numpy as np

class Solution:
    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)

        return np.round(exp_z / np.sum(exp_z), 4)