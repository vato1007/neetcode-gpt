import numpy as np

class Solution:
    def backward(self, x, w, b, y_true):
        # Forward pass
        z = np.dot(w, x) + b
        y_pred = 1 / (1 + np.exp(-z))

        # Gradient of loss through sigmoid
        delta = (y_pred - y_true) * y_pred * (1 - y_pred)

        # Gradients
        dL_dw = delta * x
        dL_db = delta

        return np.round(dL_dw, 5), round(float(dL_db), 5)