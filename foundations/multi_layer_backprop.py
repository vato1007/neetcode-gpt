import numpy as np

class Solution:
    def forward_and_backward(self, x, W1, b1, W2, b2, y_true):
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        y_pred = W2 @ a1 + b2

        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)

        # Backward pass
        dy = 2 * (y_pred - y_true) / len(y_true)

        # Layer 2
        dW2 = np.outer(dy, a1)
        db2 = dy

        # Backprop through W2
        da1 = W2.T @ dy

        # ReLU derivative
        dz1 = da1 * (z1 > 0)

        # Layer 1
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }