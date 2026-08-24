import numpy as np

class Solution:
    def train(self, X, y, epochs, lr):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n = len(X)

        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            y_pred = X @ w + b
            error = y_pred - y

            dw = (2 / n) * (X.T @ error)
            db = (2 / n) * np.sum(error)

            w -= lr * dw
            b -= lr * db

        return np.round(w, 5), round(b, 5)