import numpy as np

class Solution:
    learning_rate = 0.01

    def train_model(self, X, Y, num_iterations, initial_weights):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        weights = np.array(initial_weights, dtype=float)

        n = len(X)

        for _ in range(num_iterations):
            predictions = np.dot(X, weights)

            error = predictions - Y

            gradient = (2 / n) * np.dot(X.T, error)

            weights = weights - self.learning_rate * gradient

        return np.round(weights, 5)