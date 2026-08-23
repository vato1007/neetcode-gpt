class Solution:
    def get_minimizer(self, iterations, learning_rate, init):
        x = init

        for _ in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient

        return round(x, 5)