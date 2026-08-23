import numpy as np

class Solution:
    def get_model_prediction(self, X, weights):
        return np.round(np.dot(X, weights), 5)

    def get_error(self, model_prediction, ground_truth):
        mse = np.mean(
            (np.array(model_prediction) - np.array(ground_truth)) ** 2
        )
        return round(float(mse), 5)