import numpy as np

class Solution:
    def lookup(self, embeddings, token_ids):
        result = embeddings[token_ids]
        return np.round(result, 5)