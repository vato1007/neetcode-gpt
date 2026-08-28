import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # pos: column vector (seq_len, 1)
        pos = np.arange(seq_len).reshape(-1, 1)

        # i: row vector (1, d_model/2) — one per wave pair
        i = np.arange(d_model // 2).reshape(1, -1)

        # divisor grows with i: controls wave speed
        divisor = np.power(10000, (2 * i) / d_model)

        # angles shape: (seq_len, d_model/2)
        angles = pos / divisor

        PE = np.zeros((seq_len, d_model))
        PE[:, 0::2] = np.sin(angles)   # even columns <- sin
        PE[:, 1::2] = np.cos(angles)   # odd columns  <- cos

        return np.round(PE, 5)
