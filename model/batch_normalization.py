import numpy as np

class Solution:
    def batch_norm(
        self,
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        momentum=0.1,
        eps=1e-5,
        training=True
    ):
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.mean((x - batch_mean) ** 2, axis=0)

            normalized = (x - batch_mean) / np.sqrt(batch_var + eps)

            running_mean = (
                (1 - momentum) * running_mean
                + momentum * batch_mean
            )

            running_var = (
                (1 - momentum) * running_var
                + momentum * batch_var
            )

        else:
            normalized = (
                (x - running_mean)
                / np.sqrt(running_var + eps)
            )

        output = gamma * normalized + beta

        return (
            np.round(output, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )