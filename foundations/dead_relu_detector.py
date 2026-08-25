import torch
import torch.nn as nn


class Solution:

    def detect_dead_neurons(self, model, x):
        dead_fractions = []

        with torch.no_grad():
            # Hook outputs of every ReLU
            activations = []

            hooks = []

            def hook_fn(module, inp, output):
                activations.append(output)

            for layer in model.modules():
                if isinstance(layer, nn.ReLU):
                    hooks.append(layer.register_forward_hook(hook_fn))

            model(x)

            for hook in hooks:
                hook.remove()

            for activation in activations:
                # A neuron is dead if it outputs 0 for every sample
                dead = (activation == 0).all(dim=0)

                dead_fraction = dead.float().mean().item()

                dead_fractions.append(round(dead_fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions):

        # Severe dead neurons
        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

        # First layer has significant death
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # Death strictly increases with depth
        if (
            len(dead_fractions) >= 2
            and all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # Mostly healthy
        if dead_fractions and max(dead_fractions) < 0.1:
            return "healthy"

        # Default
        return "healthy"