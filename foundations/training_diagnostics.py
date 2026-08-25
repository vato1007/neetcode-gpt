import torch
import torch.nn as nn


class Solution:

    def compute_activation_stats(self, model, x):
        stats = []
        activations = []

        # Capture outputs of every Linear layer
        hooks = []

        def hook_fn(module, inp, output):
            activations.append(output.detach())

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        for hook in hooks:
            hook.remove()

        for activation in activations:
            # A neuron is dead if it is <= 0 for every sample
            dead = (activation <= 0).all(dim=0)
            dead_fraction = dead.float().mean().item()

            stats.append({
                "mean": round(activation.mean().item(), 4),
                "std": round(activation.std().item(), 4),
                "dead_fraction": round(dead_fraction, 4)
            })

        return stats

    def compute_gradient_stats(self, model, x, y):
        model.zero_grad()

        prediction = model(x)
        loss = nn.MSELoss()(prediction, y)
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats, gradient_stats):

        # 1. Dead neurons
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"

        # 2. Exploding gradients
        if any(s["norm"] > 1000 for s in gradient_stats):
            return "exploding_gradients"

        # 3. Vanishing gradients in last layer
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation statistics
        if any(s["std"] < 0.1 for s in activation_stats):
            return "vanishing_gradients"

        if any(s["std"] > 10.0 for s in activation_stats):
            return "exploding_gradients"

        return "healthy"