import torch

class Solution:
    def reshape(self, to_reshape):
        return to_reshape.reshape(-1, 2)

    def average(self, to_avg):
        return torch.mean(to_avg, dim=0)

    def concatenate(self, cat_one, cat_two):
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction, target):
        return torch.mean((prediction - target) ** 2)