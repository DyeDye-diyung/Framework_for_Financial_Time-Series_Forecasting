from collections import namedtuple

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# constants

Statistics = namedtuple('Statistics', [
    'mean',
    'variance',
    'gamma',
    'beta'
])

# reversible instance normalization
# proposed in https://openreview.net/forum?id=cGDAkQo1C0p

class RevIN(Module):
    def __init__(
        self,
        num_variates,
        affine = True,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1), requires_grad = affine)
        self.beta = nn.Parameter(torch.zeros(num_variates, 1), requires_grad = affine)

    def forward(self, x, return_statistics = False):
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        var_rsqrt = var.clamp(min = self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min = self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        statistics = Statistics(mean, var, self.gamma, self.beta)

        return rescaled, reverse_fn, statistics

# sanity check

if __name__ == '__main__':

    rev_in = RevIN(3)

    x = torch.randn(2, 3, 4)  # shape: (batch_size, num_variate, seq_len)
    target_variable = x[:, -1:, :]
    print(target_variable)
    print(target_variable.shape)

    normalized, reverse_fn, statistics = rev_in(x, return_statistics = True)
    normalized_target_variable = normalized[:, -1:, :]
    print(normalized_target_variable)
    print(normalized_target_variable.shape)

    target_variable_mean = statistics.mean[:, -1:, :]
    target_variable_variance = statistics.variance[:, -1:, :]
    target_variable_gamma = statistics.gamma[-1:, :]
    target_variable_beta = statistics.beta[-1:, :]

    # print(target_variable_mean)

    # print(target_variable_mean.shape)
    # print(target_variable_variance.shape)
    # print(target_variable_gamma.shape)
    # print(target_variable_beta.shape)

    eps = 1e-5
    clamped_gamma = torch.sign(target_variable_gamma) * target_variable_gamma.abs().clamp(min=eps)
    unscaled_output = (normalized_target_variable - target_variable_beta) / clamped_gamma
    reverse_output = unscaled_output * target_variable_variance.sqrt() + target_variable_mean
    print(reverse_output)
    print(reverse_output.shape)

    out = reverse_fn(normalized)

    # print(statistics.mean.shape)
    # print(statistics.variance.shape)
    # print(statistics.gamma.shape)
    # print(statistics.beta.shape)

    assert torch.allclose(x, out)
