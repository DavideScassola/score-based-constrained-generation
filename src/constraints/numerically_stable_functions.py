import math

import torch

"""
common numerical trick: I use two different but analitically equivalent functions in order to avoid computing large exponentials in different regions
"""

MAXGRAD = 50
MAX_NEG = 20


def numerically_stable_log_logistic(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0, -torch.log(1 + torch.exp(-x)), -torch.log(1 + torch.exp(x)) + x
    )


def numerically_stable_log_logistic_derivative(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0, torch.exp(-x) / (torch.exp(-x) + 1), 1 / (torch.exp(x) + 1)
    )


class LogLogistic(torch.autograd.Function):
    """
    Numerically stable implementation of the log of the logistic function ( -log(1 + exp(-x)) )
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return numerically_stable_log_logistic(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return grad_output * numerically_stable_log_logistic_derivative(input)


log_logistic = LogLogistic.apply
