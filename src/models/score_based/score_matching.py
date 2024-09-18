from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.constants import MINIMUM_SAMPLING_T
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.nn.parameters_ema import ParametersEMA
from src.util import reciprocal_distribution_sampler, uniform_sampler

from .score_function import ScoreFunction
from .sdes.sde import Sde

EPS = MINIMUM_SAMPLING_T * 0.1


def log_statistic(*, name: str, value) -> None:
    print(f"{name}: {value}")


def loss_plot(
    *, losses, epoch: int = 0, freq: int = 1, path: str | None = None
) -> None:
    loss_plot_name = "losses.png"
    file = f"{path}/{loss_plot_name}" if path else loss_plot_name
    if epoch % freq == 0:
        plt.plot(losses, alpha=0.8)
        plt.savefig(file)
        plt.close()


def denoising_score_matching_loss(
    *,
    X: torch.Tensor,
    score_model: ScoreFunction,
    train: bool = True,
    likelihood_weighting: bool = False,
    reciprocal_distribution_t: bool = False,
) -> torch.Tensor:
    # TODO: maybe it would be better to do this only one time, not every time the loss is computed
    score_model.train(train)
    t_size = (len(X), 1)

    t_sampler = (
        reciprocal_distribution_sampler
        if reciprocal_distribution_t
        else uniform_sampler
    )
    t = t_sampler(low=EPS, up=1.0, size=t_size, dtype=X.dtype, device=X.device)

    noise = torch.randn_like(X, device=X.device)
    mean, std = score_model.sde.transition_kernel_mean_std(X=X, t=t)
    X_t = mean + std * noise

    estimated_score = (
        score_model.forward(X=X_t, t=t) if train else score_model(X=X_t, t=t)
    )
    if not likelihood_weighting:
        losses = torch.square(estimated_score * std + noise)
    else:
        losses = torch.square(estimated_score + noise / std) * (
            score_model.sde.g(t=t) ** 2
        )
    return torch.mean(losses)


"""
def direct_denoising_score_matching_loss(
    *,
    X: torch.Tensor,
    score_model: ScoreFunction,
    train: bool = True,
    reciprocal_distribution_t: bool = False,
) -> torch.Tensor:
    # TODO: maybe it would be better to do this only one time, not every time the loss is computed
    
    score_model.denoiser_parametrization = True
    
    score_model.train(train)
    t_size = (len(X), 1)

    t_sampler = (
        reciprocal_distribution_sampler
        if reciprocal_distribution_t
        else uniform_sampler
    )
    t = t_sampler(low=EPS, up=1.0, size=t_size, dtype=X.dtype, device=X.device)

    noise = torch.randn_like(X, device=X.device)
    mean, std = score_model.sde.transition_kernel_mean_std(X=X, t=t)
    X_t = mean + std * noise
    
    estimated_x0 = score_model.forward(X=X_t, t=t) if train else score_model(X=X_t, t=t)

    losses = torch.square(estimated_x0 - X)
    
    

    return torch.mean(losses)
"""


def accumulated_denoising_score_matching_loss(*, n: int = 20, **loss_args) -> Tensor:
    loss = denoising_score_matching_loss(**loss_args)
    for i in range(1, n):
        loss += denoising_score_matching_loss(**loss_args)
    return loss / n


def sliced_score_matching_loss(
    *,
    X: torch.Tensor,
    score_model: ScoreFunction,
    n_particles: int = 1,
    train: bool = True,
    noise=False,
) -> torch.Tensor:
    """
    From https://github.com/ermongroup/sliced_score_matching
    """
    # TODO: what's n_particles ?
    # TODO: maybe it would be better to do this only one time, not every time the loss is computed
    score_model.train(train)

    if noise:
        t = reciprocal_distribution_sampler(
            low=EPS, up=1.0, size=(len(X), 1), dtype=X.dtype, device=X.device
        )
        noise = torch.randn_like(X, device=X.device)
        mean, std = score_model.sde.transition_kernel_mean_std(X=X, t=t)
        X = mean + std * noise
    else:
        t = torch.zeros(size=(len(X), 1), device=X.device, dtype=X.dtype)

    dup_samples = (
        X.unsqueeze(0).expand(n_particles, *X.shape).contiguous().view(-1, *X.shape[1:])
    )
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_model.forward(X=dup_samples, t=t)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean()


def hybrid_ssm_dsm_loss(*, w: float, **kwargs):
    return w * sliced_score_matching_loss(noise=True, **kwargs) + (
        1 - w
    ) * denoising_score_matching_loss(reciprocal_distribution_t=True, **kwargs)


def score_matching(
    *,
    train_set: torch.Tensor,
    score_model: ScoreFunction,
    optimization: Optimization,
    sm_loss: Callable = denoising_score_matching_loss,
) -> list:
    tensor_train_set = TensorDataset(train_set.to(score_model.device))

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=optimization.batch_size,
        shuffle=True,
        drop_last=False,
    )
    score_model.train(True)
    optimizer = optimization.build_optimizer(score_model.parameters())

    epochs_losses = []
    n_batches = len(train_loader)

    parameters_ema = (
        ParametersEMA(
            parameters=score_model.parameters(), decay=optimization.parameters_momentum
        )
        if optimization.parameters_momentum
        else None
    )

    for epoch in range(optimization.epochs):
        log_statistic(name="epoch", value=epoch + 1)
        losses = np.zeros(n_batches)

        pbar = tqdm(total=n_batches)
        for i, X_batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = sm_loss(X=X_batch[0], score_model=score_model, train=True)
            loss.backward()

            if optimization.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

            optimizer.step()
            losses[i] = loss.detach().item()
            pbar.update(1)
        pbar.close()

        epoch_loss = np.mean(losses)
        epochs_losses.append(epoch_loss)
        log_statistic(name="train loss", value=epoch_loss)
        print()

        loss_plot(
            losses=epochs_losses,
            epoch=epoch,
            freq=1 + optimization.epochs // 5,
        )
        if parameters_ema:
            parameters_ema.update(score_model.parameters())

    if parameters_ema:
        parameters_ema.copy_to(score_model.parameters())
    loss_plot(losses=epochs_losses)
    return epochs_losses
