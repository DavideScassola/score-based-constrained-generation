import os
from typing import Callable, Tuple

from torch import Tensor

from src.data import Dataset
from src.models.score_based.nn.optimization import Optimization
from src.models.score_based.score_matching import denoising_score_matching_loss
from src.models.score_based.sdes.sde import Sde
from src.models.time_series_model import TimeSeriesModel
from src.report import *
from src.util import store_json

from .score_based_sde import ScoreBasedSde


def store_array(x: np.ndarray | Tensor, path: Path, name: str) -> None:
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    np.save(f"{path}/{name}", x)


def torch_rejection_sampling(x: Tensor, acceptance_function) -> Tensor:
    """acceptance_function is a probability density function"""
    acceptance = acceptance_function(x)
    assert tuple(acceptance.shape) == (len(x),)

    max_acceptance = torch.max(acceptance)
    if max_acceptance.item() <= 0:
        print(f"Warning: max_acceptance is {max_acceptance}, no sample can be accepted")
        return Tensor()
    acceptance /= max_acceptance
    selection = torch.bernoulli(acceptance) == 1
    return x[selection]


def constraint_rejection_sampling(X: Tensor, constraint: Constraint) -> Tensor:
    probability_distribution = lambda x: torch.exp(
        constraint.strength * constraint.f(x)
    )
    return torch_rejection_sampling(X, probability_distribution)


def refill_rejection_sampling(
    X: torch.Tensor, constraint: Constraint, sampler, generation_options
) -> Tuple[Tensor, float]:
    X_rejection_sampling = constraint_rejection_sampling(X, constraint)

    discarded_samples = len(X) - len(X_rejection_sampling)
    if discarded_samples > 0:
        acceptance_rate = len(X_rejection_sampling) / len(X)

        if acceptance_rate == 0:
            print(f"Warning: all samples were rejected")
            return Tensor(), 0

        if (discarded_samples // acceptance_rate) > 1e4:
            print(
                f"Warning: cannot generate {discarded_samples // acceptance_rate} samples"
            )
            return Tensor(), 0

        generation_options["n_samples"] = int(discarded_samples // acceptance_rate)

        print("not enough samples after rejection, generating again ...")
        X_new = sampler(**generation_options)
        X_rejection_sampling_new = constraint_rejection_sampling(X_new, constraint)
        X_rejection_sampling = torch.concat(
            [X_rejection_sampling, X_rejection_sampling_new]
        )

    return X_rejection_sampling[: len(X)], len(X_rejection_sampling) / (
        len(X) + len(X_new)
    )


class TimeSeriesScoreBasedSde(ScoreBasedSde, TimeSeriesModel):
    def __init__(
        self,
        *,
        sde: Sde,
        score_function_class,
        score_function_hyperparameters: dict,
        optimization: Optimization,
        names: list[str],
        score_matching_loss: Callable = denoising_score_matching_loss,
        **kwargs,
    ) -> None:
        super().__init__(
            sde=sde,
            score_function_class=score_function_class,
            score_function_hyperparameters=score_function_hyperparameters,
            score_matching_loss=score_matching_loss,
            optimization=optimization,
            **kwargs,
        )
        self.names = names

    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: Dataset,
        generation_options: dict,
        constraint: Constraint | None = None,
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        print("Sampling from constrained model...")
        samples = self.generate(**generation_options)
        plots_folder = str(path) + "/samples"
        os.makedirs(plots_folder)

        store_array(samples, path=report_folder, name="guidance.npy")

        print("Plotting samples from constrained model...")
        time_series_plots(
            samples=samples.cpu().numpy(),
            folder=plots_folder,
            n_plots=10,
            series_per_plot=10,
            features_names=self.names,
        )
        if constraint:
            satisfaction_plot(
                X=samples,
                constraint=constraint,
                path=report_folder,
                label="guidance",
                color="orange",
            )

        if not constraint:
            print("Plotting data from train set...")
            plots_folder = str(path) + "/samples_from_train"
            os.makedirs(plots_folder)

            time_series_plots(
                samples=np.random.permutation(dataset.get(train=True).cpu().numpy()),
                folder=plots_folder,
                n_plots=5,
                series_per_plot=15,
                features_names=self.names,
                color="blue",
            )
        else:
            print("Plotting samples from rejection sampling...")
            plots_folder = str(path) + "/rejection_sampling"
            os.makedirs(plots_folder)

            # TODO: setting the score to None temporarily is not so elegant
            original_score = self.score_model.constraint
            self.score_model.constraint = None

            print("Sampling from unconstrained model...")
            samples = self.generate(**generation_options)
            store_array(samples, path=report_folder, name="no_guidance.npy")

            samples, acceptance_rate = refill_rejection_sampling(
                samples, constraint, self.generate, generation_options
            )
            store_array(samples, path=report_folder, name="rejection_sampling.npy")

            store_json(
                {"acceptance_rate": acceptance_rate},
                file=report_folder / Path("acceptance_rate.json"),
            )
            self.score_model.constraint = original_score

            if len(samples) > 0:
                time_series_plots(
                    samples=samples.cpu().numpy(),
                    folder=plots_folder,
                    n_plots=10,
                    series_per_plot=10,
                    features_names=self.names,
                    color="green",
                )

                satisfaction_plot(
                    X=samples,
                    constraint=constraint,
                    path=report_folder,
                    label="rejection sampling",
                    color="green",
                )
