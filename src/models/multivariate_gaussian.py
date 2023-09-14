import numpy as np
import torch

from .model import Model


class MultivariateGaussian(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit_multivariate_gaussian(self, X: torch.Tensor):
        X = X.numpy()
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T, ddof=1)
        if len(self.cov.shape) == 0:
            self.cov = self.cov.reshape(1, 1)

    def sample_multivariate_gaussian(self, n: int):
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def _train(self, X: torch.Tensor):
        self.fit_multivariate_gaussian(X)

    # TODO: change to tensor?
    def _generate(self, n_samples: int = 1000) -> torch.Tensor:
        return torch.from_numpy(self.sample_multivariate_gaussian(n_samples))

    def store(self, path: str):
        # TODO
        pass

    def load_(self, path: str):
        # TODO
        pass

    def _store(self, path: str):
        # TODO
        pass

    def _load_(self, path: str):
        # TODO
        pass
