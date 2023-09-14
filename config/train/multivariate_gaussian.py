from src.data import Dataset
from src.models.multivariate_gaussian import MultivariateGaussian
from src.preprocessors.quantile_normalizer import QuantileNormalizer
from src.preprocessors.to_numpy import ToNumpy
from src.train_config import TrainConfig

dataset = Dataset(path="data/uniform_exp_mixture.csv", train_proportion=0.8)

model = MultivariateGaussian(preprocessors=([ToNumpy(), QuantileNormalizer()]))

generation_options = dict(n_samples=1000)

CONFIG = TrainConfig(
    name="multivariate_gaussian",
    dataset=dataset,
    model=model,
    generation_options=generation_options,
)
