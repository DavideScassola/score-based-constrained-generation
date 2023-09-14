import numpy as np
from sklearn.preprocessing import QuantileTransformer

from .preprocessor import Preprocessor


class QuantileNormalizer(Preprocessor):
    # TODO: update this, and add non-differentiable transformers like this to model class
    def __init__(self, output_distribution: str = "normal"):
        self.transformer = QuantileTransformer(output_distribution=output_distribution)
        self.fit = self.transformer.fit
        self.transform = self.transformer.transform
        self.reverse_transform = self.transformer.inverse_transform
