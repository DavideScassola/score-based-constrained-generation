import numpy as np
import pandas as pd

from .preprocessor import Preprocessor


class ToNumpy(Preprocessor):
    def __init__(self, float_type: type = np.float32):
        super().__init__()
        self.float_type = float_type

    def fit(self, x: pd.DataFrame):
        self.parameters["names"] = list(x.columns)

    def transform(self, x: pd.DataFrame):
        return x.to_numpy().astype(self.float_type) if self.float_type else x.to_numpy()

    def reverse_transform(self, x: np.ndarray):
        return pd.DataFrame(x, columns=self.parameters["names"])
