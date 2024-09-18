import pandas as pd
import torch

from .preprocessor import Preprocessor


class ToTensor(Preprocessor):
    def fit(self, x: pd.DataFrame):
        self.parameters["names"] = list(x.columns)

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        x = torch.from_numpy(df.values)
        if all(df.dtypes == float):
            return x.float()
        return x

    def fit_transform(self, x: pd.DataFrame) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)

    def reverse_transform(self, x: torch.Tensor):
        return pd.DataFrame(x.cpu().numpy(), columns=self.parameters["names"])
