import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from src.util import is_numerical

from .preprocessor import Preprocessor

INT_IS_NUMERICAL_THRESHOLD = 20
DEFAULT_FLOAT_BASE10_DIGITS = 5
DIGITS_BASE = 10


def get_max_digits(s: pd.Series):
    if s.dtype == int:
        return int(np.ceil(np.log10(np.max(s))))
    return DEFAULT_FLOAT_BASE10_DIGITS


def min_max_normalize(
    x: Tensor, *, minimum: float | int, maximum: float | int
) -> Tensor:
    if torch.any((x > maximum) | (x < minimum)):
        raise ValueError("x out of range")
    return (x - minimum) / (maximum - minimum)


def min_max_rescale(x: Tensor, *, minimum: float, maximum: float) -> Tensor:
    return x * (maximum - minimum) + minimum


class EverythingToFloat(Preprocessor):
    def string_to_one_hot(self, c: pd.Series, *, fit: bool) -> Tensor:
        le = LabelEncoder()
        if fit:
            out = le.fit(c)
        else:
            le.classes_ = np.array(self.parameters[c.name]["classes"])
        out = le.transform(c)
        if fit:
            self.parameters[c.name]["type"] = "string"
            self.parameters[c.name]["classes"] = le.classes_.tolist()

        int_tensor = torch.tensor(out, dtype=torch.int64)
        return torch.nn.functional.one_hot(
            int_tensor, num_classes=len(le.classes_)
        ).float()

    def column_to_tensor(self, c: pd.Series, fit: bool):
        if c.dtype == "O":
            out = self.string_to_one_hot(c, fit=fit)
        elif "float" in c.dtype.name:
            assert (
                not c.isnull().any()
            ), f"There are NaNs in this numerical column: {c.name}"
            # TODO: this could be handled
            if fit:
                self.parameters[c.name]["type"] = "float"
            out = torch.tensor(c.values, dtype=torch.float32).unsqueeze(1)
        elif "int" in c.dtype.name:
            if is_numerical(c):
                if fit:
                    self.parameters[c.name]["type"] = "numerical_int"
                out = torch.tensor(c.values, dtype=torch.float32).unsqueeze(1)
            else:
                if fit:
                    self.parameters[c.name]["type"] = "categorical_int"
                    self.parameters[c.name]["classes"] = c.unique().tolist()
                out = torch.nn.functional.one_hot(
                    torch.tensor(c.values, dtype=torch.int64)
                    - min(self.parameters[c.name]["classes"]),
                    num_classes=len(self.parameters[c.name]["classes"]),
                ).float()

        if fit:
            self.parameters[c.name]["slice"] = (
                self.slice_index,
                self.slice_index + out.shape[1],
            )
            self.slice_index = self.parameters[c.name]["slice"][1]
        return out

    def tensor_to_column(self, x: np.ndarray, *, column_name: str):
        y = x[:, slice(*self.parameters[column_name]["slice"])].squeeze()

        if x.shape[0] == 1:
            y = np.expand_dims(y, axis=0)

        match self.parameters[column_name]["type"]:
            case "float":
                c = y.squeeze()
            case "numerical_int":
                c = np.round(y.squeeze()).astype(int)
            case "categorical_int":
                c = y.argmax(axis=-1) + min(self.parameters[column_name]["classes"])
            case "string":
                le = LabelEncoder()
                le.classes_ = np.array(self.parameters[column_name]["classes"])
                c = le.inverse_transform(y.argmax(axis=-1))
            case _:
                c = y

        return pd.Series(c, name=column_name)

    def fit(self, x: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame):
        self.slice_index = 0
        tensor_columns = [self.column_to_tensor(df[c], fit=False) for c in df.columns]
        out = torch.concat(tensor_columns, dim=1)
        assert (
            out.shape[1] == self.parameters["num_features"]
        ), f"error in number of features ({out.shape[1]} but expected {self.parameters['num_features']}), there is a bug in this preprocessor"
        return out

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        self.parameters |= {c: {} for c in df.columns}
        self.parameters["names"] = list(df.columns)
        self.slice_index = 0
        tensor_columns = [self.column_to_tensor(df[c], fit=True) for c in df.columns]
        out = torch.concat(tensor_columns, dim=1)
        self.parameters["num_features"] = out.shape[1]
        print("num_features: ", self.parameters["num_features"])
        return out

    def reverse_transform(self, x: torch.Tensor) -> pd.DataFrame:
        return pd.concat(
            [
                self.tensor_to_column(x.cpu().numpy(), column_name=c)
                for c in self.parameters["names"]
            ],
            axis=1,
        )
