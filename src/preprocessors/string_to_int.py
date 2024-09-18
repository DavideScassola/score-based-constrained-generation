import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .preprocessor import Preprocessor


def is_categorical(s: pd.Series):
    # TODO: by default any 'object' is a category
    return s.dtype == "O"


class StringToInt(Preprocessor):
    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame):
        # TODO: implement
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.label_encoder = {
            c: LabelEncoder() for c in df.columns if is_categorical(df[c])
        }
        for v in self.label_encoder:
            df[v] = self.label_encoder[v].fit_transform(df[v])

        self.parameters["categories"] = {column_name: list(enc.classes_) for column_name, enc in self.label_encoder.items()}  # type: ignore

        # putting categories at the end
        self.parameters["names_order"] = list(df.columns)
        self.parameters["continuous_columns"] = [
            c
            for c in self.parameters["names_order"]
            if c not in self.parameters["categories"]
        ]
        self.parameters["categorical_columns"] = list(
            self.parameters["categories"].keys()
        )
        return df[
            self.parameters["continuous_columns"]
            + self.parameters["categorical_columns"]
        ]

    def reverse_transform(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        # restoring original columns order
        df = df_transformed[self.parameters["names_order"]]

        for categorical_column, classes in self.parameters["categories"].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            df[categorical_column] = le.inverse_transform(df[categorical_column])
        return df
