"""
data_split.py

Responsible for:
- Train-test split
- One-hot encoding
- Preparing Stage-1 data
- Preparing Stage-2 data
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib


class DataSplitter:

    def __init__(self, df):
        self.df = df

    def prepare_data(self):

        X = self.df.drop(
            columns=[
                "stage1_target",
                "stage2_target"
            ]
        )

        y1 = self.df["stage1_target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y1,
            test_size=0.30,
            stratify=y1,
            random_state=42,
        )

        numeric_columns = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        train_cat = encoder.fit_transform(
            X_train[categorical_columns]
        )

        test_cat = encoder.transform(
            X_test[categorical_columns]
        )

        X_train_final = np.hstack(
            [
                X_train[numeric_columns].values,
                train_cat,
            ]
        )

        X_test_final = np.hstack(
            [
                X_test[numeric_columns].values,
                test_cat,
            ]
        )

        joblib.dump(
            encoder,
            "models/encoder.pkl"
        )

        mask_train = y_train == 1
        mask_test = y_test == 1

        X2_train = X_train_final[mask_train]
        X2_test = X_test_final[mask_test]

        y2_train = self.df.loc[
            X_train.index[mask_train],
            "stage2_target"
        ]

        y2_test = self.df.loc[
            X_test.index[mask_test],
            "stage2_target"
        ]

        return {
            "X1_train": X_train_final,
            "X1_test": X_test_final,
            "y1_train": y_train,
            "y1_test": y_test,
            "X2_train": X2_train,
            "X2_test": X2_test,
            "y2_train": y2_train,
            "y2_test": y2_test,
        }
