"""
predict.py

Predict delivery status for new shipment data.

Author: Rajat
"""

import pandas as pd
import numpy as np
import joblib

from config import (
    ENCODER_PATH,
    STAGE1_MODEL_PATH,
    STAGE2_RF_MODEL_PATH
)


class Predictor:

    def __init__(self):

        self.encoder = joblib.load(ENCODER_PATH)

        self.stage1_model = joblib.load(STAGE1_MODEL_PATH)

        self.stage2_model = joblib.load(STAGE2_RF_MODEL_PATH)

    def preprocess(self, df):

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        encoded = self.encoder.transform(
            df[categorical_columns]
        )

        X = np.hstack([
            df[numeric_columns].values,
            encoded
        ])

        return X

    def predict(self, df):

        X = self.preprocess(df)

        # ----------------------------
        # Stage 1
        # ----------------------------

        stage1_probs = self.stage1_model.predict_proba(X)

        stage1_pred = (
            stage1_probs[:, 1] >= 0.42
        ).astype(int)

        final_prediction = []
        confidence = []

        # ----------------------------
        # Stage 2
        # ----------------------------

        for i in range(len(stage1_pred)):

            if stage1_pred[i] == 0:

                final_prediction.append("On Time")

                confidence.append(
                    round(stage1_probs[i][0], 3)
                )

            else:

                stage2_prob = self.stage2_model.predict_proba(
                    X[i].reshape(1, -1)
                )[0]

                stage2_pred = np.argmax(stage2_prob)

                if stage2_pred == 0:

                    final_prediction.append("At Risk")

                else:

                    final_prediction.append("Delayed")

                confidence.append(
                    round(np.max(stage2_prob), 3)
                )

        results = df.copy()

        results["Prediction"] = final_prediction

        results["Confidence"] = confidence

        return results
