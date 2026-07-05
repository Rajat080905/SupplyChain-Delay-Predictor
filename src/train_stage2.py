"""
train_stage2.py

Stage-2 Classification
At-Risk vs Delayed

Supports:
- Random Forest
- XGBoost

Author: Rajat
"""

import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from config import STAGE2_RF_MODEL_PATH, STAGE2_XGB_MODEL_PATH


class Stage2Trainer:

    def __init__(self, model="random_forest"):

        self.model_name = model.lower()

        if self.model_name == "random_forest":

            self.model = RandomForestClassifier(

                n_estimators=650,
                max_depth=18,

                min_samples_leaf=12,
                min_samples_split=20,

                max_features="sqrt",

                class_weight={
                    0: 1.0,
                    1: 2.2
                },

                random_state=42,
                n_jobs=-1
            )

        elif self.model_name == "xgboost":

            self.model = xgb.XGBClassifier(

                n_estimators=220,
                learning_rate=0.045,

                max_depth=3,

                min_child_weight=35,

                gamma=2.5,

                subsample=0.65,
                colsample_bytree=0.65,

                reg_alpha=4.0,
                reg_lambda=8.0,

                objective="binary:logistic",

                eval_metric="logloss",

                random_state=42,

                n_jobs=-1

            )

        else:
            raise ValueError(
                "Model must be 'random_forest' or 'xgboost'"
            )

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        # -----------------------------
        # XGBoost imbalance handling
        # -----------------------------

        if self.model_name == "xgboost":

            ratio = (
                y_train.value_counts()[0]
                /
                y_train.value_counts()[1]
            )

            self.model.set_params(
                scale_pos_weight=ratio
            )

        # -----------------------------
        # Training
        # -----------------------------

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("\n==============================")
        print(f"Stage-2 ({self.model_name.upper()})")
        print("==============================")

        print(f"Accuracy : {accuracy:.4f}\n")

        print(

            classification_report(

                y_test,

                predictions,

                target_names=[
                    "at_risk",
                    "delayed"
                ]

            )

        )

        # -----------------------------
        # Save Model
        # -----------------------------

        if self.model_name == "random_forest":
            save_path = STAGE2_RF_MODEL_PATH
        else:
            save_path = STAGE2_XGB_MODEL_PATH

        joblib.dump(
            self.model,
            save_path
        )

        print(
            f"\nModel saved to models/stage2_{self.model_name}.pkl"
        )

        return self.model, predictions
