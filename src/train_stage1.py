import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from config import STAGE1_MODEL_PATH


class Stage1Trainer:

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=10,
            gamma=1.0,
            scale_pos_weight=2.8,
            eval_metric="logloss",
            random_state=42,
        )

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        self.model.fit(
            X_train,
            y_train,
        )

        probabilities = self.model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities > 0.42
        ).astype(int)

        print("\nStage-1 Results")
        print("------------------------")
        print(
            "Accuracy:",
            accuracy_score(
                y_test,
                predictions,
            ),
        )

        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "on_time",
                    "delayed",
                ],
            )
        )

        joblib.dump(
            self.model,
            STAGE1_MODEL_PATH
        )

        return self.model
