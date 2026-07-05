"""
main.py

Main pipeline for the Supply Chain Delay Predictor.

Workflow:
1. Load and preprocess data
2. Split & encode data
3. Train Stage-1 model
4. Train Stage-2 model
5. Evaluate both models
6. Save models and plots

Author: Rajat
"""
from config import *
from src.preprocessing import DataPreprocessor
from src.data_split import DataSplitter
from src.train_stage1 import Stage1Trainer
from src.train_stage2 import Stage2Trainer
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer
from src.utils import (
    create_directories,
    print_heading,
    set_seed,
)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "data/dataset.xlsx"

# ==========================================================
# MAIN
# ==========================================================

def main():

    # Create folders
    create_directories()

    # Set random seed
    set_seed(42)

    # ======================================================
    # PREPROCESSING
    # ======================================================

    print_heading("DATA PREPROCESSING")
    preprocessor = DataPreprocessor(DATASET_PATH)

    df = preprocessor.preprocess()

    print(df.head())

    # ======================================================
    # DATA SPLITTING
    # ======================================================

    print_heading("TRAIN TEST SPLIT")

    splitter = DataSplitter(df)

    data = splitter.prepare_data()

    # ======================================================
    # STAGE-1 TRAINING
    # ======================================================

    print_heading("STAGE 1 TRAINING")

    stage1 = Stage1Trainer()

    stage1_model = stage1.train(

        data["X1_train"],
        data["y1_train"],

        data["X1_test"],
        data["y1_test"]

    )

    # ======================================================
    # STAGE-1 EVALUATION
    # ======================================================

    print_heading("STAGE 1 EVALUATION")

    evaluator1 = ModelEvaluator(stage1_model)

    results1 = evaluator1.evaluate(

        data["X1_test"],
        data["y1_test"],

        threshold=0.42

    )

    Visualizer.plot_confusion_matrix(

        results1["confusion_matrix"],

        "Stage 1 Confusion Matrix"

    )

    Visualizer.plot_roc_curve(

        results1["fpr"],

        results1["tpr"],

        results1["auc"],

        "Stage 1 ROC"

    )

    # ======================================================
    # STAGE-2 TRAINING
    # ======================================================

    print_heading("STAGE 2 TRAINING")

    stage2 = Stage2Trainer(

        model="random_forest"

    )

    stage2_model, _ = stage2.train(

        data["X2_train"],
        data["y2_train"],

        data["X2_test"],
        data["y2_test"]

    )

    # ======================================================
    # STAGE-2 EVALUATION
    # ======================================================

    print_heading("STAGE 2 EVALUATION")

    evaluator2 = ModelEvaluator(stage2_model)

    results2 = evaluator2.evaluate(

        data["X2_test"],
        data["y2_test"]

    )

    Visualizer.plot_confusion_matrix(

        results2["confusion_matrix"],

        "Stage 2 Confusion Matrix"

    )

    Visualizer.plot_roc_curve(

        results2["fpr"],

        results2["tpr"],

        results2["auc"],

        "Stage 2 ROC"

    )

    # ======================================================
    # FINISHED
    # ======================================================

    print_heading("PIPELINE COMPLETED")

    print("Models saved in  : models/")
    print("Plots saved in   : results/")
    print("Training Complete!")

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
