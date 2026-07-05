"""
utils.py

Utility functions for the SupplyChain Delay Predictor.

Author: Rajat
"""

import random
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from config import (
    MODEL_DIR,
    RESULT_DIR,
    IMAGE_DIR,
    RANDOM_STATE,
)


# ==========================================================
# RANDOM SEED
# ==========================================================

def set_seed(seed=RANDOM_STATE):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)


# ==========================================================
# CREATE PROJECT DIRECTORIES
# ==========================================================

def create_directories():
    """
    Create required project directories.
    """

    directories = [
        MODEL_DIR,
        RESULT_DIR,
        IMAGE_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(model, filepath):
    """
    Save trained model.
    """

    joblib.dump(model, filepath)
    print(f"Model saved → {filepath}")


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(filepath):
    """
    Load trained model.
    """

    print(f"Loading model: {filepath}")
    return joblib.load(filepath)


# ==========================================================
# SAVE DATAFRAME
# ==========================================================

def save_dataframe(df, filepath):
    """
    Save DataFrame as CSV.
    """

    df.to_csv(filepath, index=False)
    print(f"CSV saved → {filepath}")


# ==========================================================
# SAVE METRICS
# ==========================================================

def save_metrics(metrics, filepath):
    """
    Save evaluation metrics.
    """

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(filepath, index=False)

    print(f"Metrics saved → {filepath}")


# ==========================================================
# PRINT HEADER
# ==========================================================

def print_heading(title):
    """
    Print formatted section heading.
    """

    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


# ==========================================================
# CLASS LABELS
# ==========================================================

STAGE1_LABELS = {
    0: "On Time",
    1: "Delayed",
}

STAGE2_LABELS = {
    0: "At Risk",
    1: "Delayed",
}


# ==========================================================
# DISPLAY MODEL SUMMARY
# ==========================================================

def model_summary(model):
    """
    Print model information.
    """

    print("\nModel Summary")
    print("-" * 40)
    print(type(model).__name__)
    print(model)
    print("-" * 40)


# ==========================================================
# DISPLAY DATASET INFO
# ==========================================================

def dataset_summary(df):
    """
    Print dataset summary.
    """

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print("\nMissing Values")
    print(df.isnull().sum())
    print("-" * 40)
