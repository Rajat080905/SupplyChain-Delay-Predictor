"""
config.py

Project configuration file.

Author: Rajat
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

# ==========================================================
# PATHS
# ==========================================================

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"
IMAGE_DIR = ROOT_DIR / "images"
NOTEBOOK_DIR = ROOT_DIR / "notebook"

# ==========================================================
# DATASET
# ==========================================================

DATASET_PATH = DATA_DIR / "dataset.xlsx"

# ==========================================================
# SAVED MODELS
# ==========================================================

ENCODER_PATH = MODEL_DIR / "encoder.pkl"

STAGE1_MODEL_PATH = MODEL_DIR / "stage1_xgboost.pkl"

STAGE2_RF_MODEL_PATH = MODEL_DIR / "stage2_random_forest.pkl"

STAGE2_XGB_MODEL_PATH = MODEL_DIR / "stage2_xgboost.pkl"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# STAGE-1 THRESHOLD
# ==========================================================

STAGE1_THRESHOLD = 0.42

# ==========================================================
# CLASS LABELS
# ==========================================================

STAGE1_CLASSES = [
    "On Time",
    "Delayed"
]

STAGE2_CLASSES = [
    "At Risk",
    "Delayed"
]
