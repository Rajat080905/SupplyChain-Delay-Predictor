"""
preprocessing.py

Handles:
- Loading dataset
- Missing value treatment
- Feature engineering
- Target creation
- Data cleaning

Author: Rajat
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Load dataset"""
        self.df = pd.read_excel(self.filepath)
        self.df.columns = self.df.columns.str.strip().str.lower()

        print(f"Dataset Loaded Successfully")
        print(f"Rows : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")

        return self.df

    def handle_missing_values(self):
        """Fill missing values"""

        num_cols = self.df.select_dtypes(include=np.number).columns
        cat_cols = self.df.select_dtypes(exclude=np.number).columns

        self.df[num_cols] = self.df[num_cols].fillna(
            self.df[num_cols].median()
        )

        self.df[cat_cols] = self.df[cat_cols].fillna("Unknown")

        return self.df

    def feature_engineering(self):
        """Create additional features"""

        # ----------------------------
        # Pre-dispatch Features
        # ----------------------------

        self.df["sla_time_hours"] = (
            np.abs(self.df["time_buffer_hours"])
            + self.df["distance_km"] / 40
        )

        self.df["order_density"] = (
            self.df["order_volume"]
            / (self.df["distance_km"] + 1)
        )

        self.df["cost_intensity"] = (
            self.df["delivery_cost"]
            / (self.df["order_volume"] + 1)
        )

        # ----------------------------
        # Post-dispatch Features
        # ----------------------------

        self.df["traffic_pressure"] = (
            self.df["traffic_index"]
            * self.df["hub_dwell_minutes"]
        )

        self.df["reliability_risk"] = (
            self.df["driver_delay_rate"]
            * (1 - self.df["vehicle_reliability"])
        )

        self.df["distance_cost_interaction"] = (
            self.df["distance_km"]
            * self.df["delivery_cost"]
        )

        return self.df

    def create_targets(self):
        """Create Stage-1 and Stage-2 targets"""

        tb = self.df["time_buffer_hours"]

        # Stage 1
        self.df["stage1_target"] = np.where(tb < 0, 0, 1)

        # Stage 2
        self.df["stage2_target"] = np.where(tb <= 3, 0, 1)

        return self.df

    def remove_leakage(self):
        """Remove data leakage columns"""

        leakage_columns = [
            "time_buffer_hours",
            "delivery_time_hours",
            "expected_time_hours",
            "actual_delivery_timestamp",
            "expected_delivery_timestamp",
            "delivery_status",
            "delivery_id",
        ]

        self.df.drop(
            columns=leakage_columns,
            errors="ignore",
            inplace=True,
        )

        return self.df

    def preprocess(self):
        """Complete preprocessing pipeline"""

        self.load_data()
        self.handle_missing_values()
        self.feature_engineering()
        self.create_targets()
        self.remove_leakage()

        return self.df
