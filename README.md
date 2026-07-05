# SupplyChain Delay Predictor

A two-stage machine learning pipeline that predicts delivery delays in a supply chain, then further classifies delayed shipments by risk level.

## How it works

The model works in two stages:

1. **Stage 1 — On Time vs. Delayed** (XGBoost)
   Predicts whether a shipment will be delayed at all.

2. **Stage 2 — At Risk vs. Delayed** (Random Forest, or XGBoost)
   For shipments flagged as delayed in Stage 1, predicts whether they're merely "at risk" or fully "delayed."

This cascaded approach lets the second model focus specifically on distinguishing severity among shipments already identified as problematic, rather than trying to solve both problems at once.

## Project structure

```
SupplyChain-Delay-Predictor/
├── config.py              # Paths, random seed, thresholds, class labels
├── main.py                # Runs the full pipeline end-to-end
├── requirements.txt       # Python dependencies
├── data/
│   └── dataset.xlsx       # Input dataset (add your own — see below)
├── models/                # Saved model + encoder artifacts (.pkl)
├── results/               # Saved confusion matrix & ROC curve plots
├── notebook/
│   └── final.ipynb        # Original exploratory notebook
└── src/
    ├── preprocessing.py   # Missing values, feature engineering, targets
    ├── data_split.py      # Train/test split + one-hot encoding
    ├── train_stage1.py    # Stage-1 XGBoost trainer
    ├── train_stage2.py    # Stage-2 Random Forest / XGBoost trainer
    ├── evaluation.py      # Accuracy, classification report, ROC/AUC
    ├── prediction.py      # Inference on new shipment data
    ├── visualization.py   # Confusion matrix & ROC curve plotting
    └── utils.py            # Seeding, directory setup, save/load helpers
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Place your dataset at `data/dataset.xlsx`. It should contain (at minimum) the following columns, which the feature engineering step relies on:

| Column | Description |
|---|---|
| `time_buffer_hours` | Difference between expected and actual delivery time (used to build both targets — removed after, to avoid leakage) |
| `distance_km` | Shipment distance |
| `order_volume` | Order size |
| `delivery_cost` | Cost of delivery |
| `traffic_index` | Traffic congestion measure |
| `hub_dwell_minutes` | Time spent at hub |
| `driver_delay_rate` | Historical driver delay rate |
| `vehicle_reliability` | Vehicle reliability score |

Any additional categorical columns (e.g. region, carrier) are automatically one-hot encoded.

## Usage

Run the full pipeline (preprocessing → split → train Stage 1 → train Stage 2 → evaluate → save):

```bash
python main.py
```

This will:
- Save trained models and the fitted encoder to `models/`
- Save confusion matrix and ROC curve plots to `results/`
- Print accuracy and classification reports for both stages to the console

### Running predictions on new data

```python
from src.prediction import Predictor
import pandas as pd

predictor = Predictor()
new_data = pd.read_excel("path/to/new_shipments.xlsx")
results = predictor.predict(new_data)
print(results[["Prediction", "Confidence"]])
```

## Configuration

Key settings live in `config.py`:
- `RANDOM_STATE` — reproducibility seed (default `42`)
- `STAGE1_THRESHOLD` — probability threshold for Stage-1 classification (default `0.42`)
- Model/encoder save paths

## Notes

- Leakage-prone columns (e.g. `delivery_time_hours`, `expected_delivery_timestamp`, `delivery_status`) are automatically dropped before training.
- Stage 2 defaults to Random Forest; pass `model="xgboost"` to `Stage2Trainer` to use XGBoost instead.
