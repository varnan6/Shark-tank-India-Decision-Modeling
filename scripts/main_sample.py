"""
main.py — Shark Tank India Preprocessing Pipeline
===================================================

Orchestrates the full preprocessing pipeline from raw Kaggle download
to train/test-ready feature matrices.

Pipeline Steps:
    STEP 0 → Load Raw Dataset              (dataloader.py)
    STEP 1 → Preprocessing                 (preprocess.py)
    STEP 2 → Feature Engineering            (feature_engineering.py)
    STEP 3 → Encoding                      (encoding.py)
    STEP 4 → Scaling                       (scaling.py)
    STEP 5 → Target Separation
    STEP 6 → Train/Test Split

Usage:
    python scripts/main.py
    python scripts/main.py --test-size 0.25
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Module Imports
# ─────────────────────────────────────────────────────────────────
from dataloader import load_dataset

# TODO: Uncomment as each module is built
# from preprocess import preprocess_data
# from feature_engineering import engineer_features
# from encoding import encode_features
# from scaling import scale_features


# ═════════════════════════════════════════════════════════════════
# STEP 0 → Load Raw Dataset
# ═════════════════════════════════════════════════════════════════
def step_0_load_data() -> pd.DataFrame:
    """Download from Kaggle (or use cached CSV) and return raw DataFrame."""
    logger.info("=" * 70)
    logger.info("STEP 0 → Loading Raw Dataset")
    logger.info("=" * 70)

    df_raw = load_dataset()
    logger.info(f"✅ Raw dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns\n")
    return df_raw


# ═════════════════════════════════════════════════════════════════
# STEP 1 → Preprocessing (preprocess.py)
# ═════════════════════════════════════════════════════════════════
def step_1_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run the unified preprocessing module.

    Internally handles:
      - Person 3: Shark data preprocessing
      - Person 2: Financial + Deal preprocessing (uses Person 3 output)
      - Person 1: Context + Pitcher preprocessing (independent)
      - Merge all into a single clean DataFrame
    """
    logger.info("=" * 70)
    logger.info("STEP 1 → Preprocessing")
    logger.info("=" * 70)

    # TODO: Replace with actual call
    # df_preprocessed = preprocess_data(df_raw)
    df_preprocessed = df_raw.copy()  # placeholder

    logger.info(f"✅ Preprocessing complete: {df_preprocessed.shape}\n")
    return df_preprocessed


# ═════════════════════════════════════════════════════════════════
# STEP 2 → Feature Engineering
# ═════════════════════════════════════════════════════════════════
def step_2_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the preprocessed dataset.
      - Interaction features
      - Aggregated features
      - Domain-specific derived columns
    """
    logger.info("=" * 70)
    logger.info("STEP 2 → Feature Engineering")
    logger.info("=" * 70)

    # TODO: Replace with actual call
    # df_engineered = engineer_features(df)
    df_engineered = df.copy()  # placeholder

    logger.info(f"✅ Feature engineering complete: {df_engineered.shape}\n")
    return df_engineered


# ═════════════════════════════════════════════════════════════════
# STEP 3 → Encoding
# ═════════════════════════════════════════════════════════════════
def step_3_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.
      - Label encoding / One-hot encoding / Target encoding
    """
    logger.info("=" * 70)
    logger.info("STEP 3 → Encoding")
    logger.info("=" * 70)

    # TODO: Replace with actual call
    # df_encoded = encode_features(df)
    df_encoded = df.copy()  # placeholder

    logger.info(f"✅ Encoding complete: {df_encoded.shape}\n")
    return df_encoded


# ═════════════════════════════════════════════════════════════════
# STEP 4 → Scaling
# ═════════════════════════════════════════════════════════════════
def step_4_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features.
      - StandardScaler / MinMaxScaler / RobustScaler
    """
    logger.info("=" * 70)
    logger.info("STEP 4 → Scaling")
    logger.info("=" * 70)

    # TODO: Replace with actual call
    # df_scaled = scale_features(df)
    df_scaled = df.copy()  # placeholder

    logger.info(f"✅ Scaling complete: {df_scaled.shape}\n")
    return df_scaled


# ═════════════════════════════════════════════════════════════════
# STEP 5 → Target Separation
# ═════════════════════════════════════════════════════════════════
def step_5_separate_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Separate feature matrix (X) from target variables (y).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix without target columns.
    targets : dict
        Dictionary of target Series:
          - 'deal_prediction'     → Received Offer (binary classification)
          - 'valuation'           → Deal Valuation (regression)
          - 'shark_participation' → Multi-label (one per shark)
    """
    logger.info("=" * 70)
    logger.info("STEP 5 → Target Separation")
    logger.info("=" * 70)

    targets = {}

    # Task 1: Deal / No-Deal (Classification)
    if "Received Offer" in df.columns:
        targets["deal_prediction"] = df["Received Offer"]

    # Task 2: Deal Valuation (Regression)
    if "Deal Valuation" in df.columns:
        targets["valuation"] = df["Deal Valuation"]

    # Task 3: Shark Participation (Multi-label Classification)
    shark_cols = [col for col in df.columns if col.endswith("Investment Amount")]
    if shark_cols:
        shark_labels = (df[shark_cols] > 0).astype(int)
        shark_labels.columns = [col.replace(" Investment Amount", "") for col in shark_cols]
        targets["shark_participation"] = shark_labels

    # Drop target columns from feature matrix
    cols_to_drop = []
    if "Received Offer" in df.columns:
        cols_to_drop.append("Received Offer")
    if "Deal Valuation" in df.columns:
        cols_to_drop.append("Deal Valuation")
    cols_to_drop.extend(shark_cols)

    X = df.drop(columns=cols_to_drop, errors="ignore")

    logger.info(f"✅ Feature matrix X: {X.shape}")
    logger.info(f"   Targets: {list(targets.keys())}\n")
    return X, targets


# ═════════════════════════════════════════════════════════════════
# STEP 6 → Train/Test Split
# ═════════════════════════════════════════════════════════════════
def step_6_split(
    X: pd.DataFrame,
    targets: Dict[str, pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Split data into train and test sets.

    Returns
    -------
    dict with keys:
        'X_train', 'X_test',
        'y_train' (dict of targets), 'y_test' (dict of targets)
    """
    from sklearn.model_selection import train_test_split

    logger.info("=" * 70)
    logger.info("STEP 6 → Train/Test Split")
    logger.info("=" * 70)

    stratify = targets.get("deal_prediction", None)

    X_train, X_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    y_train = {}
    y_test = {}
    for task_name, y in targets.items():
        y_train[task_name] = y.loc[X_train.index]
        y_test[task_name] = y.loc[X_test.index]

    logger.info(f"✅ Train set: {X_train.shape[0]} samples")
    logger.info(f"   Test set:  {X_test.shape[0]} samples")
    logger.info(f"   Split ratio: {1-test_size:.0%} / {test_size:.0%}\n")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ═════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═════════════════════════════════════════════════════════════════
def run_pipeline(test_size: float = 0.2) -> Dict:
    """
    Execute the full preprocessing pipeline end-to-end.

    Each step receives the output of the previous step:
        df_raw → df_preprocessed → df_engineered → df_encoded → df_scaled → (X, targets) → split

    Returns
    -------
    dict
        Contains X_train, X_test, y_train, y_test ready for modeling.
    """
    logger.info("🚀 STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 70 + "\n")

    # STEP 0: Load raw data from Kaggle
    df_raw = step_0_load_data()

    # STEP 1: Preprocess (shark + financial + context → merged)
    df_preprocessed = step_1_preprocess(df_raw)

    # STEP 2: Feature engineering
    df_engineered = step_2_feature_engineering(df_preprocessed)

    # STEP 3: Encoding
    df_encoded = step_3_encode(df_engineered)

    # STEP 4: Scaling
    df_scaled = step_4_scale(df_encoded)

    # STEP 5: Target separation
    X, targets = step_5_separate_targets(df_scaled)

    # STEP 6: Train/test split
    split_data = step_6_split(X, targets, test_size=test_size)

    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  X_train: {split_data['X_train'].shape}")
    logger.info(f"  X_test:  {split_data['X_test'].shape}")
    logger.info(f"  Tasks:   {list(split_data['y_train'].keys())}")
    logger.info("=" * 70 + "\n")

    return split_data


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Shark Tank India preprocessing pipeline"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    args = parser.parse_args()

    split_data = run_pipeline(test_size=args.test_size)
