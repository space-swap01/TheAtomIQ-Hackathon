"""
Customer Subscription Prediction and Ranking Module (Stage 1).

Responsibilities:
- Select the best probabilistic model based on CV evaluation (PR-AUC, ROC-AUC, Brier score, calibration)
- Retrain the selected model on all 300 Stage1_Train records
- Generate continuous probability predictions for all 100 Stage1_Test records
- Rank test customers descending by probability with tie-breaking on customer_id ascending
- Flag top 10 candidates for Stage 2
- Export stage1_predictions.csv and model_comparison.csv
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone

from .utils import setup_logger, get_project_root

logger = setup_logger("CustomerPrediction")

def select_best_model(comparison_df: pd.DataFrame, models_dict: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Select the optimal model according to specified prioritization:
    1. PR-AUC (key for imbalanced ranking)
    2. ROC-AUC
    3. Low Brier Score and Log Loss
    """
    # Create composite score prioritizing PR-AUC and ROC-AUC while penalizing Brier
    # Higher is better
    df = comparison_df.copy()
    # Normalize metrics to 0-1 range for scoring
    pr_rank = df['PR-AUC'].rank(ascending=True)
    roc_rank = df['ROC-AUC'].rank(ascending=True)
    brier_rank = df['Brier Score'].rank(ascending=False) # lower Brier is better
    
    df['SelectionScore'] = (2.0 * pr_rank) + (1.5 * roc_rank) + (1.0 * brier_rank)
    best_row = df.sort_values(by='SelectionScore', ascending=False).iloc[0]
    best_name = best_row['Model']
    
    logger.info(f"Selected Best Model: '{best_name}' (PR-AUC: {best_row['PR-AUC']}, ROC-AUC: {best_row['ROC-AUC']}, Brier: {best_row['Brier Score']})")
    return best_name, models_dict[best_name]

def generate_stage1_predictions(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    test_customer_ids: pd.Series,
    output_dir: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit finalized model on all 300 train samples, predict on 100 test samples,
    and rank candidates with deterministic tie-breaking.
    """
    if output_dir is None:
        output_dir = get_project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fitting finalized model ({type(model).__name__}) on full training set (300 records)...")
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    # Predict continuous probabilities
    test_probs = fitted_model.predict_proba(X_test)[:, 1]

    # Build predictions DataFrame
    pred_df = pd.DataFrame({
        "customer_id": test_customer_ids.values,
        "predicted_probability": test_probs
    })

    # Sort descending by probability, tie-break by customer_id ascending alphanumeric
    pred_df = pred_df.sort_values(
        by=["predicted_probability", "customer_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Rank 1 to 100
    pred_df["rank"] = np.arange(1, len(pred_df) + 1)
    # top_10_flag
    pred_df["top_10_flag"] = pred_df["rank"] <= 10

    # Ensure required column order
    pred_df = pred_df[["customer_id", "predicted_probability", "rank", "top_10_flag"]]

    # Save to CSV
    csv_path = output_dir / "stage1_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Saved Stage 1 predictions to: {csv_path}")

    # Top 10 preview
    top_10_df = pred_df[pred_df["top_10_flag"]].copy()
    logger.info("Top 10 Stage 1 Candidate Customers:")
    for _, row in top_10_df.iterrows():
        logger.info(f"Rank {row['rank']:2d}: Customer {row['customer_id']} | Prob: {row['predicted_probability']:.5f}")

    return pred_df, top_10_df
