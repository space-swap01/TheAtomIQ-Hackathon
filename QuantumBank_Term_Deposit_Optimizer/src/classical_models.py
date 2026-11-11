"""
Classical Machine Learning Models and Cross-Validation Module.

Implements and evaluates:
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Support Vector Machine (SVC with probability estimation)
5. Gradient Boosting Classifier

Computes:
- ROC-AUC, PR-AUC, Precision, Recall, F1-Score
- Brier Score, Log Loss, Calibration Analysis (ECE)
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from .utils import RANDOM_SEED, setup_logger

logger = setup_logger("ClassicalModels")

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1]) if i < n_bins - 1 else (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

class ClassicalModelTrainer:
    """Manages classical probabilistic models, cross-validation, and metrics."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models = self._init_models()

    def _init_models(self) -> Dict[str, Any]:
        """Initialize all 5 classical models with reproducible settings."""
        # Train class ratio: 35 positive / 265 negative (~ 1 : 7.57)
        scale_pos_weight = 265.0 / 35.0

        models = {
            "Logistic Regression": LogisticRegression(
                class_weight='balanced',
                C=0.5,
                max_iter=1000,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=4,
                class_weight='balanced',
                random_state=self.random_state
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=self.random_state
            ),
            "Support Vector Machine": SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.random_state
            )
        }
        return models

    def evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Perform 5-fold Stratified Cross-Validation on all models.
        Returns:
            comparison_df: metrics per model
            cv_results_store: detailed OOF predictions and curves per model
        """
        comparison_records = []
        cv_results_store = {}

        for name, model in self.models.items():
            oof_preds = np.zeros(len(y))
            oof_probs = np.zeros(len(y))
            fold_metrics = []

            for train_idx, val_idx in cv_folds:
                X_train_f, y_train_f = X[train_idx], y[train_idx]
                X_val_f, y_val_f = X[val_idx], y[val_idx]

                # Clone or re-instantiate model to ensure fresh fold
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train_f, y_train_f)

                prob_val = fold_model.predict_proba(X_val_f)[:, 1]
                pred_val = (prob_val >= 0.5).astype(int)

                oof_probs[val_idx] = prob_val
                oof_preds[val_idx] = pred_val

            # Compute out-of-fold global metrics
            roc_auc = roc_auc_score(y, oof_probs)
            pr_auc = average_precision_score(y, oof_probs)
            brier = brier_score_loss(y, oof_probs)
            # Clip for safe log_loss computation
            clipped_probs = np.clip(oof_probs, 1e-15, 1 - 1e-15)
            logloss = log_loss(y, clipped_probs)
            ece = calculate_ece(y, oof_probs)
            
            # Binary classification metrics at standard 0.5 cutoff
            precision = precision_score(y, oof_preds, zero_division=0)
            recall = recall_score(y, oof_preds, zero_division=0)
            f1 = f1_score(y, oof_preds, zero_division=0)

            # Precision@10 and Recall@10 on OOF predictions
            top_10_idx = np.argsort(oof_probs)[::-1][:10]
            top_10_actual = y[top_10_idx]
            precision_at_10 = np.sum(top_10_actual) / 10.0
            recall_at_10 = np.sum(top_10_actual) / np.sum(y)

            # Curves for plotting
            fpr, tpr, _ = roc_curve(y, oof_probs)
            prec_c, rec_c, _ = precision_recall_curve(y, oof_probs)
            prob_true, prob_pred = calibration_curve(y, oof_probs, n_bins=8, strategy='uniform')
            cm = confusion_matrix(y, oof_preds)

            record = {
                "Model": name,
                "Type": "Classical",
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Brier Score": round(brier, 4),
                "Log Loss": round(logloss, 4),
                "ECE": round(ece, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "Precision@10": round(precision_at_10, 4),
                "Recall@10": round(recall_at_10, 4)
            }
            comparison_records.append(record)

            cv_results_store[name] = {
                "oof_probs": oof_probs,
                "oof_preds": oof_preds,
                "roc_curve": (fpr, tpr),
                "pr_curve": (prec_c, rec_c),
                "calibration_curve": (prob_true, prob_pred),
                "confusion_matrix": cm,
                "metrics": record
            }

            logger.info(f"Evaluated {name:25s} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")

        comparison_df = pd.DataFrame(comparison_records)
        return comparison_df, cv_results_store
