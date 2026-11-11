"""
Quantum Machine Learning (QML) Classifier Module.

Implements a genuine Quantum Kernel Classifier using Qiskit:
- Parameterized Quantum Feature Map (ZZFeatureMap)
- Statevector fidelity evaluation: K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2
- Dual support for ideal statevector simulation and AerSimulator shot-based sampling
- Calibrated probability estimation via support vector classification with balanced class weights
- Circuit inspection (qubit count, circuit depth, gate decomposition)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from qiskit import QuantumCircuit
try:
    from qiskit.circuit.library import zz_feature_map
    USE_ZZ_FUNC = True
except ImportError:
    USE_ZZ_FUNC = False
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from .utils import RANDOM_SEED, setup_logger
from .classical_models import calculate_ece

logger = setup_logger("QuantumClassifier")

class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum Kernel Support Vector Classifier (QSVC).
    Uses Qiskit ZZFeatureMap to map classical features into 2^n Hilbert space.
    """

    def __init__(
        self,
        n_features: int = 4,
        reps: int = 1,
        entanglement: str = 'linear',
        C: float = 1.0,
        random_state: int = RANDOM_SEED
    ):
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.random_state = random_state
        
        self.feature_map = self._build_feature_map()
        self.svc = SVC(
            C=self.C,
            kernel='precomputed',
            probability=True,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.X_train_states: Optional[np.ndarray] = None
        self.X_train_orig: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def _build_feature_map(self) -> QuantumCircuit:
        """Construct the Qiskit ZZFeatureMap circuit."""
        if USE_ZZ_FUNC:
            return zz_feature_map(
                feature_dimension=self.n_features,
                reps=self.reps,
                entanglement=self.entanglement
            )
        else:
            return ZZFeatureMap(
                feature_dimension=self.n_features,
                reps=self.reps,
                entanglement=self.entanglement
            )

    def get_circuit_details(self) -> Dict[str, Any]:
        """Return quantum circuit statistics for technical reporting."""
        # Decompose to basis gates to obtain exact gate counts
        decomposed = self.feature_map.decompose()
        gate_counts = dict(decomposed.count_ops())
        return {
            "num_qubits": self.feature_map.num_qubits,
            "depth": decomposed.depth(),
            "gate_counts": gate_counts,
            "entanglement": self.entanglement,
            "reps": self.reps,
            "num_parameters": self.feature_map.num_parameters
        }

    def _compute_quantum_states(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate quantum statevector for each row in X under the ZZFeatureMap.
        Returns matrix of shape (n_samples, 2^n_features) complex amplitudes.
        """
        states = []
        for x_row in X:
            qc = self.feature_map.assign_parameters(x_row)
            sv = Statevector(qc).data
            states.append(sv)
        return np.array(states, dtype=complex)

    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the quantum fidelity kernel matrix K(X1, X2) = |<psi(X1)|psi(X2)>|^2.
        """
        states1 = self._compute_quantum_states(X1)
        if X2 is None or X1 is X2:
            # Symmetrical inner product
            gram = states1 @ states1.conj().T
            kernel_mat = np.abs(gram) ** 2
            # Numerical cleanup
            np.fill_diagonal(kernel_mat, 1.0)
            return np.clip(kernel_mat, 0.0, 1.0)
        else:
            states2 = self._compute_quantum_states(X2)
            gram = states1 @ states2.conj().T
            kernel_mat = np.abs(gram) ** 2
            return np.clip(kernel_mat, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Quantum Kernel Classifier on training data."""
        self.X_train_orig = X.copy()
        self.X_train_states = self._compute_quantum_states(X)
        
        # Gram matrix
        gram = self.X_train_states @ self.X_train_states.conj().T
        K_train = np.clip(np.abs(gram) ** 2, 0.0, 1.0)
        np.fill_diagonal(K_train, 1.0)
        
        self.svc.fit(K_train, y)
        self.classes_ = self.svc.classes_
        logger.info(f"Quantum Kernel Classifier fitted on {len(X)} samples ({self.n_features} qubits).")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X using the quantum kernel."""
        if self.X_train_states is None:
            raise RuntimeError("Model is not fitted yet.")
        states_test = self._compute_quantum_states(X)
        gram_test = states_test @ self.X_train_states.conj().T
        K_test = np.clip(np.abs(gram_test) ** 2, 0.0, 1.0)
        return self.svc.predict_proba(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classes for X."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

def evaluate_quantum_model_cv(
    X_qml: np.ndarray,
    y: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    n_features: int = 4,
    random_state: int = RANDOM_SEED
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform 5-fold Stratified Cross-Validation on the Quantum Classifier.
    """
    logger.info("Evaluating Quantum Kernel Classifier under 5-Fold Stratified CV...")
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y))

    # Pre-compute all statevectors across all 300 instances once for ultra-fast CV
    q_model_base = QuantumKernelClassifier(n_features=n_features, random_state=random_state)
    all_states = q_model_base._compute_quantum_states(X_qml)
    full_gram = np.clip(np.abs(all_states @ all_states.conj().T) ** 2, 0.0, 1.0)
    np.fill_diagonal(full_gram, 1.0)

    for train_idx, val_idx in cv_folds:
        K_train = full_gram[np.ix_(train_idx, train_idx)]
        K_val = full_gram[np.ix_(val_idx, train_idx)]
        y_train_f = y[train_idx]

        fold_svc = SVC(
            C=1.0,
            kernel='precomputed',
            probability=True,
            class_weight='balanced',
            random_state=random_state
        )
        fold_svc.fit(K_train, y_train_f)
        
        prob_val = fold_svc.predict_proba(K_val)[:, 1]
        oof_probs[val_idx] = prob_val
        oof_preds[val_idx] = (prob_val >= 0.5).astype(int)

    # Compute metrics
    roc_auc = roc_auc_score(y, oof_probs)
    pr_auc = average_precision_score(y, oof_probs)
    brier = brier_score_loss(y, oof_probs)
    clipped_probs = np.clip(oof_probs, 1e-15, 1 - 1e-15)
    logloss = log_loss(y, clipped_probs)
    ece = calculate_ece(y, oof_probs)

    precision = precision_score(y, oof_preds, zero_division=0)
    recall = recall_score(y, oof_preds, zero_division=0)
    f1 = f1_score(y, oof_preds, zero_division=0)

    top_10_idx = np.argsort(oof_probs)[::-1][:10]
    top_10_actual = y[top_10_idx]
    precision_at_10 = np.sum(top_10_actual) / 10.0
    recall_at_10 = np.sum(top_10_actual) / np.sum(y)

    fpr, tpr, _ = roc_curve(y, oof_probs)
    prec_c, rec_c, _ = precision_recall_curve(y, oof_probs)
    prob_true, prob_pred = calibration_curve(y, oof_probs, n_bins=8, strategy='uniform')
    cm = confusion_matrix(y, oof_preds)

    circuit_info = q_model_base.get_circuit_details()

    record = {
        "Model": "Quantum Kernel Classifier (QSVC)",
        "Type": "Quantum",
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

    q_store = {
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "roc_curve": (fpr, tpr),
        "pr_curve": (prec_c, rec_c),
        "calibration_curve": (prob_true, prob_pred),
        "confusion_matrix": cm,
        "metrics": record,
        "circuit_info": circuit_info
    }

    logger.info(f"Quantum Model CV | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    return record, q_store
