"""
Data Preprocessing Module for QuantumBank Term Deposit Campaign Pipeline.

Responsibilities:
- Loading raw sheets from bank_campaign_participant_release.xlsx
- Validating schema, columns, missing values, duplicates, and constraints
- Strictly fitting transformations (StandardScaler, OneHotEncoder) on training data only
- Creating PCA reduced features for Quantum Machine Learning (QML)
- Generating Stratified K-Fold splits
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from .utils import RANDOM_SEED, setup_logger, get_project_root

logger = setup_logger("DataPreprocessing")

NUMERICAL_COLS = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
FORBIDDEN_COLS = ['duration']

class CampaignDataPreprocessor:
    """End-to-end data loader and preprocessor adhering to hackathon rules."""

    def __init__(self, excel_path: Path = None, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        if excel_path is None:
            self.excel_path = get_project_root() / "data" / "bank_campaign_participant_release.xlsx"
        else:
            self.excel_path = Path(excel_path)
            
        self.column_transformer = None
        self.feature_names_out: List[str] = []
        self.pca_transformer = None
        self.qml_scaler = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant sheets from the Excel file."""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.excel_path}")
            
        logger.info(f"Loading dataset from: {self.excel_path}")
        xl = pd.ExcelFile(self.excel_path)
        
        sheets = {
            'readme': xl.parse('README'),
            'train': xl.parse('Stage1_Train'),
            'test': xl.parse('Stage1_Test'),
            'stage2_data': xl.parse('Stage2_Data'),
            'stage2_constraints': xl.parse('Stage2_Constraints')
        }
        
        self._validate_schema(sheets['train'], sheets['test'])
        return sheets

    def _validate_schema(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Perform validation checks on train and test datasets."""
        # Rule: duration must not exist
        for col in FORBIDDEN_COLS:
            if col in train_df.columns or col in test_df.columns:
                raise ValueError(f"Forbidden column '{col}' detected! It must not be used.")
                
        # Rule: customer_id must be present as identifier
        if 'customer_id' not in train_df.columns or 'customer_id' not in test_df.columns:
            raise KeyError("customer_id is required in both train and test sheets.")
            
        # Target check
        if 'y' not in train_df.columns:
            raise KeyError("Target variable 'y' not found in training data.")
            
        # Ensure row counts match specification
        assert len(train_df) == 300, f"Expected 300 train rows, found {len(train_df)}"
        assert len(test_df) == 100, f"Expected 100 test rows, found {len(test_df)}"
        
        logger.info("Schema validation passed: 300 train rows, 100 test rows, no forbidden columns.")

    def fit_transform_train(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Fit preprocessor strictly on training data and return (X_train, y_train, customer_ids).
        """
        customer_ids = train_df['customer_id'].copy()
        
        # Target encoding: 'yes' -> 1, 'no' -> 0
        y_train = (train_df['y'].astype(str).str.strip().str.lower() == 'yes').astype(int).values
        
        # Candidate features
        X_df = train_df.drop(columns=['customer_id', 'y'])
        
        # Preprocessing pipeline
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_COLS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
            ],
            remainder='drop'
        )
        
        X_train = self.column_transformer.fit_transform(X_df)
        self.feature_names_out = self.column_transformer.get_feature_names_out().tolist()
        
        logger.info(f"Preprocessed training data: shape {X_train.shape}, target mean: {y_train.mean():.4f}")
        return X_train, y_train, customer_ids

    def transform_test(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Transform test data using the transformer already fitted on train data.
        """
        if self.column_transformer is None:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming test data.")
            
        customer_ids = test_df['customer_id'].copy()
        X_df = test_df.drop(columns=['customer_id'])
        
        X_test = self.column_transformer.transform(X_df)
        logger.info(f"Preprocessed test data: shape {X_test.shape}")
        return X_test, customer_ids

    def fit_transform_qml_features(self, X_train: np.ndarray, n_components: int = 4) -> np.ndarray:
        """
        Fit PCA on X_train to reduce dimensionality for quantum circuits (e.g. 4-6 qubits),
        and scale features to [0, pi] for angle embedding.
        """
        self.pca_transformer = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca_transformer.fit_transform(X_train)
        
        self.qml_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_qml_train = self.qml_scaler.fit_transform(X_pca)
        
        explained_var = np.sum(self.pca_transformer.explained_variance_ratio_)
        logger.info(f"QML PCA ({n_components} components) explains {explained_var*100:.2f}% of variance.")
        return X_qml_train

    def transform_qml_features(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transform test features into QML feature representation using fitted PCA and scaler.
        """
        if self.pca_transformer is None or self.qml_scaler is None:
            raise RuntimeError("QML transformers must be fitted on training data first.")
            
        X_pca = self.pca_transformer.transform(X_test)
        X_qml_test = self.qml_scaler.transform(X_pca)
        return X_qml_test

    def get_stratified_folds(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """Provide StratifiedKFold splitter generator."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return skf.split(X, y)
