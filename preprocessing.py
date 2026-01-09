"""
Data preprocessing utilities for AutoML with sparse format support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Iterable, List, Optional, Tuple, Union

__all__ = ["DataPreprocessor"]


def _is_sparse_format(series: pd.Series) -> bool:
    """Check if a series contains sparse format data (e.g., '111:1', '45:0.5')."""
    if series.dtype != object:
        return False
    
    # Sample first non-null values
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Check if values match sparse format pattern
    sparse_count = 0
    for val in sample:
        if isinstance(val, str) and ':' in val:
            sparse_count += 1
    
    # If >50% of values are sparse format, consider it sparse
    return sparse_count / len(sample) > 0.5


def _parse_sparse_column(series: pd.Series) -> pd.Series:
    """Parse a sparse format column (e.g., '111:1') to numeric."""
    def parse_value(val):
        if pd.isna(val) or val == '':
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if ':' in val:
                # Format: 'index:value' -> extract value
                try:
                    return float(val.split(':')[1])
                except (IndexError, ValueError):
                    return np.nan
        return np.nan
    
    return series.apply(parse_value)


class DataPreprocessor:
    """Preprocess numerical and categorical features for AutoML.
    
    Now supports sparse format detection and conversion.
    """

    def __init__(self, types: Iterable[str]) -> None:
        self.types: List[str] = list(types)
        self.numeric_features = []
        self.categorical_features = []
        self.sparse_features = []
        self.preprocessor = None
        self._sparse_detected = False

    def _detect_sparse_features(self, X: pd.DataFrame) -> List[int]:
        """Detect which features are in sparse format."""
        sparse_cols = []
        for i, col in enumerate(X.columns):
            if _is_sparse_format(X[col]):
                sparse_cols.append(i)
        return sparse_cols

    def _convert_sparse_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert sparse format columns to numeric."""
        X_converted = X.copy()
        
        for i in range(len(X_converted.columns)):
            col = X_converted.iloc[:, i]
            if _is_sparse_format(col):
                X_converted.iloc[:, i] = _parse_sparse_column(col)
        
        return X_converted

    def _categorize_features(self, types: List[str], X: Optional[pd.DataFrame] = None) -> None:
        """Categorize features based on types and data inspection."""
        if X is not None:
            # Detect sparse features first
            self.sparse_features = self._detect_sparse_features(X)
            if len(self.sparse_features) > 0:
                self._sparse_detected = True
        
        # Categorize based on types, excluding sparse features
        for i, t in enumerate(types):
            if i in self.sparse_features:
                # Sparse features are treated as numerical after conversion
                self.numeric_features.append(i)
            elif t.lower() == "categorical":
                self.categorical_features.append(i)
            else:
                self.numeric_features.append(i)

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the ColumnTransformer that imputes and scales/encodes features."""
        transformers = []
        
        if len(self.numeric_features) > 0:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            transformers.append(("num", numeric_transformer, self.numeric_features))
        
        if len(self.categorical_features) > 0:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat", categorical_transformer, self.categorical_features))
        
        if len(transformers) == 0:
            # Fallback: treat all as numeric
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            transformers.append(("num", numeric_transformer, list(range(len(self.types)))))
        
        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,
        )
        return transformer

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data."""
        # Convert sparse format if detected
        if self._sparse_detected or any(_is_sparse_format(X[col]) for col in X.columns):
            X = self._convert_sparse_to_numeric(X)
            self._sparse_detected = True
        
        # Build preprocessor if not done yet
        if self.preprocessor is None:
            self._categorize_features(self.types, X)
            self.preprocessor = self._build_preprocessor()
        
        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        # Convert sparse format if was detected during fit
        if self._sparse_detected:
            X = self._convert_sparse_to_numeric(X)
        
        return self.preprocessor.transform(X)

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        test_size: float = 0.2,
        seed: int = 42,
        task_type: Optional[str] = None,
    ) -> Tuple:
        """Split data into train and validation sets with optional stratification.
        
        This method now handles sparse format conversion before splitting.
        """
        # Convert X to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Detect and convert sparse format
        self._categorize_features(self.types, X)
        if self._sparse_detected:
            X = self._convert_sparse_to_numeric(X)
        
        # Build preprocessor
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor()
        
        # Prepare stratification
        stratify = None
        y_arr = np.asarray(y)
        if task_type == "classification":
            if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
                stratify = y_arr.ravel()
        
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
