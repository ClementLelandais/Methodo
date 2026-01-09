"""
Utilitaires de pré-traitement des données pour AutoML avec support du format sparse.
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
    """Vérifie si une série contient des données au format sparse (ex: '111:1', '45:0.5')."""
    if series.dtype != object:
        return False
    
    # Échantillonne les premières valeurs non-nulles
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Vérifie si les valeurs correspondent au pattern sparse
    sparse_count = 0
    for val in sample:
        if isinstance(val, str) and ':' in val:
            sparse_count += 1
    
    # Si >50% des valeurs sont au format sparse, considère comme sparse
    return sparse_count / len(sample) > 0.5


def _parse_sparse_column(series: pd.Series) -> pd.Series:
    """Parse une colonne au format sparse (ex: '111:1') vers numérique."""
    def parse_value(val):
        if pd.isna(val) or val == '':
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if ':' in val:
                # Format: 'index:valeur' -> extraire la valeur
                try:
                    return float(val.split(':')[1])
                except (IndexError, ValueError):
                    return np.nan
        return np.nan
    
    return series.apply(parse_value)


class DataPreprocessor:
    """Pré-traite les features numériques et catégorielles pour AutoML.
    
    Supporte maintenant la détection et conversion du format sparse.
    """

    def __init__(self, types: Iterable[str]) -> None:
        self.types: List[str] = list(types)
        self.numeric_features = []
        self.categorical_features = []
        self.sparse_features = []
        self.preprocessor = None
        self._sparse_detected = False

    def _detect_sparse_features(self, X: pd.DataFrame) -> List[int]:
        """Détecte quelles features sont au format sparse."""
        sparse_cols = []
        for i, col in enumerate(X.columns):
            if _is_sparse_format(X[col]):
                sparse_cols.append(i)
        return sparse_cols

    def _convert_sparse_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convertit les colonnes au format sparse vers numérique."""
        X_converted = X.copy()
        
        for i in range(len(X_converted.columns)):
            col = X_converted.iloc[:, i]
            if _is_sparse_format(col):
                X_converted.iloc[:, i] = _parse_sparse_column(col)
        
        return X_converted

    def _categorize_features(self, types: List[str], X: Optional[pd.DataFrame] = None) -> None:
        """Catégorise les features selon les types et inspection des données."""
        if X is not None:
            # Détecte d'abord les features sparse
            self.sparse_features = self._detect_sparse_features(X)
            if len(self.sparse_features) > 0:
                self._sparse_detected = True
        
        # Catégorise selon les types, en excluant les features sparse
        for i, t in enumerate(types):
            if i in self.sparse_features:
                # Les features sparse sont traitées comme numériques après conversion
                self.numeric_features.append(i)
            elif t.lower() == "categorical":
                self.categorical_features.append(i)
            else:
                self.numeric_features.append(i)

    def _build_preprocessor(self) -> ColumnTransformer:
        """Construit le ColumnTransformer qui impute et scale/encode les features."""
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
            # Fallback: traite tout comme numérique
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
        """Fit le préprocesseur et transforme les données."""
        # Convertit le format sparse si détecté
        if self._sparse_detected or any(_is_sparse_format(X[col]) for col in X.columns):
            X = self._convert_sparse_to_numeric(X)
            self._sparse_detected = True
        
        # Construit le préprocesseur si pas encore fait
        if self.preprocessor is None:
            self._categorize_features(self.types, X)
            self.preprocessor = self._build_preprocessor()
        
        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforme les données avec le préprocesseur fitté."""
        if self.preprocessor is None:
            raise RuntimeError("Préprocesseur non fitté. Appelez fit_transform d'abord.")
        
        # Convertit le format sparse si détecté lors du fit
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
        """Divise les données en ensembles d'entraînement et validation avec stratification optionnelle.
        
        Cette méthode gère maintenant la conversion du format sparse avant division.
        """
        # Convertit X en DataFrame si nécessaire
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Détecte et convertit le format sparse
        self._categorize_features(self.types, X)
        if self._sparse_detected:
            X = self._convert_sparse_to_numeric(X)
        
        # Construit le préprocesseur
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor()
        
        # Prépare la stratification
        stratify = None
        y_arr = np.asarray(y)
        if task_type == "classification":
            if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
                stratify = y_arr.ravel()
        
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
