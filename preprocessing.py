import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, types):
        self.types = types
        self.numeric_features = [i for i, t in enumerate(types) if t.lower() == "numerical"]
        self.categorical_features = [i for i, t in enumerate(types) if t.lower() == "categorical"]
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        """SPARSE TOTAL: MaxAbsScaler + sparse OHE."""
        # MAXABS: UNIQUEMENT scaling, AUCUN centering → sparse OK
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # 0 pour sparse
            ("scaler", MaxAbsScaler()),  # max(|X|) → sparse parfait
        ])

        # CATEGORICAL sparse
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            sparse_threshold=0.1,  # Sparse output garanti
            n_jobs=1,  # Pas de parallel crash
        )
        return preprocessor

    def split(self, X, y, test_size=0.2, seed=42):
        """Split simple."""
        return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=False)

    def fit_transform(self, X_train): return self.preprocessor.fit_transform(X_train)
    def transform(self, X_val): return self.preprocessor.transform(X_val)
