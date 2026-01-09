from __future__ import annotations

import time
import joblib
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
    Ridge,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import resample

from dataio import load_dataset
from preprocessing import DataPreprocessor

__all__ = ["AutoML"]


class AutoML:
    """
    AutoML automatique pour classification et régression multi-output.
    Sélectionne automatiquement le meilleur modèle parmi plusieurs algorithmes
    avec pré-traitement des données et validation croisée.
    """

    def __init__(
        self,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
        time_budget_sec: Optional[int] = None,
        cv_folds: int = 1,
    ) -> None:
        """
        Initialise l'AutoML.

        Args:
            task_type: 'classification' ou 'regression' (auto-détecté si None)
            test_size: Proportion pour le split train/test (si cv_folds=1)
            random_state: Seed pour reproductibilité
            verbose: Affichage des logs
            time_budget_sec: Budget temps total en secondes
            cv_folds: Nombre de folds pour validation croisée (1 = train/test split)
        """
        # Paramètres de configuration
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.time_budget_sec = time_budget_sec
        self.cv_folds = max(1, int(cv_folds))
        
        # État du modèle (initialisé à None)
        self.preprocessor: Optional[DataPreprocessor] = None
        self.best_model_: Optional[Pipeline] = None
        self.best_model_name_: Optional[str] = None
        self.best_score_: Optional[float] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.models_results_: list[Dict[str, object]] = []
        
        # Données complètes stockées pour refit
        self._X_full: Optional[pd.DataFrame] = None
        self._y_full: Optional[np.ndarray] = None

    def _log(self, msg: str) -> None:
        """Affiche un message si verbose=True."""
        if self.verbose:
            print(msg, flush=True)

    def _log_section(self, title: str) -> None:
        """Affiche un titre de section avec bordure."""
        if not self.verbose:
            return
        bar = "=" * (len(title) + 4)
        print(f"\n{bar}\n  {title}\n{bar}", flush=True)

    def _infer_task_type(self, y_arr: np.ndarray) -> str:
        """
        Détecte automatiquement le type de tâche (classification/régression).
        
        Logique:
        - Si toutes les valeurs sont des entiers ET <= 50 classes uniques 
          ET < 50% des échantillons -> classification
        - Sinon -> régression
        """
        col0 = y_arr[:, 0] if (y_arr.ndim > 1) else y_arr
        col0 = col0[~pd.isna(col0)]
        if col0.size == 0:
            return "regression"
        
        uniques = np.unique(col0)
        all_int = np.all(np.equal(np.mod(uniques, 1), 0))
        
        if all_int and len(uniques) <= 50 and len(uniques) < len(col0) * 0.5:
            return "classification"
        return "regression"

    def _get_models(self, task_type: str, n_outputs: int, n_samples: int, n_features: int) -> Dict[str, object]:
        """
        Sélectionne les modèles selon la taille des données et le type de tâche.
        
        Adaptation automatique:
        - is_large: >10k échantillons ou >500 features
        - is_very_large: >50k échantillons ou >1000 features  
        - is_multioutput_large: >50 outputs
        
        Réduit la complexité (n_estimators, max_iter, max_depth) pour gros datasets.
        """
        is_large = n_samples > 10000 or n_features > 500
        is_very_large = n_samples > 50000 or n_features > 1000
        is_multioutput_large = n_outputs > 50 
        
        if task_type == "classification":
            base: Dict[str, object] = {}
            
            # SGD rapide pour gros datasets
            if not is_multioutput_large:
                base["SGD_logloss"] = SGDClassifier(
                    loss="log_loss", 
                    alpha=1e-4, 
                    max_iter=500 if is_large else 1000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=1,
                    verbose=0
                )
            
            # LogisticRegression (désactivée sur très gros datasets)
            if not is_very_large and not is_multioutput_large:
                base["LogisticRegression"] = LogisticRegression(
                    max_iter=500 if is_large else 1000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                    tol=1e-3,
                    verbose=0
                )
            
            # RandomForest adapté à la taille
            n_estimators = 50 if is_multioutput_large else (100 if is_large else 200)
            max_depth = 15 if is_multioutput_large else (20 if is_large else None)
            base["RandomForestClassifier"] = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=10 if is_large else 2,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            # HistGradientBoosting (rapide et performant)
            max_iter_gb = 50 if is_multioutput_large else (100 if is_large else 200)
            base["HistGBClassifier"] = HistGradientBoostingClassifier(
                max_iter=max_iter_gb,
                max_depth=8 if is_large else 10,
                random_state=self.random_state,
                verbose=0
            )
            
            # Wrapper MultiOutput pour multi-label
            if n_outputs > 1:
                return {f"MultiOutput_{k}": MultiOutputClassifier(v, n_jobs=1) for k, v in base.items()}
            return base
        else:  # regression
            base = {}
            
            # Ridge (stable et rapide)
            base["Ridge"] = Ridge(alpha=1.0, random_state=self.random_state)
            
            # SGD pour gros datasets
            if not is_multioutput_large:
                base["SGDRegressor"] = SGDRegressor(
                    max_iter=500 if is_large else 1000,
                    tol=1e-3,
                    random_state=self.random_state,
                    verbose=0
                )
            
            # RandomForest (désactivé sur très gros datasets)
            if not is_very_large:
                n_estimators = 50 if is_multioutput_large else (100 if is_large else 200)
                base["RandomForestRegressor"] = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=15 if is_large else 20,
                    min_samples_split=10 if is_large else 2,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
            
            # HistGradientBoosting
            max_iter_gb = 50 if is_multioutput_large else (100 if is_large else 200)
            base["HistGBRegressor"] = HistGradientBoostingRegressor(
                max_iter=max_iter_gb,
                max_depth=8 if is_large else 10,
                random_state=self.random_state,
                verbose=0
            )
            
            # Wrapper MultiOutput pour multi-target
            if n_outputs > 1:
                return {f"MultiOutput_{k}": MultiOutputRegressor(v, n_jobs=1) for k, v in base.items()}
            return base

    def _evaluate_classification(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calcule les métriques pour classification:
        - f1_macro (score principal)
        - f1_weighted 
        - accuracy
        Supporte mono et multi-label.
        """
        try:
            yt = np.asarray(y_true)
            yp = np.asarray(y_pred)
            if yt.ndim == 1 or yt.shape[1] == 1:
                yt = yt.ravel()
                yp = yp.ravel()
                f1m = f1_score(yt, yp, average="macro", zero_division=0)
                f1w = f1_score(yt, yp, average="weighted", zero_division=0)
                acc = accuracy_score(yt, yp)
            else:
                # Multi-label: moyenne par colonne
                f1m = float(np.mean([f1_score(yt[:, j], yp[:, j], average="macro", zero_division=0) for j in range(yt.shape[1])]))
                f1w = float(np.mean([f1_score(yt[:, j], yp[:, j], average="weighted", zero_division=0) for j in range(yt.shape[1])]))
                acc = float(np.mean([accuracy_score(yt[:, j], yp[:, j]) for j in range(yt.shape[1])]))
            return {"main_score": f1m, "f1_macro": f1m, "f1_weighted": f1w, "accuracy": acc}
        except Exception as e:
            self._log(f"[WARN] Erreur métrique classification: {e}")
            return {"main_score": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "accuracy": 0.0}

    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calcule les métriques pour régression:
        - r2 (score principal)
        - mse, mae
        Supporte mono et multi-target.
        """
        try:
            yt = np.asarray(y_true)
            yp = np.asarray(y_pred)
            if yt.ndim == 2 and yt.shape[1] == 1:
                yt = yt.ravel()
                yp = yp.ravel()
            r2 = r2_score(yt, yp, multioutput="uniform_average")
            mse = mean_squared_error(yt, yp)
            mae = mean_absolute_error(yt, yp)
            return {"main_score": r2, "r2": r2, "mse": mse, "mae": mae}
        except Exception as e:
            self._log(f"[WARN] Erreur métrique régression: {e}")
            return {"main_score": 0.0, "r2": 0.0, "mse": 0.0, "mae": 0.0}

    def _check_data_quality(self, X, y_arr) -> Tuple[bool, str]:
        """
        Vérifications qualité données:
        - <80% valeurs manquantes dans X
        - >50% targets valides
        - >=2 classes pour classification
        """
        if isinstance(X, pd.DataFrame):
            nan_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
            if nan_ratio > 0.8:
                return False, f"Trop de valeurs manquantes ({nan_ratio*100:.1f}%)"
        
        if isinstance(y_arr, np.ndarray):
            if y_arr.ndim == 1:
                valid_targets = ~pd.isna(y_arr)
            else:
                valid_targets = ~pd.isna(y_arr).any(axis=1)
            
            valid_ratio = valid_targets.sum() / len(y_arr)
            if valid_ratio < 0.5:
                return False, f"Trop peu de targets valides ({valid_ratio*100:.1f}%)"
        
        if self.task_type == "classification":
            col0 = y_arr[:, 0] if (y_arr.ndim > 1) else y_arr
            col0 = col0[~pd.isna(col0)]
            if len(np.unique(col0)) < 2:
                return False, "Une seule classe dans les targets"
        
        return True, "OK"

    def fit(self, base_path: str) -> "AutoML":
        """
        Entraînement complet AutoML:
        1. Chargement données
        2. Détection type de tâche
        3. Vérification qualité
        4. Pré-traitement
        5. Entraînement + évaluation tous modèles
        6. Sélection meilleur modèle
        """
        start_overall = time.time()
        
        try:
            self._log_section("Chargement des données")
            self._log(f"[AutoML] Dataset : {base_path}")
            
            # Chargement et alignement X/y
            X, y, types = load_dataset(base_path)
            y_arr = y.values if isinstance(y, pd.DataFrame) else np.asarray(y)
            
            if len(X) != len(y_arr):
                m = min(len(X), len(y_arr))
                self._log(f"[AutoML][WARN] Alignement X/y: {m} échantillons")
                X = X.iloc[:m].reset_index(drop=True) if hasattr(X, "iloc") else X[:m]
                y_arr = y_arr[:m]
            
            # Stockage données complètes
            self._X_full = X
            self._y_full = y_arr
            self._log(f"[AutoML] X shape : {X.shape}, y shape : {y_arr.shape}")
            
            # Auto-détection type de tâche
            if self.task_type is None:
                self.task_type = self._infer_task_type(y_arr)
            self._log(f"[AutoML] Type de tâche : {self.task_type}")
            
            # Vérification qualité données
            is_valid, message = self._check_data_quality(X, y_arr)
            if not is_valid:
                raise ValueError(f"Données invalides: {message}")
            
            self._log_section("Pré-traitement des données")
            prep = DataPreprocessor(types)
            self._prep = prep
            
            # Préparation des données selon stratégie CV
            if self.cv_folds == 1:
                # Split train/validation simple
                X_train, X_val, y_train, y_val = prep.split(
                    X, y_arr, test_size=self.test_size, seed=self.random_state, task_type=self.task_type
                )
                self.preprocessor = prep.preprocessor
                
                # Flatten si mono-output
                if isinstance(y_train, np.ndarray) and y_train.ndim == 2 and y_train.shape[1] == 1:
                    y_train = y_train.ravel()
                    y_val = y_val.ravel()
                n_outputs = y_train.shape[1] if (isinstance(y_train, np.ndarray) and y_train.ndim > 1) else 1
            else:
                # Pré-traitement pour CV (fit sur toutes les données)
                prep._categorize_features(types, X)
                self.preprocessor = prep._build_preprocessor()
                if prep._sparse_detected:
                    X = prep._convert_sparse_to_numeric(X)
                n_outputs = y_arr.shape[1] if (isinstance(y_arr, np.ndarray) and y_arr.ndim > 1) else 1
            
            # Sélection modèles adaptés
            models = self._get_models(self.task_type, n_outputs, X.shape[0], X.shape[1])
            
            # Reset résultats
            self.models_results_.clear()
            self.best_model_ = None
            self.best_model_name_ = None
            self.best_score_ = None
            self.metrics_ = None
            
            self._log_section("Entraînement des modèles")
            self._log(f"[AutoML] Modèles candidats : {list(models.keys())}")
            if n_outputs > 50:
                self._log(f"[AutoML] Multi-output large ({n_outputs} outputs) -> SGD/LogReg désactivés")
            
            # Suppression warnings sklearn
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            successful_models = 0
            
            # Boucle sur tous les modèles
            for name, model in models.items():
                # Respect budget temps
                if self.time_budget_sec is not None and (time.time() - start_overall) > self.time_budget_sec:
                    self._log("[AutoML] Budget temps atteint -> arrêt.")
                    break
                
                self._log(f"\n[AutoML] -> Modèle : {name}")
                model_start = time.time()
                
                # Pipeline preprocessing + modèle
                pipe = Pipeline([
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ])
                
                try:
                    if self.cv_folds == 1:
                        # Entraînement/validation simple
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_val)
                        
                        if self.task_type == "classification":
                            metrics = self._evaluate_classification(y_val, y_pred)
                        else:
                            metrics = self._evaluate_regression(y_val, y_pred)
                        score = float(metrics["main_score"])
                    else:
                        # Validation croisée
                        if self.task_type == "classification" and (n_outputs == 1):
                            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                        else:
                            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                        
                        scoring = "f1_macro" if self.task_type == "classification" else "r2"
                        y_cv = y_arr
                        if self.task_type == "classification" and y_arr.ndim == 2 and y_arr.shape[1] == 1:
                            y_cv = y_arr.ravel()
                        
                        scores = cross_val_score(pipe, X, y_cv, cv=cv, scoring=scoring, n_jobs=1)
                        score = float(scores.mean())
                        metrics = {"main_score": score, scoring: score}
                    
                    successful_models += 1
                    elapsed = time.time() - model_start
                    self.models_results_.append({"name": name, "model": pipe, "metrics": metrics})
                    
                    # Mise à jour meilleur modèle
                    if self.best_score_ is None or score > self.best_score_:
                        self.best_score_ = score
                        self.best_model_ = pipe
                        self.best_model_name_ = name
                        self.metrics_ = metrics
                    
                    self._log(f"[AutoML] Score = {score:.4f} (temps: {elapsed:.1f}s)")
                    
                except Exception as e:
                    self._log(f"[AutoML][WARN] Modèle {name} a échoué: {str(e)[:100]}")
                    continue
            
            # Vérification au moins 1 modèle réussi
            if self.best_model_ is None:
                raise RuntimeError(f"Aucun modèle n'a pu être entraîné. Testé {len(models)} modèles, {successful_models} réussis.")
            
            self._log_section("Résultat")
            self._log(f"[AutoML] Meilleur modèle : {self.best_model_name_} | score={self.best_score_:.4f}")
            self._log(f"[AutoML] Temps total: {time.time() - start_overall:.1f}s")
            
            return self
            
        except Exception as e:
            self._log(f"[AutoML][ERROR] Erreur fatale: {e}")
            raise

    def refit_full_data(self) -> "AutoML":
        """
        Ré-entraîne le meilleur modèle sur 100% des données 
        (après sélection par fit()).
        """
        if self.best_model_ is None:
            raise RuntimeError("fit() doit être appelé avant refit_full_data().")
        if self._X_full is None or self._y_full is None:
            raise RuntimeError("Données complètes non disponibles.")
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor non disponible.")
        
        y_full = self._y_full
        if isinstance(y_full, np.ndarray) and y_full.ndim == 2 and y_full.shape[1] == 1:
            y_full = y_full.ravel()
        
        self._log_section("Refit sur les données")
        
        X_full = self._X_full
        if hasattr(self, '_prep') and self._prep is not None:
            if self._prep._sparse_detected:
                X_full = self._prep._convert_sparse_to_numeric(X_full)
        
        self.best_model_.fit(X_full, y_full)
        return self

    def eval(self, X_test: Optional[pd.DataFrame] = None, y_test: Optional[np.ndarray] = None) -> Dict[str, object]:
        """
        Évaluation sur données de test.
        Si pas de test -> retourne infos modèle.
        """
        if self.best_model_ is None:
            raise RuntimeError("fit() d'abord.")
        if X_test is None or y_test is None:
            return {"best_model": self.best_model_name_, "task_type": self.task_type, "metrics": self.metrics_}
        
        y_pred = self.best_model_.predict(X_test)
        if self.task_type == "classification":
            metrics = self._evaluate_classification(y_test, y_pred)
        else:
            metrics = self._evaluate_regression(y_test, y_pred)
        return {"best_model": self.best_model_name_, "task_type": self.task_type, "metrics": metrics}

    def predict(self, X_new) -> np.ndarray:
        """Prédictions sur nouvelles données."""
        if self.best_model_ is None:
            raise RuntimeError("fit() d'abord.")
        
        # Gestion sparse features si détectées
        if hasattr(self, '_prep') and self._prep is not None and self._prep._sparse_detected:
            if not isinstance(X_new, pd.DataFrame):
                X_new = pd.DataFrame(X_new)
            X_new = self._prep._convert_sparse_to_numeric(X_new)
        
        return self.best_model_.predict(X_new)

    def predict_proba(self, X_new) -> np.ndarray:
        """Probabilités (classification uniquement)."""
        if self.best_model_ is None:
            raise RuntimeError("fit() d'abord.")
        
        # Gestion sparse features
        if hasattr(self, '_prep') and self._prep is not None and self._prep._sparse_detected:
            if not isinstance(X_new, pd.DataFrame):
                X_new = pd.DataFrame(X_new)
            X_new = self._prep._convert_sparse_to_numeric(X_new)
        
        model = self.best_model_.named_steps["model"]
        if hasattr(model, "predict_proba"):
            return self.best_model_.predict_proba(X_new)
        raise AttributeError(f"Le modèle {self.best_model_name_} ne supporte pas predict_proba().")

    def save(self, path: str) -> None:
        """Sauvegarde modèle + métadonnées."""
        payload = {
            "best_model": self.best_model_,
            "best_model_name": self.best_model_name_,
            "task_type": self.task_type,
            "metrics": self.metrics_,
            "random_state": self.random_state,
            "prep": self._prep,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "AutoML":
        """Chargement modèle sauvegardé."""
        payload = joblib.load(path)
        obj = AutoML(task_type=payload["task_type"], random_state=payload["random_state"], verbose=False)
        obj.best_model_ = payload["best_model"]
        obj.best_model_name_ = payload["best_model_name"]
        obj.metrics_ = payload["metrics"]
        obj._prep = payload.get("prep", None)
        return obj
