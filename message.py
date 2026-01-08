import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from dataio import load_dataset
from preprocessing import DataPreprocessor


class AutoML:
    """
    AutoML simple et réutilisable pour les datasets du Challenge Machine Learning.

    Fonctionnalités :
    - Chargement des données (.data, .type, .solution) via dataio.load_dataset
    - Pré-traitement automatique (imputation, encodage, normalisation) via DataPreprocessor
    - Détection du type de tâche (classification ou régression)
    - Gestion du multi-output (plusieurs colonnes de sortie)
    - Entraînement de plusieurs modèles et sélection du meilleur selon une métrique principale
      * Classification : F1 macro
      * Régression : R²
    - Évaluation du meilleur modèle avec plusieurs métriques
    """

    def __init__(
        self,
        task_type: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        task_type : {"classification", "regression"} ou None
            Si None, le type est détecté automatiquement à partir de y.
        test_size : float
            Proportion du jeu de validation (par défaut 0.2).
        random_state : int
            Graine de reproductibilité.
        verbose : bool
            Si True, affiche des logs détaillés pendant l'entraînement.
        """
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        self.preprocessor = None
        self.best_model_ = None
        self.best_model_name_: str | None = None
        self.best_score_: float | None = None
        self.metrics_: dict | None = None
        self.models_results_: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Helpers d'affichage                                               #
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _log_section(self, title: str) -> None:
        if not self.verbose:
            return
        bar = "=" * (len(title) + 4)
        print(f"\n{bar}\n  {title}\n{bar}")

    # ------------------------------------------------------------------ #
    #  Détection du type de tâche                                       #
    # ------------------------------------------------------------------ #
    def _infer_task_type(self, y) -> str:
        """
        Détecte automatiquement si la tâche est de classification ou de régression.

        Règle simple :
        - Si y est entier et avec peu de valeurs distinctes -> classification
        - Sinon -> régression
        """
        y_flat = y.values if isinstance(y, pd.DataFrame) else np.asarray(y)

        # Si multi-output, on inspecte la première colonne
        if y_flat.ndim > 1:
            col0 = y_flat[:, 0]
        else:
            col0 = y_flat

        # On enlève les NaN (IMPUTE...)
        col0 = col0[~pd.isna(col0)]

        # Cas pathologique : colonne vide
        if col0.size == 0:
            return "regression"

        uniques = np.unique(col0)

        # Si toutes les valeurs sont entières et peu nombreuses -> classification
        all_int = np.all(np.equal(np.mod(uniques, 1), 0))
        if all_int and len(uniques) <= 20:
            return "classification"

        return "regression"

    # ------------------------------------------------------------------ #
    #  Définition des modèles                                           #
    # ------------------------------------------------------------------ #
    def _get_models(self, task_type: str, n_outputs: int) -> dict:
        """
        Retourne un dictionnaire {nom: modèle} selon le type de tâche
        et le nombre de sorties (multi-output ou non).
        """
        models: dict[str, object] = {}

        if task_type == "classification":
            base_classifiers = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000, n_jobs=None
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=self.random_state,
                ),
                "GradientBoostingClassifier": GradientBoostingClassifier(
                    random_state=self.random_state
                ),
            }

            if n_outputs > 1:
                # Multi-output classification
                for name, clf in base_classifiers.items():
                    models[f"MultiOutput_{name}"] = MultiOutputClassifier(clf)
            else:
                models = base_classifiers

        else:  # regression
            base_regressors = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=self.random_state,
                ),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    random_state=self.random_state
                ),
            }

            if n_outputs > 1:
                # Multi-output regression
                for name, reg in base_regressors.items():
                    models[f"MultiOutput_{name}"] = MultiOutputRegressor(reg)
            else:
                models = base_regressors

        return models

    # ------------------------------------------------------------------ #
    #  Évaluation d'un modèle                                           #
    # ------------------------------------------------------------------ #
    def _evaluate_classification(self, y_true, y_pred) -> dict:
        """
        Calcule les métriques pour la classification.
        - F1 macro (métrique principale)
        - F1 weighted
        - Accuracy
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # Gestion du multi-output : moyenne des scores sur chaque sortie
        if y_true_arr.ndim == 1 or y_true_arr.shape[1] == 1:
            f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro")
            f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted")
            acc = accuracy_score(y_true_arr, y_pred_arr)
        else:
            f1_macro_list = []
            f1_weighted_list = []
            acc_list = []
            for j in range(y_true_arr.shape[1]):
                yt = y_true_arr[:, j]
                yp = y_pred_arr[:, j]
                f1_macro_list.append(f1_score(yt, yp, average="macro"))
                f1_weighted_list.append(f1_score(yt, yp, average="weighted"))
                acc_list.append(accuracy_score(yt, yp))
            f1_macro = float(np.mean(f1_macro_list))
            f1_weighted = float(np.mean(f1_weighted_list))
            acc = float(np.mean(acc_list))

        return {
            "main_score": f1_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": acc,
        }

    def _evaluate_regression(self, y_true, y_pred) -> dict:
        """
        Calcule les métriques pour la régression.
        - R² (métrique principale)
        - MSE
        - MAE
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        r2 = r2_score(y_true_arr, y_pred_arr, multioutput="uniform_average")
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        mae = mean_absolute_error(y_true_arr, y_pred_arr)

        return {
            "main_score": r2,
            "r2": r2,
            "mse": mse,
            "mae": mae,
        }

    # ------------------------------------------------------------------ #
    #  Méthode principale d'entraînement                                #
    # ------------------------------------------------------------------ #
    def fit(self, base_path: str) -> "AutoML":
        """
        Entraîne automatiquement plusieurs modèles sur le dataset spécifié
        et garde le meilleur modèle selon la métrique principale.

        Parameters
        ----------
        base_path : str
            Chemin de base vers le dataset, par ex :
            "/info/corpus/ChallengeMachineLearning/data_A/data_A"
        """
        self._log_section("Chargement des données")
        self._log(f"[AutoML] Dataset : {base_path}")

        # 1) Chargement des données
        X, y, types = load_dataset(base_path)

        if isinstance(y, pd.DataFrame):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        self._log(f"[AutoML] X shape : {X.shape}, y shape : {y_arr.shape}")

        # 2) Détection du type de tâche si non fourni
        if self.task_type is None:
            self.task_type = self._infer_task_type(y_arr)
        self._log(f"[AutoML] Type de tâche : {self.task_type}")

        # 3) Préprocesseur
        self._log_section("Pré-traitement des données")
        prep = DataPreprocessor(types)
        self.preprocessor = prep.preprocessor

        # 4) Split train / validation
        X_train, X_val, y_train, y_val = prep.split(
            X, y_arr, test_size=self.test_size, seed=self.random_state
        )
        self._log(
            f"[AutoML] Split train/val : "
            f"X_train={X_train.shape}, X_val={X_val.shape}"
        )

        n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1

        # 5) Définition des modèles candidats
        models = self._get_models(self.task_type, n_outputs)
        self.models_results_.clear()
        self.best_model_ = None
        self.best_model_name_ = None
        self.best_score_ = None
        self.metrics_ = None

        self._log_section("Entraînement des modèles")
        self._log(f"[AutoML] Nombre de modèles candidats : {len(models)}")

        # 6) Boucle sur les modèles
        for name, model in models.items():
            self._log(f"\n[AutoML] → Modèle en cours : {name}")

            pipeline = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            # Évaluation
            if self.task_type == "classification":
                metrics = self._evaluate_classification(y_val, y_pred)
            else:
                metrics = self._evaluate_regression(y_val, y_pred)

            main_score = metrics["main_score"]
            self._log(f"[AutoML] Score principal ({self.task_type}) = {main_score:.4f}")

            # Sauvegarde des résultats
            self.models_results_.append(
                {
                    "name": name,
                    "model": pipeline,
                    "metrics": metrics,
                }
            )

            # Mise à jour du meilleur modèle
            if self.best_score_ is None or main_score > self.best_score_:
                self.best_score_ = main_score
                self.best_model_ = pipeline
                self.best_model_name_ = name
                self.metrics_ = metrics

        # Résumé final
        self._log_section("Résultat de la recherche de modèle")
        self._log(f"[AutoML] Meilleur modèle : {self.best_model_name_}")
        self._log(f"[AutoML] Score principal : {self.best_score_:.4f}")
        if self.metrics_ is not None:
            for k, v in self.metrics_.items():
                if k == "main_score":
                    continue
                self._log(f"[AutoML] {k} : {v:.4f}")

        return self

    # ------------------------------------------------------------------ #
    #  Évaluation du meilleur modèle                                     #
    # ------------------------------------------------------------------ #
    def eval(self, X_test=None, y_test=None) -> dict:
        """
        Évalue le meilleur modèle.

        - Si X_test, y_test sont fournis : évaluation sur un jeu de test.
        - Sinon : retourne simplement les métriques déjà calculées sur validation.

        Returns
        -------
        dict : métriques calculées.
        """
        if self.best_model_ is None:
            raise RuntimeError(
                "Le modèle n'a pas encore été entraîné. Appelez fit() d'abord."
            )

        # Si pas de données de test fournies, on retourne les métriques de validation
        if X_test is None or y_test is None:
            self._log_section("Évaluation (jeu de validation)")
            self._log(
                "[AutoML] Aucune donnée de test fournie. "
                "Retour des métriques calculées sur la validation."
            )
            return {
                "best_model": self.best_model_name_,
                "task_type": self.task_type,
                "metrics": self.metrics_,
            }

        # Sinon, on prédit sur X_test
        self._log_section("Évaluation (jeu de test externe)")
        y_pred = self.best_model_.predict(X_test)

        if self.task_type == "classification":
            metrics = self._evaluate_classification(y_test, y_pred)
        else:
            metrics = self._evaluate_regression(y_test, y_pred)

        self._log("[AutoML] Métriques sur le jeu de test :")
        for k, v in metrics.items():
            if k == "main_score":
                continue
            self._log(f"  - {k} : {v:.4f}")

        return {
            "best_model": self.best_model_name_,
            "task_type": self.task_type,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------ #
    #  Prédiction sur de nouvelles données                               #
    # ------------------------------------------------------------------ #
    def predict(self, X_new):
        """
        Prédit les sorties pour de nouvelles données X_new.

        - X_new doit avoir la même structure que X utilisée pendant fit().
        - Le pré-traitement est appliqué automatiquement via le pipeline.

        Returns
        -------
        np.ndarray : prédictions du meilleur modèle.
        """
        if self.best_model_ is None:
            raise RuntimeError(
                "Aucun modèle entraîné. Appelez fit() avant predict()."
            )

        self._log("[AutoML] Prédiction sur de nouvelles données...")
        y_pred = self.best_model_.predict(X_new)
        return y_pred


# ---------------------------------------------------------------------- #
#  Exemple d'utilisation directe                                        #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    base_path = "/info/corpus/ChallengeMachineLearning/data_H/data_H"

    automl = AutoML(task_type=None, verbose=True)
    automl.fit(base_path)

    results = automl.eval()
    print("\n[AutoML] Résultats récapitulatifs :")
    print(results)

